from typing import Any, cast
import argparse
import random
import math
import copy
import os
import time
from pathlib import Path
import yaml

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode
from tqdm import tqdm
import wandb

from src.utils.download_cifar import get_cifar10
from src.flow import FlowMatchingModel

def train(config: dict[str, Any]):
    if not config["data_dir"].is_dir():
        raise ValueError(f"{config['data_dir']} is not a valid data directory")
    train_set, val_set = get_cifar10(config['data_dir'])

    device = validate_device(config["device"])
    amp_dtype = get_supported_weights_precision(device)
    use_amp = device.type in ("cuda", "mps")
    use_grad_scaler = use_amp and amp_dtype == torch.float16
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(config["seed"])

    val_generator = torch.Generator()

    flow_matching_model = FlowMatchingModel(
        backbone=config["backbone"],
        backbone_config_file=config["backbone_config"],
        embedding_dim=config["embedding_dim"],
        num_classes=len(train_set.classes),
        device=device
    )

    # Counted once, on an uncompiled copy. The image shape comes from the raw data: indexing
    # train_set would draw a random flip and shift the training RNG stream.
    forward_gflops = count_forward_gflops(
        config["backbone"],
        config["backbone_config"],
        config["embedding_dim"],
        len(train_set.classes),
        (train_set.data.shape[3], *train_set.data.shape[1:3]),
    )
    print(f"Forward GFLOPs per sample: {forward_gflops:.3f}")

    if config["data_on_device"]:
        # The whole train set as uint8 on the device (~150 MB for CIFAR-10), normalized and flipped per batch
        train_images_on_device = torch.from_numpy(train_set.data).permute(0, 3, 1, 2).contiguous().to(device)
        train_labels_on_device = torch.tensor(train_set.targets, dtype=torch.long, device=device)

    if config["compile_model"]:
        flow_matching_model = cast(FlowMatchingModel, torch.compile(flow_matching_model))

    # When compiled, torch prefixes parameter names with "_orig_mod.". Always refer to the
    # uncompiled module so EMA keys match the ones save_checkpoint writes.
    base_model = getattr(flow_matching_model, "_orig_mod", flow_matching_model)

    ema_weights = {name: param.detach().clone() for name, param in base_model.named_parameters() if param.requires_grad}

    optimizer = torch.optim.AdamW(
        flow_matching_model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_grad_scaler)
    mse_loss = nn.MSELoss()

    optimizer_steps = 0
    start_epoch = 1
    next_log_iteration = config["log_every_iterations"]
    train_samples_seen = 0
    training_time_seconds = 0.0
    best_val_loss = float("inf")
    wandb_run_id: str | None = None

    if config["resume_from"] is not None:
        checkpoint = torch.load(config["resume_from"], map_location=device, weights_only=True)
        base_model.load_state_dict(checkpoint["model_state_dict"])
        ema_weights = checkpoint["ema_model_state_dict"]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        optimizer_steps = checkpoint["optimizer_steps"]
        train_samples_seen = checkpoint["train_samples_seen"]
        if "training_time_seconds" in checkpoint:
            training_time_seconds = checkpoint["training_time_seconds"]
        else:
            print("Warning: the checkpoint has no training time, the time axis restarts from zero")
        best_val_loss = checkpoint["best_val_loss"]
        start_epoch = checkpoint["epochs"]
        wandb_run_id = checkpoint.get("wandb_run_id")
        next_log_iteration = (optimizer_steps // config["log_every_iterations"] + 1) * config["log_every_iterations"]

    if config["resume_from"] is not None and config["snapshot_every_iterations"] is not None:
        # Snapshots past the resumed step belong to a discarded training branch: the ones on the cadence
        # would be overwritten anyway, but an off-cadence final one would otherwise stay in the series
        for snapshot_path in sorted(get_snapshot_dir(config).glob("step_*.pt")):
            if int(snapshot_path.stem.removeprefix("step_")) > optimizer_steps:
                print(f"Removing {snapshot_path}: it belongs to a discarded training branch")
                snapshot_path.unlink()

    initial_train_samples_seen = train_samples_seen
    training_loss = 0
    train_sample_loss = 0
    ema_loss = None
    ema_alpha = 0.98

    if wandb_run_id is None:
        wandb_run_id = config["wandb_resume_id"]

    wandb_current_run = None
    if config["wandb_enabled"]:
        wandb_current_run = wandb.init(
            project=config["wandb_project"],
            name=config["wandb_run_name"],
            mode=config["wandb_mode"],
            id=wandb_run_id,
            resume="must" if wandb_run_id is not None else None,
            config={
                "training": training_config_to_dict(config),
                "model": {**model_config_to_dict(flow_matching_model), "forward_gflops_per_sample": forward_gflops},
            },
        )

    skip_logging_until: int = 0
    if config["wandb_enabled"] and wandb_run_id is not None and wandb_current_run is not None:
        skip_logging_until = int(wandb_current_run.summary.get("train/samples_seen", 0))

    dataset_samples_order = {
        "train": list(range(len(train_set))),
        "val": list(range(len(val_set)))
    }
    dataset_cursor = {
        "train": 0,
        "val": 0
    }

    n_optimizer_steps_per_epoch = max(
        len(train_set) // (config["batch_size"] * config["gradient_accumulation_steps"]),
        1,
    )
    computed_max = config["n_epochs"] * n_optimizer_steps_per_epoch
    effective_max_iterations: int | None = (
        min(config["max_iterations"], computed_max)
        if config["max_iterations"] is not None
        else computed_max
    )
    if config["resume_from"] is not None:
        set_optimizer_learning_rate(
            optimizer,
            get_learning_rate(config, optimizer_steps, effective_max_iterations),
        )

    progress = tqdm(
        desc="\033[1mTraining\033[0m",
        unit=" total samples",
        bar_format="{desc}: {n_fmt}{unit} [elapsed: {elapsed}, {rate_fmt}{postfix}]",
        initial=train_samples_seen,
    )

    if start_epoch > config["n_epochs"]:
        print(f"Skipping training: resumed at epoch {start_epoch} but n_epochs is {config['n_epochs']}")

    training_clock = TrainingClock(device, training_time_seconds)

    for current_epoch in range(start_epoch, config["n_epochs"] + 1):
        random.seed(config["seed"] + current_epoch)
        random.shuffle(dataset_samples_order["train"])
        dataset_cursor["train"] = 0
        if config["data_on_device"]:
            train_order_on_device = torch.tensor(dataset_samples_order["train"], dtype=torch.long, device=device)

        while True:
            if config["max_iterations"] is not None and optimizer_steps >= config["max_iterations"]:
                break

            optimizer.zero_grad()
            accumulated_loss = 0.0
            samples_in_optimizer_step = 0
            epoch_ended = False

            for _ in range(config["gradient_accumulation_steps"]):
                if config["data_on_device"]:
                    # Same semantics as the loader path below: the partial last batch is dropped and ends the epoch
                    if dataset_cursor["train"] + config["batch_size"] > len(train_set):
                        epoch_ended = True
                        break
                    batch_indices = train_order_on_device[dataset_cursor["train"]:dataset_cursor["train"] + config["batch_size"]]
                    dataset_cursor["train"] += config["batch_size"]
                    x, y = get_train_batch_on_device(train_images_on_device, train_labels_on_device, batch_indices)
                else:
                    x_batch = []
                    y_batch = []
                    try:
                        for _ in range(config["batch_size"]):
                            x, y = train_set[dataset_samples_order["train"][dataset_cursor["train"]]]
                            x_batch.append(x)
                            y_batch.append(y)
                            dataset_cursor["train"] += 1
                    except IndexError:
                        epoch_ended = True
                        break

                    x = torch.stack(x_batch).to(device)
                    y = torch.tensor(y_batch, dtype=torch.long, device=device)

                # TODO: implement importance sampling
                timesteps = torch.rand((config["batch_size"],), device=device)

                x_1 = x # Follow Lipman's paper notation
                x_0 = torch.randn_like(x_1)
                broad_t = timesteps.view(-1, 1, 1, 1)
                x_ts = (1.0 - (1.0 - config["sigma_min"]) * broad_t) * x_0 + broad_t * x_1
                target = x_1 - (1 - config["sigma_min"]) * x_0
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    v_theta = flow_matching_model(x_ts, timesteps, y)
                    loss = mse_loss(v_theta, target)

                scaler.scale(loss / config["gradient_accumulation_steps"]).backward()
                accumulated_loss += float(loss.detach().item())
                samples_in_optimizer_step += x.shape[0]

            if samples_in_optimizer_step:
                optimizer_steps += 1
                learning_rate = get_learning_rate(config, optimizer_steps, effective_max_iterations)
                set_optimizer_learning_rate(optimizer, learning_rate)
                if config["max_grad_norm"] is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(flow_matching_model.parameters(), max_norm=config["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()

                # Update EMA weights
                decay = min(config["ema_decay"], (1.0 + optimizer_steps) / (10.0 + optimizer_steps))
                with torch.no_grad():
                    for name, param in base_model.named_parameters():
                        if param.requires_grad:
                            ema_weights[name].mul_(decay).add_(param.data, alpha=1.0 - decay)

                training_loss = accumulated_loss / config["gradient_accumulation_steps"]
                train_samples_seen += samples_in_optimizer_step
                train_sample_loss += training_loss * samples_in_optimizer_step
                avg_train_loss = train_sample_loss / (train_samples_seen - initial_train_samples_seen)
                ema_loss = (
                    training_loss if ema_loss is None
                    else ema_alpha * ema_loss + (1 - ema_alpha) * training_loss
                )
                progress.update(samples_in_optimizer_step)
                progress.set_postfix(loss=f"{ema_loss:.4f}", lr=f"{learning_rate:.2e}")

                # The clock starts after the first optimizer step of this run, where torch.compile compiles
                # the model; every later pause is closed within the same step, so it is running from then on
                if not training_clock.running:
                    training_clock.start()

                # Saved after this step's EMA update, so it holds the EMA weights at optimizer_steps
                if config["snapshot_every_iterations"] is not None and optimizer_steps % config["snapshot_every_iterations"] == 0:
                    training_clock.pause()
                    save_snapshot(config, ema_weights, base_model.num_classes, optimizer_steps, train_samples_seen,
                                  current_epoch, training_clock.elapsed_seconds, forward_gflops)
                    training_clock.start()

                if optimizer_steps >= next_log_iteration and train_samples_seen > skip_logging_until:
                    if config["wandb_enabled"] and wandb_current_run is not None:
                        wandb_current_run.log(
                            data={
                                "train/loss": training_loss,
                                "train/loss_ema": ema_loss,
                                "train/avg_loss": avg_train_loss,
                                "train/samples_seen": train_samples_seen,
                                "train/optimizer_steps": optimizer_steps,
                                "train/learning_rate": learning_rate,
                            },
                            step=train_samples_seen,
                        )
                    while next_log_iteration <= optimizer_steps:
                        next_log_iteration += config["log_every_iterations"]

                if optimizer_steps % config["val_every_iterations"] == 0:
                    # Validation and checkpoint saving are excluded from the training time
                    training_clock.pause()
                    print("\nStarting validation...")
                    validation_loss = 0
                    dataset_cursor["val"] = 0
                    val_iterations = 0
                    # Backup of current training weights to evaluate the EMA weights only
                    # and then restore the training weights.
                    current_model_state_dict = copy.deepcopy(base_model.state_dict())
                    try:
                        ema_state = copy.deepcopy(current_model_state_dict)
                        ema_state.update(ema_weights)
                        base_model.load_state_dict(ema_state)
                        del ema_state
                        flow_matching_model.eval()
                        val_generator.manual_seed(config["seed"])
                        val_progress = tqdm(
                            desc="\033[1mValidation\033[0m",
                            unit=" total samples",
                            bar_format="{desc}: {n_fmt}{unit} [elapsed: {elapsed}, {rate_fmt}{postfix}]",
                        )
                        while True:
                            if config["val_max_iterations"] is not None and val_iterations >= config["val_max_iterations"]:
                                break
                            x_batch = []
                            y_batch = []
                            try:
                                for _ in range(config["batch_size"]):
                                    x, y = val_set[dataset_samples_order["val"][dataset_cursor["val"]]]
                                    x_batch.append(x)
                                    y_batch.append(y)
                                    dataset_cursor["val"] += 1
                            except IndexError:
                                break

                            val_iterations += 1
                            x = torch.stack(x_batch).to(device)
                            y = torch.tensor(y_batch, dtype=torch.long, device=device)
                
                            timesteps = torch.rand((config["batch_size"],), generator=val_generator).to(device)

                            x_1 = x # Follow Lipman's paper notation
                            x_0 = torch.randn(x_1.shape, generator=val_generator).to(device)
                            broad_t = timesteps.view(-1, 1, 1, 1)
                            x_ts = (1.0 - (1.0 - config["sigma_min"]) * broad_t) * x_0 + broad_t * x_1
                            target = x_1 - (1 - config["sigma_min"]) * x_0
                            with torch.no_grad():
                                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                                    v_theta = flow_matching_model(x_ts, timesteps, y)
                                    loss = mse_loss(v_theta, target)
                                validation_loss += loss.detach()
                                val_progress.update(config["batch_size"])
                                val_progress.set_postfix(loss=f"{validation_loss / val_iterations:.4f}")
                        val_progress.close()
                        if val_iterations == 0:
                            raise ValueError(
                                "No validation batch was evaluated: check that val_max_iterations "
                                "is greater than 0 and that the validation set holds at least "
                                f"batch_size ({config["batch_size"]}) samples"
                            )
                        validation_loss = float(validation_loss / val_iterations)
                    finally:
                        base_model.load_state_dict(current_model_state_dict)
                        flow_matching_model.train()

                    if validation_loss < best_val_loss:
                        best_val_loss = validation_loss
                        save_checkpoint(
                            path=config["checkpoint_dir"] / f"checkpoint_best_{config["backbone"]}.pt",
                            model=flow_matching_model,
                            ema_weights=ema_weights,
                            optimizer=optimizer,
                            scaler=scaler,
                            optimizer_steps=optimizer_steps,
                            train_samples_seen=train_samples_seen,
                            train_loss=training_loss,
                            epochs=current_epoch,
                            val_loss=validation_loss,
                            best_val_loss=best_val_loss,
                            training_time_seconds=training_clock.elapsed_seconds,
                            wandb_run_id=wandb_current_run.id if wandb_current_run is not None else None,
                        )

                    save_checkpoint(
                        path=config["checkpoint_dir"] / f"checkpoint_last_{config["backbone"]}.pt",
                        model=flow_matching_model,
                        ema_weights=ema_weights,
                        optimizer=optimizer,
                        scaler=scaler,
                        optimizer_steps=optimizer_steps,
                        train_samples_seen=train_samples_seen,
                        train_loss=training_loss,
                        epochs=current_epoch,
                        val_loss=validation_loss,
                        best_val_loss=best_val_loss,
                        training_time_seconds=training_clock.elapsed_seconds,
                        wandb_run_id=wandb_current_run.id if wandb_current_run is not None else None,
                    )

                    if config["wandb_enabled"] and wandb_current_run is not None and train_samples_seen > skip_logging_until:
                        wandb_current_run.log(
                            data={
                                "val/loss": validation_loss,
                                "val/best_loss": best_val_loss,
                                "train/samples_seen": train_samples_seen,
                                "train/optimizer_steps": optimizer_steps,
                            },
                            step=train_samples_seen,
                        )
                    training_clock.start()

            if epoch_ended:
                if config["wandb_enabled"] and wandb_current_run is not None and config["n_epochs"] > 1 and train_samples_seen > skip_logging_until:
                    wandb_current_run.log(
                        data={
                            "train/epoch": current_epoch,
                            "train/samples_seen": train_samples_seen,
                        },
                        step=train_samples_seen,
                    )
                break

        if config["max_iterations"] is not None and optimizer_steps >= config["max_iterations"]:
            break

    progress.close()
    training_clock.pause()

    # Final snapshot, so the snapshot series always ends at the last step executed by this run
    if (config["snapshot_every_iterations"] is not None
            and train_samples_seen > initial_train_samples_seen
            and optimizer_steps % config["snapshot_every_iterations"] != 0):
        save_snapshot(config, ema_weights, base_model.num_classes, optimizer_steps, train_samples_seen,
                      current_epoch, training_clock.elapsed_seconds, forward_gflops)

    if config["wandb_enabled"] and wandb_current_run is not None and train_samples_seen > 0 and train_samples_seen > skip_logging_until:
        wandb_current_run.log(
            data={
                "train/final_loss": training_loss,
                "train/samples_seen": train_samples_seen,
                "train/optimizer_steps": optimizer_steps,
                "train/learning_rate": get_learning_rate(config, optimizer_steps, effective_max_iterations),
            },
            step=train_samples_seen,
        )

    # Final validation phase
    validation_loss = 0
    val_iterations = 0
    dataset_cursor["val"] = 0
    best_checkpoint_path = config["checkpoint_dir"] / f"checkpoint_best_{config["backbone"]}.pt"
    if not best_checkpoint_path.is_file():
        print(f"Skipping final validation: no best checkpoint at {best_checkpoint_path}")
        if config["wandb_enabled"] and wandb_current_run is not None:
            wandb_current_run.finish()
        return
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=True)
    current_model_state_dict = base_model.state_dict()
    current_model_state_dict.update(checkpoint["ema_model_state_dict"])
    base_model.load_state_dict(current_model_state_dict)
    flow_matching_model.eval()
    val_generator.manual_seed(config["seed"])
    val_progress = tqdm(
        desc="\033[1mValidation\033[0m",
        unit=" total samples",
        bar_format="{desc}: {n_fmt}{unit} [elapsed: {elapsed}, {rate_fmt}{postfix}]",
    )
    while True:
        if config["val_max_iterations"] is not None and val_iterations >= config["val_max_iterations"]:
            break
        x_batch = []
        y_batch = []
        try:
            for _ in range(config["batch_size"]):
                x, y = val_set[dataset_samples_order["val"][dataset_cursor["val"]]]
                x_batch.append(x)
                y_batch.append(y)
                dataset_cursor["val"] += 1
        except IndexError:
            break

        val_iterations += 1
        x = torch.stack(x_batch).to(device)
        y = torch.tensor(y_batch, dtype=torch.long, device=device)

        timesteps = torch.rand((config["batch_size"],), generator=val_generator).to(device)

        x_1 = x # Follow Lipman's paper notation
        x_0 = torch.randn(x_1.shape, generator=val_generator).to(device)
        broad_t = timesteps.view(-1, 1, 1, 1)
        x_ts = (1.0 - (1.0 - config["sigma_min"]) * broad_t) * x_0 + broad_t * x_1
        target = x_1 - (1 - config["sigma_min"]) * x_0
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                v_theta = flow_matching_model(x_ts, timesteps, y)
                loss = mse_loss(v_theta, target)
            validation_loss += loss.detach()
            val_progress.update(config["batch_size"])
            val_progress.set_postfix(loss=f"{validation_loss / val_iterations:.4f}")

    val_progress.close()
    if val_iterations == 0:
        raise ValueError(
            "No validation batch was evaluated: check that val_max_iterations is greater "
            f"than 0 and that the validation set holds at least batch_size ({config["batch_size"]}) samples"
        )
    final_validation_loss = float(validation_loss / val_iterations)
    print(f"Final validation loss: {final_validation_loss:.4f}")

    if config["wandb_enabled"] and wandb_current_run is not None and train_samples_seen > skip_logging_until:
        wandb_current_run.log(
            data={"val/final_loss": final_validation_loss},
            step=train_samples_seen,
        )

    if config["wandb_enabled"] and wandb_current_run is not None:
        wandb_current_run.finish()

def get_supported_weights_precision(device: torch.device) -> torch.dtype:
    """Return the highest-precision dtype supported for AMP on the given device."""
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32

def set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate

def validate_device(desired_device: str) -> torch.device:
    if desired_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        device = torch.device(desired_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available")
        
        return device

def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()

class TrainingClock:
    """Cumulative training time in seconds, paused around the sections excluded from it.

    CUDA and MPS run asynchronously: the device is synchronized at every start and pause, so queued
    work is charged to the section that issued it. Nothing is synchronized between two boundaries.
    """
    def __init__(self, device: torch.device, elapsed_seconds: float = 0.0):
        self.device = device
        self.elapsed_seconds = elapsed_seconds
        self.started_at: float | None = None

    @property
    def running(self) -> bool:
        return self.started_at is not None

    def start(self) -> None:
        synchronize_device(self.device)
        self.started_at = time.perf_counter()

    def pause(self) -> None:
        if self.started_at is not None:
            synchronize_device(self.device)
            self.elapsed_seconds += time.perf_counter() - self.started_at
            self.started_at = None

def count_forward_gflops(backbone: str,
                         backbone_config: Path,
                         embedding_dim: int,
                         num_classes: int,
                         image_shape: tuple[int, ...]) -> float:
    """Forward GFLOPs for one sample, counted on a fresh uncompiled copy of the model built on the CPU.

    FlopCounterMode misses fused attention kernels: on the CPU the math SDPA backend decomposes attention
    into matmuls, so QK^T and AV are counted, while on MPS SDPA runs a fused kernel under no_grad even
    when the math backend is forced. FLOPs do not depend on the device, so the count is always done on
    the CPU. The CPU RNG is forked so building the copy does not shift the training random streams.
    """
    with torch.random.fork_rng(devices=[]):
        model = FlowMatchingModel(backbone, backbone_config, embedding_dim, num_classes, torch.device("cpu"))
    model.eval()
    x = torch.zeros((1, *image_shape))
    t = torch.zeros((1,))
    y = torch.zeros((1,), dtype=torch.long)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as flop_counter:
        model(x, t, y)
    return flop_counter.get_total_flops() / 1e9

def get_train_batch_on_device(images: torch.Tensor,
                              labels: torch.Tensor,
                              batch_indices: torch.Tensor,
                              horizontal_flip_p: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch from the uint8 train set kept on the device, normalized exactly as the loader does
    (ToTensor: float / 255, then Normalize(0.5, 0.5)), with a per-sample horizontal flip."""
    x = images[batch_indices].float().div(255).sub(0.5).div(0.5)
    if horizontal_flip_p > 0:
        flip = torch.rand(x.shape[0], device=x.device) < horizontal_flip_p
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    return x, labels[batch_indices]

def save_checkpoint(
    path: Path,
    model: FlowMatchingModel,
    ema_weights: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    optimizer_steps: int,
    train_loss: float,
    train_samples_seen: int,
    epochs: int,
    val_loss: float,
    best_val_loss: float,
    training_time_seconds: float,
    wandb_run_id: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state_dict = model.state_dict()
    model_state_dict = {k.removeprefix("_orig_mod."): v for k, v in model_state_dict.items()}
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": model_state_dict,
            'ema_model_state_dict': ema_weights,
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "optimizer_steps": optimizer_steps,
            "train_loss": train_loss,
            "train_samples_seen": train_samples_seen,
            "epochs": epochs,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "training_time_seconds": training_time_seconds,
            "wandb_run_id": wandb_run_id,
        },
        temporary_path,
    )
    # Atomic swap, so an interruption cannot leave a corrupted checkpoint in place
    os.replace(temporary_path, path)

def get_snapshot_dir(config: dict[str, Any]) -> Path:
    return config["checkpoint_dir"] / "snapshots" / config["backbone"]

def save_snapshot(
    config: dict[str, Any],
    ema_weights: dict[str, torch.Tensor],
    num_classes: int,
    optimizer_steps: int,
    train_samples_seen: int,
    epochs: int,
    training_time_seconds: float,
    forward_gflops_per_sample: float,
) -> None:
    path = get_snapshot_dir(config) / f"step_{optimizer_steps:07d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "ema_model_state_dict": ema_weights,
            "optimizer_steps": optimizer_steps,
            "train_samples_seen": train_samples_seen,
            "epochs": epochs,
            "backbone": config["backbone"],
            "backbone_config": str(config["backbone_config"]),
            "embedding_dim": config["embedding_dim"],
            "num_classes": num_classes,
            "training_time_seconds": training_time_seconds,
            "forward_gflops_per_sample": forward_gflops_per_sample,
        },
        temporary_path,
    )
    os.replace(temporary_path, path)

def get_learning_rate(
    config: dict[str, Any],
    iteration: int,
    max_iterations: int | None = None,
) -> float:
    assert iteration > 0, "Iteration must be greater than 0"

    lr_warmup = config["lr_warmup_iterations"]
    learning_rate = config["learning_rate"]
    min_learning_rate = config["min_learning_rate"]
    effective_max = max_iterations if max_iterations is not None else config["max_iterations"]

    if lr_warmup > 0 and iteration <= lr_warmup:
        return learning_rate * iteration / lr_warmup

    if effective_max is None:
        return learning_rate

    decay_iterations = effective_max - lr_warmup
    decay_iteration = max(iteration - lr_warmup - 1, 0)
    decay_progress = min(decay_iteration / decay_iterations, 1.0)
    cosine_multiplier = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_learning_rate + cosine_multiplier * (learning_rate - min_learning_rate)

def load_training_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)

    assert isinstance(raw_config, dict), "Invalid YAML mapping"
    training_config = raw_config["training"]
    assert isinstance(training_config, dict), "Invalid YAML mapping"
    wandb_config = training_config["wandb"]
    assert isinstance(wandb_config, dict), "Invalid YAML mapping"

    config: dict[str, Any] = {}

    if (v := training_config.get("data_dir", None)) is None:
        raise ValueError("Missing 'data_dir' in training config file")
    config["data_dir"] = Path(v)

    if (v := training_config.get("backbone", None)) is None:
        raise ValueError("Missing 'backbone' in training config file")
    config["backbone"] = str(v)

    if (v := training_config.get("backbone_config", None)) is None:
        raise ValueError("Missing 'backbone_config' in training config file")
    config["backbone_config"] = Path(v)

    if (v := training_config.get("embedding_dim", None)) is None:
        raise ValueError("Missing 'embedding_dim' in training config file")
    config["embedding_dim"] = int(v)

    if (v := training_config.get("sigma_min", None)) is None:
        raise ValueError("Missing 'sigma_min' in training config file")
    config["sigma_min"] = float(v)

    if (v := training_config.get("ema_decay", None)) is None:
        raise ValueError("Missing 'ema_decay' in training config file")
    config["ema_decay"] = float(v)

    if (v := training_config.get("batch_size", None)) is None:
        raise ValueError("Missing 'batch_size' in training config file")
    config["batch_size"] = int(v)

    if (v := training_config.get("learning_rate", None)) is None:
        raise ValueError("Missing 'learning_rate' in training config file")
    config["learning_rate"] = float(v)

    if (v := training_config.get("min_learning_rate", None)) is None:
        raise ValueError("Missing 'min_learning_rate' in training config file")
    config["min_learning_rate"] = float(v)

    if (v := training_config.get("lr_warmup_iterations", None)) is None:
        raise ValueError("Missing 'lr_warmup_iterations' in training config file")
    config["lr_warmup_iterations"] = int(v)

    v = training_config.get("max_iterations", None)
    config["max_iterations"] = None if v is None else int(v)

    if (v := training_config.get("weight_decay", None)) is None:
        raise ValueError("Missing 'weight_decay' in training config file")
    config["weight_decay"] = float(v)

    config["max_grad_norm"] = training_config.get("max_grad_norm", None)

    if (v := training_config.get("device", None)) is None:
        raise ValueError("Missing 'device' in training config file")
    config["device"] = str(v)

    if (v := training_config.get("compile_model", None)) is None:
        raise ValueError("Missing 'compile_model' in training config file")
    config["compile_model"] = bool(v)

    if (v := training_config.get("gradient_accumulation_steps", None)) is None:
        raise ValueError("Missing 'gradient_accumulation_steps' in training config file")
    config["gradient_accumulation_steps"] = int(v)

    if (v := training_config.get("seed", None)) is None:
        raise ValueError("Missing 'seed' in training config file")
    config["seed"] = int(v)

    if (v := training_config.get("log_every_iterations", None)) is None:
        raise ValueError("Missing 'log_every_iterations' in training config file")
    config["log_every_iterations"] = int(v)

    if (v := training_config.get("val_every_iterations", None)) is None:
        raise ValueError("Missing 'val_every_iterations' in training config file")
    config["val_every_iterations"] = int(v)

    v = training_config.get("val_max_iterations", None)
    config["val_max_iterations"] = None if v is None else int(v)

    if (v := training_config.get("n_epochs", None)) is None:
        raise ValueError("Missing 'n_epochs' in training config file")
    config["n_epochs"] = int(v)

    if (v := training_config.get("checkpoint_dir", None)) is None:
        raise ValueError("Missing 'checkpoint_dir' in training config file")
    config["checkpoint_dir"] = Path(v)

    v = training_config.get("resume_from", None)
    config["resume_from"] = Path(v) if v is not None else None

    v = training_config.get("snapshot_every_iterations", None)
    config["snapshot_every_iterations"] = None if v is None else int(v)
    if config["snapshot_every_iterations"] is not None and config["snapshot_every_iterations"] <= 0:
        raise ValueError("'snapshot_every_iterations' must be positive or null")

    config["data_on_device"] = bool(training_config.get("data_on_device", False))

    if (v := wandb_config.get("enabled", None)) is None:
        raise ValueError("Missing 'enabled' in wandb config")
    config["wandb_enabled"] = bool(v)

    if config["wandb_enabled"]:
        if (v := wandb_config.get("project", None)) is None:
            raise ValueError("Missing 'project' in wandb config")
        config["wandb_project"] = str(v)
        config["wandb_run_name"] = wandb_config.get("run_name", None)
        if (v := wandb_config.get("mode", None)) is None:
            raise ValueError("Missing 'mode' in wandb config")
        config["wandb_mode"] = str(v)
        config["wandb_resume_id"] = wandb_config.get("resume_id", None)
    else:
        config["wandb_project"] = None
        config["wandb_run_name"] = None
        config["wandb_mode"] = None
        config["wandb_resume_id"] = None

    return config


def training_config_to_dict(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "data_dir": str(config["data_dir"]),
        "backbone": str(config["backbone"]),
        "backbone_config": str(config["backbone_config"]),
        "embedding_dim": int(config["embedding_dim"]),
        "sigma_min": float(config["sigma_min"]),
        "ema_decay": float(config["ema_decay"]),
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
        "min_learning_rate": config["min_learning_rate"],
        "lr_warmup_iterations": config["lr_warmup_iterations"],
        "max_iterations": config["max_iterations"],
        "weight_decay": config["weight_decay"],
        "max_grad_norm": config["max_grad_norm"],
        "device": config["device"],
        "compile_model": config["compile_model"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "seed": config["seed"],
        "log_every_iterations": config["log_every_iterations"],
        "val_every_iterations": config["val_every_iterations"],
        "val_max_iterations": config["val_max_iterations"],
        "n_epochs": config["n_epochs"],
        "checkpoint_dir": str(config["checkpoint_dir"]),
        "resume_from": str(config["resume_from"]) if config["resume_from"] is not None else None,
        "snapshot_every_iterations": config["snapshot_every_iterations"],
        "data_on_device": config["data_on_device"],
        "wandb": {
            "enabled": config["wandb_enabled"],
            "project": config["wandb_project"],
            "run_name": config["wandb_run_name"],
            "mode": config["wandb_mode"],
        },
    }

def model_config_to_dict(backbone: FlowMatchingModel) -> dict[str, int | float | None]:
    """Serialize input FlowMatchingModel's config to a dictionary."""
    return {
        "backbone": backbone.backbone,
        "num_classes": backbone.num_classes,
        "embedding_dim": backbone.embedding_dim,
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the piccolo Flow-Matching generative model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training.yaml"),
        help="Path to the YAML training config.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = load_training_config(args.config)
    train(config)