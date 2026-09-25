from typing import Any
from pathlib import Path
from datetime import datetime
from importlib.metadata import version
import argparse
import json
import tempfile

import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from cleanfid import fid

from src.utils.download_cifar import get_cifar10
from src.utils.sampling import integrate
from src.flow import FlowMatchingModel
from src.train import load_training_config, validate_device

# Reference: clean-fid's precomputed Inception statistics of the full CIFAR-10 train split
# (50,000 images), stored as cifar10_clean_train_32.npz.
FID_DATASET_NAME = "cifar10"
FID_DATASET_RES = 32
FID_DATASET_SPLIT = "train"
FID_MODE = "clean"
FID_REFERENCE_NUM_IMAGES = 50_000

def to_uint8_images(x: torch.Tensor) -> np.ndarray:
    """Map a (B, 3, H, W) batch in [-1, 1] to (B, H, W, 3) uint8 RGB in [0, 255].

    clean-fid reads each image file with PIL (.convert("RGB") -> uint8 HWC) and computes the CIFAR-10
    reference statistics from the uint8 dataset images, so samples are quantized the same way.
    This is the exact inverse of the loader's ToTensor + Normalize(0.5, 0.5): p = (x + 1) * 127.5.
    Rounding is required, a plain cast truncates values like 254.99999 down to 254.
    """
    x = (x.detach().float().clamp(-1.0, 1.0) + 1.0) * 127.5
    return x.round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

def save_batch(x: torch.Tensor,
               labels: torch.Tensor,
               sample_dir: Path,
               start_index: int,
               grid_images: dict[int, list[np.ndarray]],
               grid_per_class: int) -> None:
    """Convert a batch to uint8, write it as lossless PNGs and keep the first images of each class for the grid."""
    images = to_uint8_images(x)
    for offset, (image, label) in enumerate(zip(images, labels.tolist())):
        Image.fromarray(image).save(sample_dir / f"{start_index + offset:06d}.png")
        if len(grid_images[label]) < grid_per_class:
            grid_images[label].append(image)

def save_grid(grid_images: dict[int, list[np.ndarray]], path: Path) -> None:
    """One row per class, built from the same uint8 images the FID is computed on."""
    n_columns = min(len(images) for images in grid_images.values())
    rows = [torch.from_numpy(np.stack(grid_images[c][:n_columns])) for c in sorted(grid_images)]
    images = torch.cat(rows).permute(0, 3, 1, 2).float() / 255.0
    grid = torchvision.utils.make_grid(images, nrow=n_columns, padding=2, pad_value=1.0)
    torchvision.utils.save_image(grid, path)

def load_ema_weights(model: FlowMatchingModel, checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    ema_weights = checkpoint["ema_model_state_dict"]
    # The EMA holds only the parameters: any parameter missing from it would silently keep its init value
    parameter_names = {name for name, _ in model.named_parameters()}
    if set(ema_weights) != parameter_names:
        raise ValueError(
            f"EMA weights do not match the model parameters. Missing: {sorted(parameter_names - set(ema_weights))}, "
            f"unexpected: {sorted(set(ema_weights) - parameter_names)}"
        )
    state_dict = model.state_dict()
    state_dict.update(ema_weights)
    model.load_state_dict(state_dict, strict=True)
    return {k: v for k, v in checkpoint.items() if not isinstance(v, dict)}

def generate_samples(model: FlowMatchingModel,
                     num_samples: int,
                     num_classes: int,
                     image_shape: tuple[int, ...],
                     n_steps: int,
                     w: float,
                     batch_size: int,
                     seed: int,
                     device: torch.device,
                     sample_dir: Path,
                     grid_images: dict[int, list[np.ndarray]],
                     grid_per_class: int) -> dict[str, float]:
    # Dedicated CPU generator: labels are drawn first, then the noise batch after batch. CPU randn
    # draws the same stream regardless of how it is chunked, so samples do not depend on batch_size.
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(num_classes).repeat_interleave(num_samples // num_classes)
    labels = labels[torch.randperm(num_samples, generator=generator)]

    n_clipped = 0
    running_sum = 0.0
    running_sq_sum = 0.0
    for start in tqdm(range(0, num_samples, batch_size), desc="\033[1mGenerating\033[0m", unit=" batches"):
        y = labels[start:start + batch_size].to(device)
        x_0 = torch.randn((y.shape[0], *image_shape), generator=generator).to(device)
        x_1_hat = integrate(model, x_0, y, n_steps, w)
        n_clipped += int((x_1_hat.abs() > 1.0).sum())
        running_sum += float(x_1_hat.sum())
        running_sq_sum += float((x_1_hat ** 2).sum())
        save_batch(x_1_hat, y, sample_dir, start, grid_images, grid_per_class)

    n_values = num_samples * int(np.prod(image_shape))
    mean = running_sum / n_values
    return {
        "mean": mean,
        "std": float(np.sqrt(max(running_sq_sum / n_values - mean ** 2, 0.0))),
        "fraction_clipped": n_clipped / n_values,
    }

def save_real_samples(val_set: torchvision.datasets.CIFAR10,
                      num_samples: int,
                      num_classes: int,
                      batch_size: int,
                      seed: int,
                      sample_dir: Path,
                      grid_images: dict[int, list[np.ndarray]],
                      grid_per_class: int) -> None:
    """Class-balanced, seeded subset of the validation set, sent through the same conversion as the samples."""
    generator = torch.Generator().manual_seed(seed)
    targets = torch.tensor(val_set.targets)
    per_class = num_samples // num_classes
    indices = []
    for c in range(num_classes):
        class_indices = (targets == c).nonzero().squeeze(1)
        if per_class > len(class_indices):
            raise ValueError(f"Class {c} has only {len(class_indices)} validation images, {per_class} requested")
        indices.append(class_indices[torch.randperm(len(class_indices), generator=generator)[:per_class]])
    indices = torch.cat(indices).tolist()

    for start in tqdm(range(0, num_samples, batch_size), desc="\033[1mReal images\033[0m", unit=" batches"):
        x_batch, y_batch = zip(*(val_set[i] for i in indices[start:start + batch_size]))
        save_batch(torch.stack(x_batch), torch.tensor(y_batch), sample_dir, start, grid_images, grid_per_class)

def compute_fid(sample_dir: Path, num_samples: int, device: torch.device, batch_size: int, num_workers: int) -> float:
    n_files = len(list(sample_dir.glob("*.png")))
    if n_files != num_samples:
        raise RuntimeError(f"Expected {num_samples} images in {sample_dir}, found {n_files}")
    return float(fid.compute_fid(
        fdir1=str(sample_dir),
        dataset_name=FID_DATASET_NAME,
        dataset_res=FID_DATASET_RES,
        dataset_split=FID_DATASET_SPLIT,
        mode=FID_MODE,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        use_dataparallel=False,
    ))

def evaluate(args: argparse.Namespace) -> None:
    config = load_training_config(args.config)
    backbone = args.backbone if args.backbone is not None else config["backbone"]
    if args.backbone_config is not None:
        backbone_config = args.backbone_config
    elif backbone == config["backbone"]:
        backbone_config = config["backbone_config"]
    else:
        raise ValueError(f"--backbone {backbone} differs from the training config, pass --backbone-config too")

    real_vs_real = args.real_vs_real
    if not real_vs_real and args.checkpoint is None and not args.untrained:
        raise ValueError("Pass --checkpoint, --untrained or --real-vs-real")
    num_samples = args.num_samples if args.num_samples is not None else (10_000 if real_vs_real else 50_000)

    _, val_set = get_cifar10(config["data_dir"])
    num_classes = len(val_set.classes)
    image_shape = tuple(val_set[0][0].shape)
    if num_samples <= 0 or num_samples % num_classes != 0:
        raise ValueError(f"--num-samples must be a positive multiple of {num_classes}, got {num_samples}")

    device = validate_device(args.device if args.device is not None else config["device"])
    fid_device = validate_device(args.fid_device) if args.fid_device is not None else device
    nfe = args.n_steps * (1 if args.guidance == 0 else 2)

    if real_vs_real:
        run_name = f"real_val_n{num_samples}_seed{args.seed}"
    else:
        source = args.checkpoint.stem if args.checkpoint is not None else f"untrained_{backbone}"
        run_name = f"{source}_n{num_samples}_steps{args.n_steps}_w{args.guidance:g}_seed{args.seed}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{run_name}.json"
    grid_path = args.output_dir / f"{run_name}_grid.png"

    results: dict[str, Any] = {
        "fid": None,
        "evaluation": "real_vs_real" if real_vs_real else "model",
        "num_samples": num_samples,
        "samples_per_class": num_samples // num_classes,
        "seed": args.seed,
    }
    if real_vs_real:
        results["source"] = "CIFAR-10 test split (the loader's val_set: ToTensor + Normalize, no augmentation)"
    else:
        results.update({
            "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
            "weights": "ema" if args.checkpoint is not None else "random_init",
            "backbone": backbone,
            "backbone_config": str(backbone_config),
            "embedding_dim": config["embedding_dim"],
            "n_steps": args.n_steps,
            "guidance_w": args.guidance,
            "nfe": nfe,
            "solver": "euler",
            "precision": "float32",
            "generation_device": str(device),
        })
    results.update({
        "batch_size": args.batch_size,
        "fid_library": "clean-fid",
        "fid_library_version": version("clean-fid"),
        "fid_reference": {
            "dataset": FID_DATASET_NAME,
            "split": FID_DATASET_SPLIT,
            "num_images": FID_REFERENCE_NUM_IMAGES,
            "resolution": FID_DATASET_RES,
            "mode": FID_MODE,
            "statistics": f"{FID_DATASET_NAME}_{FID_MODE}_{FID_DATASET_SPLIT}_{FID_DATASET_RES}.npz",
        },
        "format_conversion": "clamp(-1, 1) -> round((x + 1) * 127.5) -> uint8 RGB PNG",
        "fid_device": str(fid_device),
        "torch_version": torch.__version__,
        "grid": str(grid_path),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })

    grid_images: dict[int, list[np.ndarray]] = {c: [] for c in range(num_classes)}
    with tempfile.TemporaryDirectory(prefix="fid_samples_") as temporary_dir:
        sample_dir = args.samples_dir if args.samples_dir is not None else Path(temporary_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)
        # clean-fid scans the whole folder recursively: leftovers from another run would be counted
        if any(sample_dir.iterdir()):
            raise ValueError(f"{sample_dir} is not empty")

        if real_vs_real:
            save_real_samples(val_set, num_samples, num_classes, args.batch_size, args.seed,
                              sample_dir, grid_images, args.grid_per_class)
        else:
            torch.manual_seed(args.seed)
            flow_matching_model = FlowMatchingModel(
                backbone=backbone,
                backbone_config_file=backbone_config,
                embedding_dim=config["embedding_dim"],
                num_classes=num_classes,
                device=device
            )
            if args.checkpoint is not None:
                checkpoint_info = load_ema_weights(flow_matching_model, args.checkpoint, device)
                results["checkpoint_optimizer_steps"] = checkpoint_info.get("optimizer_steps")
                results["checkpoint_train_samples_seen"] = checkpoint_info.get("train_samples_seen")
                results["checkpoint_val_loss"] = checkpoint_info.get("val_loss")
            flow_matching_model.eval()
            results["sample_statistics"] = generate_samples(
                flow_matching_model, num_samples, num_classes, image_shape, args.n_steps, args.guidance,
                args.batch_size, args.seed, device, sample_dir, grid_images, args.grid_per_class,
            )

        save_grid(grid_images, grid_path)
        results["fid"] = compute_fid(sample_dir, num_samples, fid_device, args.fid_batch_size, args.num_workers)

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=2)

    print("\n\033[1mFID evaluation\033[0m")
    for key, value in results.items():
        print(f"  {key}: {value}")
    print(f"Results saved to {json_path}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute the FID of a Flow-Matching model against CIFAR-10 with clean-fid.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training.yaml"),
        help="Path to the YAML training config (data_dir, device, embedding_dim and default backbone).",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to evaluate. Its EMA weights are always used.",
    )
    source.add_argument(
        "--untrained",
        action="store_true",
        help="Evaluate a freshly initialized model, without loading any checkpoint.",
    )
    source.add_argument(
        "--real-vs-real",
        action="store_true",
        help="Sanity check: FID of real validation images against the train reference, "
             "through the same format conversion used for the samples.",
    )
    parser.add_argument(
        "--backbone",
        choices=["unet", "dit"],
        default=None,
        help="Backbone to build (default: the one in the training config).",
    )
    parser.add_argument(
        "--backbone-config",
        type=Path,
        default=None,
        help="Backbone YAML config (default: the one in the training config).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of images, a multiple of the number of classes (default: 50000, 10000 with --real-vs-real).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=50,
        help="Euler integration steps.",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=0.0,
        help="CFG strength w. With w != 0 every step costs 2 network evaluations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed of the initial noise, of the label order and of the real-image subset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=250,
        help="Generation batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Generation device (default: the one in the training config).",
    )
    parser.add_argument(
        "--fid-device",
        type=str,
        default=None,
        help="Inception feature extraction device (default: the generation device). Use 'cpu' as a fallback.",
    )
    parser.add_argument(
        "--fid-batch-size",
        type=int,
        default=100,
        help="Inception feature extraction batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Dataloader workers clean-fid uses to resize the images to 299x299. Values > 0 need the 'fork' "
             "start method (clean-fid's resizer is a local function that spawn/forkserver cannot pickle): "
             "not the default on macOS, nor on Linux since Python 3.14.",
    )
    parser.add_argument(
        "--grid-per-class",
        type=int,
        default=10,
        help="Images per class (one row per class) in the saved sample grid.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/fid"),
        help="Where the JSON results and the sample grid are written.",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=None,
        help="Keep the PNG images here (must be empty). By default they go to a temporary directory.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    evaluate(parse_args())
