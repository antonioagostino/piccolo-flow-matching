from pathlib import Path
import argparse
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from src.utils.download_cifar import get_cifar10
from src.utils.sampling import integrate
from src.flow import FlowMatchingModel
from src.train import load_training_config, validate_device

DEFAULT_NFE_STEPS = [1, 2, 5, 10, 25, 50]
DEFAULT_GUIDANCE = [0.0, 1.0, 3.0]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate samples with a trained Flow-Matching model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training.yaml"),
        help="Path to the YAML training config.",
    )
    parser.add_argument(
        "--checkpoint",
        choices=["best", "last"],
        default="best",
        help="Which checkpoint to load from the config's checkpoint_dir.",
    )
    parser.add_argument(
        "--mode",
        choices=["nfe", "guidance"],
        default="nfe",
        help="'nfe' sweeps the number of integration steps and plots the error against a "
             "high-NFE reference; 'guidance' sweeps the CFG strength at a fixed step count.",
    )
    parser.add_argument(
        "--nfe-steps",
        type=int,
        nargs="+",
        default=None,
        help=f"Step counts to sweep in 'nfe' mode (default: {' '.join(map(str, DEFAULT_NFE_STEPS))}). "
             "In 'guidance' mode only the first value is used, as the fixed step count (default: 25).",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        nargs="+",
        default=None,
        help=f"Guidance strengths to sweep in 'guidance' mode (default: {' '.join(map(str, DEFAULT_GUIDANCE))}). "
             "In 'nfe' mode only the first value is used, as the fixed strength (default: 0.0).",
    )
    parser.add_argument(
        "--reference-nfe",
        type=int,
        default=200,
        help="Step count of the reference sample the 'nfe' mode error is measured against.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of samples generated per swept value. Also the number of images per grid row.",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="Class to generate. When omitted, it is asked interactively.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("generated.png"),
        help="Where to save the sample grid. In 'nfe' mode the error plot is saved alongside it.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Never open a plot window: the 'nfe' error plot is only written to file.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = load_training_config(args.config)
    train_set, _ = get_cifar10(config["data_dir"])
    device = validate_device(config["device"])
    flow_matching_model = FlowMatchingModel(
            backbone=config["backbone"],
            backbone_config_file=config["backbone_config"],
            embedding_dim=config["embedding_dim"],
            num_classes=len(train_set.classes),
            device=device
    )
    checkpoint_path = config["checkpoint_dir"] / f"checkpoint_{args.checkpoint}_{config["backbone"]}.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    flow_matching_model.load_state_dict(checkpoint["ema_model_state_dict"])
    flow_matching_model.eval()

    class_to_generate = args.class_id if args.class_id is not None else int(input("Choose a class: "))
    class_tensor = torch.full((args.batch_size,), class_to_generate, device=device, dtype=torch.long)

    x_0 = torch.randn((args.batch_size, *train_set[0][0].shape), device=device)

    if args.mode == "nfe":
        nfe_steps = args.nfe_steps if args.nfe_steps is not None else DEFAULT_NFE_STEPS
        w = args.guidance[0] if args.guidance is not None else 0.0 # Guidance strenght
        x_1_hat_reference = integrate(flow_matching_model, x_0, class_tensor, args.reference_nfe, w)
        samples = [integrate(flow_matching_model, x_0, class_tensor, steps, w) for steps in nfe_steps]

        err = np.array([((sample - x_1_hat_reference) ** 2).mean().item() for sample in samples])
        n = np.array(nfe_steps, dtype=np.float32)
        # Anchor the reference slope on the third-to-last point, when there are enough of them
        anchor = max(len(nfe_steps) - 3, 0)
        plt.loglog(n, err, marker='o')
        plt.loglog(n, err[anchor] * (n / n[anchor]) ** -2, '--', label='~1/N²')
        plt.legend()
        plt.title("Error against NFE steps")
        plt.xlabel("NFE steps")
        plt.ylabel("Error against the reference sample")
        plot_path = args.output.with_name(f"{args.output.stem}_nfe_error{args.output.suffix or '.png'}")
        plt.savefig(plot_path)
        print(f"Error plot saved to {plot_path}")
        if not args.headless:
            plt.show()

        one_step = samples[0]
        last = samples[-1]
        print(one_step.std(dim=0).mean().item(), last.std(dim=0).mean().item())
    else:
        guidance = args.guidance if args.guidance is not None else DEFAULT_GUIDANCE
        nfe_steps = args.nfe_steps[0] if args.nfe_steps is not None else 25
        samples = [integrate(flow_matching_model, x_0, class_tensor, nfe_steps, w) for w in guidance]

    x_1_hat = torch.cat(samples)
    print(x_1_hat.mean().item(), x_1_hat.std().item(), x_1_hat.min().item(), x_1_hat.max().item())

    x_1_hat_rescaled = (x_1_hat.clamp(-1, 1) + 1.0) / 2.0
    grid = torchvision.utils.make_grid(x_1_hat_rescaled, nrow=args.batch_size, padding=2, pad_value=1.0)
    torchvision.utils.save_image(grid, args.output)
    print(f"Samples saved to {args.output}")
