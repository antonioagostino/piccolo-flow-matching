from pathlib import Path
import torch
from torch.utils.flop_counter import FlopCounterMode
from src.flow import FlowMatchingModel

if __name__ == "__main__":
    device = torch.device("cpu")
    EMBEDDING_DIM = 384
    flow_matching_model = FlowMatchingModel(
            backbone="unet",
            backbone_config_file=Path("./configs/unet.yaml"),
            embedding_dim=EMBEDDING_DIM,
            num_classes=10, # only for testing purposes
            device=device
    )
    x = torch.randn(1, 3, 32, 32).to(device)
    t = torch.rand(1).to(device)
    y = torch.tensor([0]).to(device)
    with FlopCounterMode(display=False) as fc:
        flow_matching_model(x, t, y)
    print(fc.get_total_flops() / 1e9, "GFLOPs per campione (forward)")