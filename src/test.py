from pathlib import Path
import random
import torch
from src.utils.sampling import integrate

from src.flow import FlowMatchingModel

def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable

if __name__ == "__main__":
    device = torch.device("mps")
    EMBEDDING_DIM = 384
    SIGMA_MIN = 0.05
    BATCH_SIZE = 1
    flow_matching_model = FlowMatchingModel(
            backbone="dit",
            backbone_config_file=Path("./configs/dit.yaml"),
            embedding_dim=EMBEDDING_DIM,
            num_classes=10, # only for testing purposes
            device=device
    )

    print("Total number of trainable params:", count_params(flow_matching_model))
    for name, child in flow_matching_model.named_children():
        print(name, count_params(child))

    x_batch = []
    y_batch = []
    for _ in range(BATCH_SIZE):
        y = random.randint(0, 4)
        x = torch.rand((3, 32, 32), device=device) * 2 - 1 # [-1, 1] range
        x_batch.append(x)
        y_batch.append(y)

    x = torch.stack(x_batch).to(device)
    y = torch.tensor(y_batch, dtype=torch.long, device=device)

    timesteps = torch.tensor([1.0 for _ in range(BATCH_SIZE)], device=device)

    x_1 = x # Follow Lipman's paper notation
    x_0 = torch.randn_like(x_1)
    broad_t = timesteps.view(-1, 1, 1, 1)
    x_ts = (1.0 - (1.0 - SIGMA_MIN) * broad_t) * x_0 + broad_t * x_1
    target = x_1 - (1 - SIGMA_MIN) * x_0
    target_eq21 = (x_1 - (1 - SIGMA_MIN) * x_ts) / (1.0 - (1 - SIGMA_MIN) * broad_t)

    v_theta = flow_matching_model(x_ts, timesteps, y)
    assert v_theta.shape == x_ts.shape
    assert torch.all(v_theta == 0)
    torch.testing.assert_close(x_1 + SIGMA_MIN * x_0, x_ts)
    torch.testing.assert_close(target, target_eq21)

    def oracle(x_t, t, y):
        return x_1 - (1 - SIGMA_MIN) * x_0

    x = x_0.clone()
    n_steps = 1
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((BATCH_SIZE,), i * dt)
        x = x + dt * oracle(x, t, y)

    torch.testing.assert_close(x, x_1 + SIGMA_MIN * x_0)