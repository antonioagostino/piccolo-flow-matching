from typing import Callable
import torch

@torch.no_grad()
def integrate(velocity_fn: Callable, x_0: torch.Tensor, n_steps: int, w: float = 0.0) -> torch.Tensor:
    x = x_0.clone()
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device)
        x = x + dt * velocity_fn(x, t, w)
    return x