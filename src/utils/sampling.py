import torch

from src.flow import FlowMatchingModel

def velocity_fn(flow_matching_model: FlowMatchingModel,
                x: torch.Tensor,
                t: torch.Tensor,
                y: torch.Tensor,
                w: float = 0.0) -> torch.Tensor:
    if w == 0:
        return flow_matching_model(x, t, y)
    else:
        null_y = torch.full_like(y, flow_matching_model.num_classes)
        x_in = torch.cat([x, x])
        t_in = torch.cat([t, t])
        y_in = torch.cat([y, null_y])
        v_theta_c, v_theta_u = flow_matching_model(x_in, t_in, y_in).chunk(2)
        # CFG
        return (1.0 + w) * v_theta_c - w * v_theta_u

@torch.no_grad()
def integrate(flow_matching_model: FlowMatchingModel,
              x_0: torch.Tensor,
              y: torch.Tensor,
              n_steps: int,
              w: float = 0.0) -> torch.Tensor:
    x = x_0.clone()
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device)
        x = x + dt * velocity_fn(flow_matching_model, x, t, y, w)
    return x