import torch
from torch import nn
from torch.nn import functional as F

def sinusoidal_2d_positional_embedding(h_ids: torch.Tensor,
                                       w_ids: torch.Tensor,
                                       embedding_dim: int,
                                       max_period: int = 10000) -> torch.Tensor:
    assert embedding_dim % 2 == 0, "embedding_dim must be even to be divided"
    dim_per_axis = embedding_dim // 2
    row_emb = sinusoidal_time_embedding(h_ids, dim_per_axis, max_period)  # (T, dim_axis)
    col_emb = sinusoidal_time_embedding(w_ids, dim_per_axis, max_period)  # (T, dim_axis)
    return torch.cat([row_emb, col_emb], dim=-1)  # (T, embedding_dim)

def sinusoidal_time_embedding(t: torch.Tensor,
                              embedding_dim: int,
                              max_period: int = 10000) -> torch.Tensor:
    t = t * 1000.0      # Fix for using DDPM-like embeddings
    half = embedding_dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(max_period, dtype=torch.float32)) * torch.arange(half, dtype=torch.float32) / half
    ).to(t.device)
    args = t[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

def build_patch_grid_ids(n_patches_h: int, n_patches_w: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    h_ids = torch.arange(n_patches_h, device=device).repeat_interleave(n_patches_w)
    w_ids = torch.arange(n_patches_w, device=device).repeat(n_patches_h)
    return h_ids, w_ids