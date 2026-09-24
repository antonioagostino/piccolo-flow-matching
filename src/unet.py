from typing import List
import torch
from torch import nn
from torch.nn import functional as F

class CondConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 embedding_dim: int,
                 device: torch.device,
                 apply_norm: bool = False,
                 num_norm_groups: int = 32,
                 apply_film_cond: bool = False,
                 save_skip_connection: bool = False,
                 apply_max_pooling: bool = False,
                 up_scale_and_concat: bool = False,
                 skip_connections: List[torch.Tensor] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_dim = embedding_dim
        self.device = device
        self.apply_norm = apply_norm
        self.apply_film_cond = apply_film_cond
        self.save_skip_connection = save_skip_connection
        self.apply_max_pooling = apply_max_pooling
        self.up_scale_and_concat = up_scale_and_concat
        self.skip_connections = skip_connections

        if self.apply_max_pooling and self.up_scale_and_concat:
            raise RuntimeError("Upscale and Max-Pooling can never be in the same convolutional layer.")

        if self.save_skip_connection and self.up_scale_and_concat:
            raise RuntimeError("You cannot save skip connections in the upscaling side of a UNet.")

        self.conv = nn.Conv2d(self.in_channels,
                              self.out_channels,
                              3,
                              padding="same")
        if self.up_scale_and_concat:
            self.upscale_conv = nn.Conv2d(self.in_channels,
                                          self.out_channels,
                                          3,
                                          padding="same")
        if self.apply_norm:
            self.norm = nn.GroupNorm(num_norm_groups, self.in_channels)

        if self.apply_film_cond:
            self.film = nn.Linear(self.embedding_dim, 2 * self.in_channels)

    def forward(self, x, embedding):
        if self.up_scale_and_concat:
            x = F.interpolate(x, scale_factor=2)
            x = self.upscale_conv(x)
            x = torch.cat([self.skip_connections.pop(), x], dim=1).to(self.device)

        # GroupNorm + FiLM (only to the second conv block of the stage) + SiLU
        if self.apply_norm:
            x = self.norm(x)

        if self.apply_film_cond:
            film_scale, film_shift = self.film(embedding).chunk(2, dim=1) # 2 x (B, in_channels)
            film_scale = film_scale[:, :, None, None]
            film_shift = film_shift[:, :, None, None]
            x = x * (1 + film_scale) + film_shift

        if self.apply_norm:
            x = F.silu(x)

        x = self.conv(x)

        # Upscale and Max-Pooling can never be in the same conv layer or
        # save skip connection in the up-scaling side of a UNet.
        if self.save_skip_connection:
            assert self.skip_connections is not None, "The list of skip connection is not valid!"
            self.skip_connections.append(x)
        if self.apply_max_pooling:
            x = F.max_pool2d(x, 2)

        return x
        

class UNet(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 down_convs_channels: List[int],
                 up_convs_channels: List[int],
                 num_norm_groups: int,
                 output_channels: int,
                 device: torch.device,):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        self.down_convs_channels = down_convs_channels
        self.up_convs_channels = up_convs_channels

        self.down_and_up_modules = nn.ModuleList()
        self.skip_connections = []

        for i in range(0, len(self.down_convs_channels) - 1):
            save_skip_connection = False
            apply_max_pooling = False
            apply_film_cond = False
            apply_norm = True if i > 0 else False
            
            if i % 2 == 1 and i < (len(self.down_convs_channels) - 2):
                save_skip_connection = True
            if i % 2 == 0 and i > 0:
                apply_max_pooling = True
                apply_film_cond = True

            # Last down conv (the bottleneck)
            if i == len(self.down_convs_channels) - 2:
                apply_film_cond = True


            self.down_and_up_modules.append(CondConv2d(self.down_convs_channels[i],
                                                       self.down_convs_channels[i + 1],
                                                       self.embedding_dim,
                                                       self.device,
                                                       apply_norm,
                                                       num_norm_groups,
                                                       apply_film_cond,
                                                       save_skip_connection,
                                                       apply_max_pooling,
                                                       False,
                                                       self.skip_connections))
        
        for i in range(0, len(self.up_convs_channels) - 1):
            up_scale_and_concat = True if i % 2 == 0 else False
            apply_film_cond = True if i % 2 == 1 else False
            self.down_and_up_modules.append(CondConv2d(self.up_convs_channels[i],
                                                       self.up_convs_channels[i + 1],
                                                       self.embedding_dim,
                                                       self.device,
                                                       apply_norm=True,
                                                       num_norm_groups=num_norm_groups,
                                                       apply_film_cond=apply_film_cond,
                                                       save_skip_connection=False,
                                                       up_scale_and_concat=up_scale_and_concat,
                                                       apply_max_pooling=False,
                                                       skip_connections=self.skip_connections))
        
        
        self.last_conv = nn.Conv2d(64, output_channels, kernel_size=1)
        # Follow ADM / IDDPM paper initialization
        torch.nn.init.zeros_(self.last_conv.weight)
        torch.nn.init.zeros_(self.last_conv.bias)
        self.to(self.device)

    def forward(self, x, embedding):
        for conv_mod in self.down_and_up_modules:
            x = conv_mod(x, embedding)

        return self.last_conv(x)
