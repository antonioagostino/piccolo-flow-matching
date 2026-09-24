from typing import Callable
from pathlib import Path
import yaml

import torch
from torch import nn

from src.utils import sinusoidal_time_embedding
from src.unet import UNet
from src.dit import DiffusionTransformer

class ClassEmbedding(nn.Module):
    """Classifier-Free Guidance style Class Embedding (Ho & Salimans 2022)"""
    def __init__(self, num_classes: int,
                 embedding_dim: int,
                 device: torch.device,
                 dropout_prob: float = 0.1):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(num_classes + 1,
                                      embedding_dim,
                                      device=self.device)  # +1 for null token
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if self.training:
            drop_mask = torch.rand(y.shape[0], device=self.device) < self.dropout_prob
            y = torch.where(drop_mask, torch.full_like(y, self.num_classes), y)
        return self.embedding(y)

class TimestepEmbedder(nn.Module):
    def __init__(self,
                 frequency_embedding_dim: int,
                 final_embedding_dim: int,
                 device: torch.device,):
        super().__init__()
        self.frequency_embedding_dim = frequency_embedding_dim
        self.final_embedding_dim = final_embedding_dim
        self.frequency_embedding = sinusoidal_time_embedding
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_dim, self.final_embedding_dim),
            nn.SiLU(),
            nn.Linear(self.final_embedding_dim, self.final_embedding_dim)
        )

        self.to(device)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        frequency_emb = self.frequency_embedding(t, self.frequency_embedding_dim)
        return self.mlp(frequency_emb)

class FlowMatchingModel(nn.Module):
    def __init__(self,
                 backbone: str,
                 backbone_config_file: Path,
                 embedding_dim: int,
                 num_classes: int,
                 device: torch.device):
        super().__init__()
        assert backbone in ["unet", "dit"], f"The Flow-Matching backbone can be only {['unet', 'dit']}"
        assert backbone_config_file.exists(), "Backbone config file not valid."
        self.backbone = backbone
        self.device = device
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        with Path(backbone_config_file).open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)

        assert isinstance(config, dict), "The model configuration file must contain a YAML mapping"
        assert isinstance(config['model'], dict), "The 'model' section must contain a YAML mapping"
        model_selection = config['model']
        self.backbone_model = self.__parse_backbone_config(backbone, model_selection)
        self.class_embedding = ClassEmbedding(self.num_classes,
                                              self.embedding_dim,
                                              self.device)
        self.timestep_embedder: nn.Module | Callable[[torch.Tensor, int, int], torch.Tensor]
        if backbone == "dit":
            self.timestep_embedder = TimestepEmbedder(self.embedding_dim,
                                                    self.embedding_dim,
                                                    device)
        else:
            self.timestep_embedder = sinusoidal_time_embedding

        self.to(self.device)

    def __parse_backbone_config(self, backbone: str, model_config: dict) -> nn.Module:
        if backbone == "unet":
            down_convs_channels = model_config["down_convs_channels"]
            up_convs_channels = model_config["up_convs_channels"]
            num_norm_groups = model_config["num_norm_groups"]
            output_channels = model_config["output_channels"]
            return UNet(self.embedding_dim, down_convs_channels, up_convs_channels, num_norm_groups, output_channels, self.device)
        elif backbone == "dit":
            image_height = model_config["image_height"]
            image_width = model_config["image_width"]
            n_encoder_blocks = model_config["n_encoder_blocks"]
            patch_size = model_config["patch_size"]
            n_heads = model_config["n_heads"]
            ffn_hidden_dim = model_config["ffn_hidden_dim"]
            dropout_rate = model_config["dropout_rate"]
            return DiffusionTransformer(
                image_height,
                image_width,
                n_encoder_blocks,
                patch_size,
                self.embedding_dim,
                n_heads,
                ffn_hidden_dim,
                dropout_rate
            )
        else:
            raise Exception("Flow-matching backbone not valid")


    def forward(self, x_t, t, y) -> torch.Tensor:
        if self.backbone == "dit":
            time_embedding = self.timestep_embedder(t)
        else:
            time_embedding = self.timestep_embedder(t, self.embedding_dim)
        
        emb = time_embedding + self.class_embedding(y)
        return self.backbone_model(x_t, emb)