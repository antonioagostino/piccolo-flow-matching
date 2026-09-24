import torch
from torch import nn
from torch.nn import functional as F

from src.utils import build_patch_grid_ids, sinusoidal_2d_positional_embedding

NUM_CHANNELS = 3

def adaLN_modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift

def unpatchify(x, patch_size: int, out_channels: int):
    c = out_channels
    p = patch_size
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1]

    x = x.reshape(x.shape[0], h, w, p, p, c)
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(x.shape[0], c, h * p, w * p)  # (B, C, I, I)
    return imgs

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 sequence_length: int,
                 n_heads: int,
                 head_size: int):
        super().__init__()
        self.n_heads = n_heads
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.head_size = head_size
        self.qkv_proj = nn.Linear(embedding_dim,
                                  embedding_dim * 3,
                                  bias=False)
        self.output_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self,
                embeddings: torch.Tensor) -> torch.Tensor:
        B, T, _ = embeddings.shape
        H = self.n_heads
        D = self.embedding_dim
        q, k, v = self.qkv_proj(embeddings).split([D, D, D], dim=-1)
        q = q.view(B, T, H, self.head_size).permute(0, 2, 1, 3)
        k = k.view(B, T, H, self.head_size).permute(0, 2, 1, 3)
        v = v.view(B, T, H, self.head_size).permute(0, 2, 1, 3)

        outputs = F.scaled_dot_product_attention(q, k, v)
        # [B, H, T, HS] → [B, T, H, HS] → [B, T, D]
        outputs = outputs.transpose(1, 2).contiguous().view(B, T, self.embedding_dim)
        y = self.output_proj(outputs)

        return y
    
class FFN(nn.Module):
    def __init__(self,
                 input_embedding_dim: int,
                 hidden_embedding_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_embedding_dim,
                                hidden_embedding_dim)
        self.linear_out = nn.Linear(hidden_embedding_dim,
                                    input_embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> (B, T, D)
        hidden_states = self.linear(x) # (B, T, hidden_dim)
        out = self.linear_out(F.gelu(hidden_states)) # (B, T, D)

        return out

    
class DiTBlocks(nn.Module):
    def __init__(self,
                 n_blocks: int,
                 sequence_length: int,
                 embedding_dim: int,
                 n_heads: int,
                 ffn_hidden_dim: int | None,
                 dropout_rate: float):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.n_blocks = n_blocks
        self.sequence_length = sequence_length
        self.head_size = embedding_dim // n_heads
        self.ffn_hidden_dim = ffn_hidden_dim if ffn_hidden_dim is not None else 4 * embedding_dim
        self.adaLN_mods = nn.ModuleList(
            [nn.Sequential(
                nn.SiLU(),
                nn.Linear(embedding_dim,
                          6 * embedding_dim)
            ) for _ in range(n_blocks)]
        )

        # Zero-init AdaLN blocks (called AdaLN-Zero in the DiT paper)
        for i in range(n_blocks):
            # 1 is the index of the nn.Sequential's Linear layer
            nn.init.zeros_(self.adaLN_mods[i][1].weight)
            nn.init.zeros_(self.adaLN_mods[i][1].bias)

        self.attentions = nn.ModuleList(
            [MultiHeadAttention(
                embedding_dim=embedding_dim,
                sequence_length=sequence_length,
                n_heads=n_heads,
                head_size=self.head_size
            ) for _ in range(n_blocks)]
        )

        self.norms_1 = nn.ModuleList(
            [nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6) 
            for _ in range(n_blocks)]
        )

        self.norms_2 = nn.ModuleList(
            [nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6) 
            for _ in range(n_blocks)]
        )

        self.ffns = nn.ModuleList(
            [FFN(embedding_dim,
                 self.ffn_hidden_dim) 
            for _ in range(n_blocks)]
        )

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
       # x -> (B, T, D), conditioning -> (B, D)
        _, T, _ = x.shape
        def run_block(x: torch.Tensor, conditioning: torch.Tensor, block_idx: int) -> torch.Tensor:
            gate_msa, shift_msa, scale_msa, gate_mlp, shift_mlp, scale_mlp = self.adaLN_mods[block_idx](conditioning).chunk(6, dim=-1)
            attn_out = self.attentions[block_idx](
                adaLN_modulate(self.norms_1[block_idx](x), shift_msa[:, None, :], scale_msa[:, None, :])
            )
            x = x + F.dropout(attn_out, self.dropout_rate, training=self.training) * gate_msa[:, None, :]
            ffn_out = self.ffns[block_idx](adaLN_modulate(self.norms_2[block_idx](x), shift_mlp[:, None, :], scale_mlp[:, None, :]))
            return x + F.dropout(ffn_out, self.dropout_rate, training=self.training) * gate_mlp[:, None, :]

        for i in range(self.n_blocks):
            x = run_block(x, conditioning, i)

        return x  # (B, T, D)
    
class DiffusionTransformer(nn.Module):
    def __init__(self,
                 image_height: int,
                 image_width: int,
                 n_encoder_blocks: int,
                 patch_size: int,
                 embedding_dim: int,
                 n_heads: int,
                 ffn_hidden_dim: int | None,
                 dropout_rate: float):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.n_encoder_blocks = n_encoder_blocks
        self.n_heads = n_heads
        self.ffn_hidden_dim = ffn_hidden_dim
        self.patch_size = patch_size
        self.patch_embedding = nn.Conv2d(
            in_channels=NUM_CHANNELS,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.n_patches_h = self.image_height // patch_size
        self.n_patches_w = self.image_width // patch_size
        self.n_patches = self.n_patches_h * self.n_patches_w
        self.final_decoder = nn.Linear(embedding_dim, patch_size * patch_size * NUM_CHANNELS)
        self.dit_blocks = DiTBlocks(
            n_blocks=n_encoder_blocks,
            sequence_length=self.n_patches,
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            ffn_hidden_dim=ffn_hidden_dim,
            dropout_rate=dropout_rate,
        )

        h_ids, w_ids = build_patch_grid_ids(self.n_patches_h, self.n_patches_w, device="cpu")
        self.register_buffer("pos_embeddings", sinusoidal_2d_positional_embedding(h_ids, w_ids, embedding_dim), persistent=False)

        self.final_adaLN_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim,
                      2 * embedding_dim)
        )

        # Zero-init AdaLN block (called AdaLN-Zero in the DiT paper)
        # 1 is the index of the nn.Sequential's Linear layer
        nn.init.zeros_(self.final_adaLN_mod[1].weight)
        nn.init.zeros_(self.final_adaLN_mod[1].bias)

        # The final linear decoder is zero-init too
        nn.init.zeros_(self.final_decoder.weight)
        nn.init.zeros_(self.final_decoder.bias)

        self.final_norm = nn.LayerNorm(
            embedding_dim, elementwise_affine=False, eps=1e-6
        )
        
    def forward(self, image: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        assert image.ndim == 4, "Input image must have 4 dimensions."
        assert image.shape[1] == 3, "Input image has invalid number of channels."
        B, _, H, W = image.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            "Image height and width must be multipliers of model's patch size."
        assert H == self.image_height and W == self.image_width, "Resolution not supported."
        assert conditioning.shape[-1] == self.embedding_dim, "Conditioning embedding dim must be " \
            "equal to the Transformer's hidden dim."
        
        # (B, D, H // patch_size, W // patch_size)
        x: torch.Tensor = self.patch_embedding(image)
        # (B, D, N_patches) -> (B, N_patches, D)
        patch_embeddings = x.flatten(2).transpose(1, 2)
        final_scale, final_shift = self.final_adaLN_mod(conditioning).chunk(2, dim=-1)
        out = self.dit_blocks(patch_embeddings + self.pos_embeddings, conditioning)                         # (B, T, D), (B, D)
        normed = adaLN_modulate(self.final_norm(out), final_shift[:, None, :], final_scale[:, None, :])     # (B, T, D)
        out_patches = self.final_decoder(normed)                                                            # (B, T, PxPxC)

        return unpatchify(out_patches, self.patch_size, NUM_CHANNELS)
