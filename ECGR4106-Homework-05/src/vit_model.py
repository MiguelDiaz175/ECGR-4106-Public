"""Vision Transformer from scratch for CIFAR-100."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ViTConfig:
    name: str
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 100
    embed_dim: int = 256
    depth: int = 4
    num_heads: int = 4
    dropout: float = 0.1

    @property
    def mlp_hidden_dim(self) -> int:
        return 4 * self.embed_dim

    def validate(self) -> None:
        if self.image_size % self.patch_size != 0:
            raise ValueError("bad patch_size")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("bad heads")

    def as_dict(self) -> dict:
        data = asdict(self)
        data["mlp_hidden_dim"] = self.mlp_hidden_dim
        return data


VIT_CONFIGS: dict[str, ViTConfig] = {
    "vit_p4_d256_l4_h4": ViTConfig(
        name="vit_p4_d256_l4_h4", patch_size=4, embed_dim=256, depth=4, num_heads=4
    ),
    "vit_p4_d512_l8_h8": ViTConfig(
        name="vit_p4_d512_l8_h8", patch_size=4, embed_dim=512, depth=8, num_heads=8
    ),
    "vit_p8_d256_l4_h4": ViTConfig(
        name="vit_p8_d256_l4_h4", patch_size=8, embed_dim=256, depth=4, num_heads=4
    ),
    "vit_p8_d512_l8_h8": ViTConfig(
        name="vit_p8_d512_l8_h8", patch_size=8, embed_dim=512, depth=8, num_heads=8
    ),
}


class PatchEmbedding(nn.Module):
    """Image to patch-token projection using a Conv2d patchifier."""

    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("bad patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Explicit multi-head self-attention with qkv projections."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("bad heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(bsz, seq_len, dim)
        out = self.proj(out)
        return self.proj_drop(out)


class TransformerEncoderBlock(nn.Module):
    """Pre-norm Transformer encoder block."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_hidden_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Scratch Vision Transformer classifier."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.patch_embed = PatchEmbedding(
            config.image_size,
            config.patch_size,
            config.in_channels,
            config.embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        self.pos_drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    config.embed_dim,
                    config.num_heads,
                    config.mlp_hidden_dim,
                    config.dropout,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if x.size(1) != self.pos_embed.size(1):
            raise ValueError("bad pos_embed")
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])


def build_vit(config_name: str) -> VisionTransformer:
    return VisionTransformer(VIT_CONFIGS[config_name])


def manual_vit_parameter_breakdown(config: ViTConfig) -> dict[str, int]:
    """Manual parameter calculation for the scratch ViT configuration."""

    config.validate()
    patches_per_side = config.image_size // config.patch_size
    num_patches = patches_per_side**2
    d = config.embed_dim
    mlp = config.mlp_hidden_dim
    patch_embedding = d * config.in_channels * config.patch_size * config.patch_size + d
    class_token = d
    positional_embedding = (num_patches + 1) * d
    attention_per_block = (d * 3 * d + 3 * d) + (d * d + d)
    mlp_per_block = (d * mlp + mlp) + (mlp * d + d)
    layer_norm_per_block = 4 * d
    block_total = attention_per_block + mlp_per_block + layer_norm_per_block
    final_norm = 2 * d
    classification_head = d * config.num_classes + config.num_classes
    total = (
        patch_embedding
        + class_token
        + positional_embedding
        + config.depth * block_total
        + final_norm
        + classification_head
    )
    return {
        "patch_embedding": patch_embedding,
        "class_token": class_token,
        "positional_embedding": positional_embedding,
        "attention_projections_all_blocks": config.depth * attention_per_block,
        "mlp_layers_all_blocks": config.depth * mlp_per_block,
        "layer_norms_all_blocks": config.depth * layer_norm_per_block,
        "final_layer_norm": final_norm,
        "classification_head": classification_head,
        "manual_total": total,
    }


def verify_vit_shapes(config: ViTConfig, batch_size: int = 2) -> tuple[torch.Size, int]:
    """Run a synthetic forward pass and return output shape plus token length."""

    model = VisionTransformer(config)
    x = torch.randn(batch_size, config.in_channels, config.image_size, config.image_size)
    logits = model(x)
    expected = (batch_size, config.num_classes)
    if tuple(logits.shape) != expected:
        raise AssertionError("bad logits")
    return logits.shape, model.pos_embed.size(1)
