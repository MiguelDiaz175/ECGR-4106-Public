"""Compact Swin Transformer from scratch for CIFAR-100 experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition BHWC feature maps into non-overlapping windows."""

    bsz, height, width, channels = x.shape
    if height % window_size != 0 or width % window_size != 0:
        raise ValueError("bad window_size")
    x = x.view(
        bsz,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, channels)


def window_reverse(
    windows: torch.Tensor,
    window_size: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Reverse window_partition for BHWC feature maps."""

    bsz = int(windows.shape[0] / (height * width / window_size / window_size))
    x = windows.view(
        bsz,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        -1,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(bsz, height, width, -1)


class DropPath(nn.Module):
    """Stochastic depth; identity when drop_prob is zero."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class PatchEmbed(nn.Module):
    """Patch embedding for Swin-style hierarchical features."""

    def __init__(self, image_size: int = 224, patch_size: int = 4, in_channels: int = 3, embed_dim: int = 48):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("bad patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        height, width = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, height, width


class WindowAttention(nn.Module):
    """Window multi-head self-attention with relative position bias."""

    def __init__(self, dim: int, window_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("bad heads")
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        table_size = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b_windows, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(b_windows, tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(tokens, tokens, -1)
        relative_bias = relative_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(b_windows // num_windows, num_windows, self.num_heads, tokens, tokens)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, tokens, tokens)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_windows, tokens, channels)
        x = self.proj(x)
        return self.proj_drop(x)


class SwinBlock(nn.Module):
    """Swin Transformer block with regular or shifted-window attention."""

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = min(window_size, input_resolution[0], input_resolution[1])
        self.shift_size = 0 if min(input_resolution) <= window_size else shift_size
        if self.shift_size >= self.window_size:
            raise ValueError("bad shift_size")

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, self.window_size, num_heads, dropout)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.register_buffer("attn_mask", self._create_mask(), persistent=False)

    def _create_mask(self) -> torch.Tensor | None:
        height, width = self.input_resolution
        if self.shift_size == 0:
            return None
        img_mask = torch.zeros((1, height, width, 1))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        count = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = count
                count += 1
        mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = self.input_resolution
        bsz, length, channels = x.shape
        if length != height * width:
            raise ValueError("bad token_count")
        shortcut = x
        x = self.norm1(x).view(bsz, height, width, channels)
        if self.shift_size > 0:
            shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted = x

        windows = window_partition(shifted, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, channels)
        attn_windows = self.attn(windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, channels)
        shifted = window_reverse(attn_windows, self.window_size, height, width)

        if self.shift_size > 0:
            x = torch.roll(shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted
        x = x.view(bsz, height * width, channels)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """Downsample by merging each 2x2 neighborhood and doubling channels."""

    def __init__(self, input_resolution: tuple[int, int], dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        height, width = self.input_resolution
        bsz, length, channels = x.shape
        if length != height * width:
            raise ValueError("bad token_count")
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError("bad merge_size")
        x = x.view(bsz, height, width, channels)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(bsz, -1, 4 * channels)
        x = self.norm(x)
        x = self.reduction(x)
        return x, height // 2, width // 2


class BasicLayer(nn.Module):
    """A Swin stage made of alternating W-MSA and SW-MSA blocks."""

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        dropout: float,
        downsample: bool,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SwinBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                    dropout=dropout,
                )
                for i in range(depth)
            ]
        )
        self.downsample = PatchMerging(input_resolution, dim) if downsample else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        for block in self.blocks:
            x = block(x)
        if self.downsample is not None:
            x, height, width = self.downsample(x)
        else:
            height, width = self.blocks[-1].input_resolution
        return x, height, width


@dataclass(frozen=True)
class ScratchSwinConfig:
    image_size: int = 224
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 100
    embed_dim: int = 48
    depths: tuple[int, int, int, int] = (2, 2, 2, 2)
    num_heads: tuple[int, int, int, int] = (3, 6, 12, 24)
    window_size: int = 7
    dropout: float = 0.0


class ScratchSwinTransformer(nn.Module):
    """Compact Swin classifier with hierarchical shifted-window attention."""

    def __init__(self, config: ScratchSwinConfig = ScratchSwinConfig()):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(
            config.image_size,
            config.patch_size,
            config.in_channels,
            config.embed_dim,
        )
        resolution = config.image_size // config.patch_size
        layers = []
        dim = config.embed_dim
        for stage_idx, depth in enumerate(config.depths):
            input_resolution = (resolution, resolution)
            layers.append(
                BasicLayer(
                    dim=dim,
                    input_resolution=input_resolution,
                    depth=depth,
                    num_heads=config.num_heads[stage_idx],
                    window_size=config.window_size,
                    dropout=config.dropout,
                    downsample=stage_idx < len(config.depths) - 1,
                )
            )
            if stage_idx < len(config.depths) - 1:
                resolution //= 2
                dim *= 2
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, height, width = self.patch_embed(x)
        for layer in self.layers:
            x, height, width = layer(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


def run_swin_shape_tests() -> dict[str, tuple[int, ...] | bool]:
    """Small synthetic checks for required scratch Swin tensor operations."""

    sample = torch.randn(2, 56, 56, 3)
    windows = window_partition(sample, 7)
    restored = window_reverse(windows, 7, 56, 56)
    if not torch.allclose(sample, restored):
        raise AssertionError("bad window_reverse")

    merging = PatchMerging((56, 56), 48)
    merged, h, w = merging(torch.randn(2, 56 * 56, 48))
    if tuple(merged.shape) != (2, 28 * 28, 96):
        raise AssertionError("bad patch_merging")

    block = SwinBlock(48, (56, 56), num_heads=3, window_size=7, shift_size=3)
    if block.attn_mask is None:
        raise AssertionError("bad shift_mask")
    block_out = block(torch.randn(2, 56 * 56, 48))

    model = ScratchSwinTransformer(
        ScratchSwinConfig(image_size=224, embed_dim=24, num_heads=(3, 6, 12, 24))
    )
    logits = model(torch.randn(2, 3, 224, 224))
    if tuple(logits.shape) != (2, 100):
        raise AssertionError("bad logits")
    return {
        "window_partition": tuple(windows.shape),
        "window_reverse_matches": True,
        "patch_merging": tuple(merged.shape),
        "shifted_block": tuple(block_out.shape),
        "end_to_end_logits": tuple(logits.shape),
    }
