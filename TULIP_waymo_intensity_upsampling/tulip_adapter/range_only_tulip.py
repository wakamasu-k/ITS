"""Range-only Waymo adapter for the TULIP 32-to-64 baseline."""
from __future__ import annotations

import torch
from torch import nn
from einops import rearrange


def required_padded_width(width: int, patch_width: int = 4,
                          merge_levels: int = 3,
                          window_width: int = 8) -> int:
    if (width < 1 or patch_width < 1 or merge_levels < 0
            or window_width < 1):
        raise ValueError("width, patch_width and window_width must be positive")
    # Every encoder stage partitions its token map into width-sized windows.
    # Before the deepest stage, patch merging halves the width merge_levels
    # times, so the input width must include this additional factor.
    multiple = patch_width * (2 ** merge_levels) * window_width
    return ((width + multiple - 1) // multiple) * multiple


def pad_width(tensor: torch.Tensor, padded_width: int, *,
              circular: bool = False) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError("tensor must be [B,C,H,W]")
    width = tensor.shape[-1]
    if padded_width < width:
        raise ValueError("padded_width cannot be smaller than tensor width")
    if padded_width == width:
        return tensor
    mode = "circular" if circular else "constant"
    return torch.nn.functional.pad(
        tensor, (0, padded_width - width, 0, 0), mode=mode)


class AnisotropicFinalPatchExpanding(nn.Module):
    """Restore a 1x4 row patch while doubling only LiDAR height."""

    def __init__(self, dim: int, vertical_scale: int = 2,
                 horizontal_scale: int = 4,
                 norm_layer: type[nn.Module] = nn.LayerNorm) -> None:
        super().__init__()
        if vertical_scale < 1 or horizontal_scale < 1:
            raise ValueError("scales must be positive")
        self.dim = dim
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.expand = nn.Linear(
            dim, vertical_scale * horizontal_scale * dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.expand(tensor)
        tensor = rearrange(
            tensor, "b h w (vh vw c) -> b (h vh) (w vw) c",
            vh=self.vertical_scale, vw=self.horizontal_scale, c=self.dim)
        return self.norm(tensor)


def configure_range_only_32_to_64(backbone: nn.Module) -> nn.Module:
    if getattr(backbone, "pixel_shuffle", False):
        raise ValueError("range-only baseline requires non-pixel-shuffle head")
    dim = int(backbone.embed_dim)
    norm_layer = backbone.norm_layer
    backbone.final_patch_expanding = AnisotropicFinalPatchExpanding(
        dim, vertical_scale=2, horizontal_scale=4, norm_layer=norm_layer)
    return backbone


def masked_l1(prediction: torch.Tensor, target: torch.Tensor,
              valid_mask: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError("range tensors must be [B,1,H,W]")
    if valid_mask.shape != prediction.shape[:1] + prediction.shape[2:]:
        raise ValueError("valid mask must be [B,H,W]")
    mask = valid_mask.unsqueeze(1).to(dtype=prediction.dtype)
    count = mask.sum()
    if count.item() == 0:
        raise ValueError("valid mask contains no pixels")
    return ((prediction - target).abs() * mask).sum() / count


class WaymoRangeOnlyTULIP(nn.Module):
    """Pad Waymo width for TULIP and crop predictions back to native width."""

    def __init__(self, backbone: nn.Module, native_width: int = 2650,
                 padded_width: int | None = None) -> None:
        super().__init__()
        self.native_width = native_width
        self.padded_width = padded_width or required_padded_width(native_width)
        if self.padded_width < native_width:
            raise ValueError("padded width is smaller than native width")
        self.backbone = configure_range_only_32_to_64(backbone)

    def forward(self, input_range: torch.Tensor, target_range: torch.Tensor,
                target_mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_range.shape[1:] != (1, 32, self.native_width):
            raise ValueError(
                f"input must be [B,1,32,{self.native_width}], "
                f"got {tuple(input_range.shape)}")
        if target_range.shape[1:] != (1, 64, self.native_width):
            raise ValueError(
                f"target must be [B,1,64,{self.native_width}], "
                f"got {tuple(target_range.shape)}")
        padded_input = pad_width(
            input_range, self.padded_width, circular=True)
        padded_target = pad_width(target_range, self.padded_width)
        prediction = self.backbone(
            padded_input, padded_target, mc_drop=True)
        expected = (
            input_range.shape[0], 1, 64, self.padded_width)
        if prediction.shape != expected:
            raise ValueError(
                f"TULIP output must be {expected}, got {tuple(prediction.shape)}")
        prediction = prediction[..., :self.native_width]
        return prediction, masked_l1(prediction, target_range, target_mask)
