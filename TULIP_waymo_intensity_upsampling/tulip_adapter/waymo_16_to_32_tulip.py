"""Waymo-specific 16-line to 32-line adapter for the existing TULIP model.

The Swin Transformer, encoder, decoder, attention, merging, unmerging, and
skip-connection modules are reused unchanged.  This module only adapts the
Waymo input stem, final anisotropic expansion, padding contract, and loss
shape validation.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .range_only_tulip import AnisotropicFinalPatchExpanding


class WaymoPatchEmbedding(nn.Module):
    """Patch embedding with Waymo's width-circular padding.

    Input shape: ``[B,2,16,2688]``.
    Output shape: ``[B,16,1344,96]`` for the default configuration.
    """

    def __init__(self, source: nn.Module, *, width_pad: int = 3) -> None:
        super().__init__()
        if width_pad < 0:
            raise ValueError("width_pad must be non-negative")
        source_proj = getattr(source, "proj", None)
        if source_proj is None or not isinstance(source_proj, nn.Conv2d):
            raise ValueError("source patch embedding must contain Conv2d proj")
        if source_proj.kernel_size != (1, 8):
            raise ValueError(
                f"source projection must use kernel (1,8), got {source_proj.kernel_size}"
            )
        if source_proj.stride != (1, 2):
            raise ValueError(
                f"source projection must use stride (1,2), got {source_proj.stride}"
            )
        self.patch_size = (1, 2)
        self.circular_padding = True
        self.width_pad = width_pad
        self.proj = nn.Conv2d(
            in_channels=source_proj.in_channels,
            out_channels=source_proj.out_channels,
            kernel_size=(1, 8),
            stride=(1, 2),
            bias=source_proj.bias is not None,
        )
        self.norm = getattr(source, "norm", nn.Identity())
        self.grid_size = (16, 1344)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Embed a padded Waymo tensor.

        Input shape: ``[B,2,16,2688]``.
        Output shape: ``[B,16,1344,96]``.
        """
        if tensor.ndim != 4 or tensor.shape[1] != self.proj.in_channels:
            raise ValueError(
                f"input must be [B,{self.proj.in_channels},H,W], got {tuple(tensor.shape)}"
            )
        if tensor.shape[2] != 16:
            raise ValueError(f"input height must be 16, got {tensor.shape[2]}")
        if tensor.shape[3] != 2688:
            raise ValueError(f"input width must be 2688, got {tensor.shape[3]}")
        tensor = F.pad(tensor, (self.width_pad, self.width_pad, 0, 0), mode="circular")
        tensor = self.proj(tensor)
        tensor = tensor.permute(0, 2, 3, 1).contiguous()
        return self.norm(tensor)


def masked_l1_2ch(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute mean absolute error over valid pixels and both channels.

    Input shapes: prediction/target ``[B,2,32,W]``; mask ``[B,32,W]``.
    Output shape: scalar tensor.
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    if prediction.ndim != 4 or prediction.shape[1] != 2:
        raise ValueError("prediction and target must be [B,2,H,W]")
    if valid_mask.shape != prediction.shape[:1] + prediction.shape[2:]:
        raise ValueError("valid_mask must be [B,H,W]")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("prediction and target must be finite")
    mask = valid_mask.to(dtype=prediction.dtype).unsqueeze(1)
    count = mask.sum() * prediction.shape[1]
    if count.item() == 0:
        raise ValueError("valid_mask contains no valid pixels")
    return ((prediction - target).abs() * mask).sum() / count


class Waymo16To32TULIP(nn.Module):
    """Adapt an existing TULIP backbone to padded Waymo 16→32 tensors.

    Input shape: input/target ``[B,2,16/32,2688]`` and mask ``[B,32,2688]``.
    Output shape: prediction ``[B,2,32,2688]`` and scalar loss.
    """

    def __init__(self, backbone: nn.Module, *, width_pad: int = 3) -> None:
        super().__init__()
        if getattr(backbone, "pixel_shuffle", False):
            raise ValueError("Waymo16To32TULIP requires pixel_shuffle=False")
        decoder_pred = getattr(backbone, "decoder_pred", None)
        if decoder_pred is None or getattr(decoder_pred, "out_channels", None) != 2:
            raise ValueError("backbone.decoder_pred must produce 2 channels")
        if getattr(backbone, "patch_embed", None) is None:
            raise ValueError("backbone must expose patch_embed")
        self.backbone = backbone
        self.backbone.patch_embed = WaymoPatchEmbedding(
            self.backbone.patch_embed, width_pad=width_pad
        )
        self.backbone.final_patch_expanding = AnisotropicFinalPatchExpanding(
            int(backbone.embed_dim),
            vertical_scale=2,
            horizontal_scale=2,
            norm_layer=backbone.norm_layer,
        )
        self.backbone.upscale_factor = 2

    def forward(
        self,
        input_signal: torch.Tensor,
        target_signal: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run TULIP and calculate masked 2-channel L1 loss.

        Input shapes: input ``[B,2,16,2688]``, target ``[B,2,32,2688]``,
        mask ``[B,32,2688]``.
        Output shapes: prediction ``[B,2,32,2688]`` and scalar loss.
        """
        if input_signal.ndim != 4 or tuple(input_signal.shape[1:]) != (2, 16, 2688):
            raise ValueError(f"input must be [B,2,16,2688], got {tuple(input_signal.shape)}")
        if target_signal.ndim != 4 or tuple(target_signal.shape[1:]) != (2, 32, 2688):
            raise ValueError(f"target must be [B,2,32,2688], got {tuple(target_signal.shape)}")
        if tuple(target_mask.shape) != (input_signal.shape[0], 32, 2688):
            raise ValueError(
                f"target_mask must be [B,32,2688], got {tuple(target_mask.shape)}"
            )
        prediction = self.backbone(input_signal, target_signal, mc_drop=True)
        expected = (input_signal.shape[0], 2, 32, 2688)
        if tuple(prediction.shape) != expected:
            raise ValueError(f"TULIP output must be {expected}, got {tuple(prediction.shape)}")
        return prediction, masked_l1_2ch(prediction, target_signal, target_mask)

    @staticmethod
    def crop_native(prediction: torch.Tensor, native_width: int = 2650) -> torch.Tensor:
        """Crop padded output to native Waymo width.

        Input shape: ``[B,2,32,2688]``.
        Output shape: ``[B,2,32,2650]`` by default.
        """
        if native_width < 1 or native_width > prediction.shape[-1]:
            raise ValueError("native_width must be within prediction width")
        return prediction[..., :native_width]


def build_waymo_16_to_32_model(
    tulip_root: Path,
    *,
    padded_width: int = 2688,
) -> Waymo16To32TULIP:
    """Construct a new 2-channel Waymo model from the unchanged TULIP factory.

    Input shape: no tensor input; ``tulip_root`` points to the existing TULIP.
    Output: a model accepting ``[B,2,16,2688]`` and returning ``[B,2,32,2688]``.
    """
    if padded_width != 2688:
        raise ValueError("the current Waymo16To32 adapter requires padded_width=2688")
    root = str(Path(tulip_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from model.tulip import tulip_base

    backbone = tulip_base(
        img_size=(16, padded_width),
        target_img_size=(32, padded_width),
        patch_size=(1, 2),
        in_chans=2,
        window_size=(2, 8),
        pixel_shuffle=False,
        circular_padding=True,
        swin_v2=False,
        log_transform=False,
        patch_unmerging=False,
    )
    return Waymo16To32TULIP(backbone)
