"""Waymo two-channel adapter for TULIP 32-to-64 upsampling.

The adapter reuses the existing TULIP backbone and the anisotropic final
head already used by the range-only baseline.  It changes no Transformer,
encoder, decoder, attention, or skip-connection module.
"""
from __future__ import annotations

import torch
from torch import nn

from .range_only_tulip import (
    AnisotropicFinalPatchExpanding,
    pad_width,
    required_padded_width,
)


def masked_l1_2ch(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute channel-mean L1 over valid 2-channel target pixels.

    Args:
        prediction: Float tensor with shape ``[B, 2, 64, W]``.
        target: Float tensor with shape ``[B, 2, 64, W]``.
        valid_mask: Boolean tensor with shape ``[B, 64, W]``.  The same
            spatial validity mask is applied to range and intensity.

    Returns:
        A scalar tensor containing the mean absolute error over valid pixels
        and both channels.

    Raises:
        ValueError: If shapes differ or the mask contains no valid pixels.
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    if prediction.ndim != 4 or prediction.shape[1] != 2:
        raise ValueError("two-channel tensors must be [B,2,H,W]")
    expected_mask_shape = prediction.shape[:1] + prediction.shape[2:]
    if valid_mask.shape != expected_mask_shape:
        raise ValueError("valid mask must be [B,H,W]")
    mask = valid_mask.unsqueeze(1).to(dtype=prediction.dtype)
    count = mask.sum() * prediction.shape[1]
    if count.item() == 0:
        raise ValueError("valid mask contains no pixels")
    return ((prediction - target).abs() * mask).sum() / count


class Waymo32To64TULIP(nn.Module):
    """Adapt an unchanged TULIP backbone to Waymo 32-to-64, 2-channel data.

    The input and target use native Waymo width.  Only the adapter performs
    horizontal padding and output cropping; the backbone receives a width
    that satisfies its patch/merge/window constraints.

    Args:
        backbone: Existing TULIP model constructed with ``in_chans=2`` and
            ``pixel_shuffle=False``.
        native_width: Native Waymo width, normally 2650.
        padded_width: Optional encoder-compatible width.  If omitted, it is
            derived from ``native_width``.
    """

    def __init__(
        self,
        backbone: nn.Module,
        native_width: int = 2650,
        padded_width: int | None = None,
    ) -> None:
        super().__init__()
        if getattr(backbone, "pixel_shuffle", False):
            raise ValueError("Waymo 2-channel adapter requires pixel_shuffle=False")
        decoder_pred = getattr(backbone, "decoder_pred", None)
        if decoder_pred is None or getattr(decoder_pred, "out_channels", None) != 2:
            raise ValueError("backbone.decoder_pred must produce 2 channels")
        if native_width < 1:
            raise ValueError("native_width must be positive")
        self.native_width = native_width
        self.padded_width = padded_width or required_padded_width(native_width)
        if self.padded_width < native_width:
            raise ValueError("padded_width is smaller than native_width")
        self.backbone = backbone
        self.backbone.final_patch_expanding = AnisotropicFinalPatchExpanding(
            int(backbone.embed_dim),
            vertical_scale=2,
            horizontal_scale=4,
            norm_layer=backbone.norm_layer,
        )

    def forward(
        self,
        input_signal: torch.Tensor,
        target_signal: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the two-channel model and return cropped prediction and loss.

        Args:
            input_signal: Float tensor with shape ``[B, 2, 32, 2650]``.
            target_signal: Float tensor with shape ``[B, 2, 64, 2650]``.
            target_mask: Boolean tensor with shape ``[B, 64, 2650]``.

        Returns:
            ``(prediction, loss)`` where prediction has shape
            ``[B, 2, 64, 2650]`` and loss is a finite scalar when inputs are
            finite and the mask has valid pixels.
        """
        if input_signal.ndim != 4 or input_signal.shape[1:] != (
            2, 32, self.native_width
        ):
            raise ValueError(
                f"input must be [B,2,32,{self.native_width}], "
                f"got {tuple(input_signal.shape)}"
            )
        if target_signal.ndim != 4 or target_signal.shape[1:] != (
            2, 64, self.native_width
        ):
            raise ValueError(
                f"target must be [B,2,64,{self.native_width}], "
                f"got {tuple(target_signal.shape)}"
            )
        if target_mask.shape != (
            input_signal.shape[0], 64, self.native_width
        ):
            raise ValueError(
                f"target_mask must be [B,64,{self.native_width}], "
                f"got {tuple(target_mask.shape)}"
            )
        padded_input = pad_width(
            input_signal, self.padded_width, circular=True
        )
        padded_target = pad_width(target_signal, self.padded_width)
        prediction = self.backbone(
            padded_input, padded_target, mc_drop=True
        )
        expected = (
            input_signal.shape[0], 2, 64, self.padded_width
        )
        if prediction.shape != expected:
            raise ValueError(
                f"TULIP output must be {expected}, got {tuple(prediction.shape)}"
            )
        prediction = prediction[..., : self.native_width]
        return prediction, masked_l1_2ch(
            prediction, target_signal, target_mask
        )
