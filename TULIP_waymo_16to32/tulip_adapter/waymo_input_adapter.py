"""Waymo tensor conversion at the TULIP input boundary.

This module does not contain a model, forward pass, loss, or training logic.
It converts raw HWC Waymo 32-line arrays into normalized CHW tensors and
performs the adapter-only circular width padding/cropping.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch


IntensityMode = Literal["none", "clip_1", "log1p"]


def _validate_intensity_mode(mode: str) -> IntensityMode:
    """Validate an intensity transform name."""
    if mode not in {"none", "clip_1", "log1p"}:
        raise ValueError("intensity_mode must be one of: none, clip_1, log1p")
    return mode  # type: ignore[return-value]


def normalize_range_intensity(
    raw_hwc: np.ndarray,
    maximum_range: float,
    intensity_mode: IntensityMode = "none",
) -> np.ndarray:
    """Normalize one raw 32-line HWC sample without changing its geometry.

    Input shape: ``[32,W,2]`` float-compatible array, channel 0 raw range and
    channel 1 raw intensity.
    Output shape: ``[32,W,2]`` float32 array.

    Range is divided by the caller-provided ``maximum_range``.  The value is
    deliberately not fixed in this module.  Invalid/non-positive range pixels
    are set to zero in both channels.  Intensity modes are: ``none`` (raw
    finite value), ``clip_1`` (clip to [0,1]), and ``log1p``.

    Raises:
        ValueError: If shape, maximum range, finiteness, or log1p domain is
            invalid.
    """
    array = np.asarray(raw_hwc, dtype=np.float32)
    if array.ndim != 3 or array.shape[0] != 32 or array.shape[2] != 2:
        raise ValueError(f"Expected raw HWC shape [32,W,2], got {array.shape}.")
    if array.shape[1] <= 0:
        raise ValueError("Raw width must be positive.")
    if not np.isfinite(array).all():
        raise ValueError("Raw range/intensity contains NaN or Inf.")
    if not np.isfinite(maximum_range) or maximum_range <= 0:
        raise ValueError("maximum_range must be a finite positive value.")
    mode = _validate_intensity_mode(intensity_mode)

    raw_range = array[..., 0]
    raw_intensity = array[..., 1]
    valid = raw_range > 0.0
    normalized_range = raw_range / np.float32(maximum_range)
    safe_intensity = np.where(valid, raw_intensity, 0.0)
    if mode == "none":
        normalized_intensity = safe_intensity.copy()
    elif mode == "clip_1":
        normalized_intensity = np.clip(safe_intensity, 0.0, 1.0)
    else:
        if np.any(raw_intensity[valid] < -1.0):
            raise ValueError("log1p intensity is undefined below -1 on valid pixels.")
        normalized_intensity = np.log1p(safe_intensity)
    result = np.stack(
        (
            np.where(valid, normalized_range, 0.0),
            np.where(valid, normalized_intensity, 0.0),
        ),
        axis=-1,
    ).astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("Normalized range/intensity contains NaN or Inf.")
    return result


def hwc_to_chw_tensor(image_hwc: np.ndarray) -> torch.Tensor:
    """Convert a finite HWC 2-channel array to a contiguous float32 tensor.

    Input shape: ``[H,W,2]``.
    Output shape: ``[2,H,W]`` and dtype ``torch.float32``.
    """
    array = np.asarray(image_hwc, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 2:
        raise ValueError(f"Expected HWC shape [H,W,2], got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("HWC image contains NaN or Inf.")
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).to(torch.float32)


def pad_width_for_tulip(tensor: torch.Tensor, target_width: int) -> torch.Tensor:
    """Circularly pad the last dimension to ``target_width``.

    Input shape: CHW or BCHW tensor, with width in the last dimension.
    Output shape: same rank and leading dimensions, last dimension
    ``target_width``.  For 2650→2688, the first 38 columns are appended.

    Raises:
        TypeError: If input is not a tensor.
        ValueError: If rank, width, or target width is invalid.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor.")
    if tensor.ndim not in (3, 4):
        raise ValueError(f"Expected CHW or BCHW tensor, got shape {tuple(tensor.shape)}.")
    if target_width <= 0 or tensor.shape[-1] <= 0:
        raise ValueError("Tensor width and target_width must be positive.")
    current_width = tensor.shape[-1]
    if target_width < current_width:
        raise ValueError(f"target_width {target_width} is smaller than current width {current_width}.")
    extra = target_width - current_width
    if extra == 0:
        return tensor
    indices = torch.arange(extra, device=tensor.device) % current_width
    wrapped = tensor.index_select(-1, indices)
    return torch.cat((tensor, wrapped), dim=-1)


def crop_to_native_width(tensor: torch.Tensor, native_width: int) -> torch.Tensor:
    """Crop only the adapter padding from the last tensor dimension.

    Input shape: CHW, BCHW, or any tensor with a positive last dimension.
    Output shape: same rank and leading dimensions, last dimension
    ``native_width``.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor.")
    if tensor.ndim < 1 or native_width <= 0:
        raise ValueError("native_width must be positive and tensor must have a width dimension.")
    if native_width > tensor.shape[-1]:
        raise ValueError(
            f"native_width {native_width} is larger than current width {tensor.shape[-1]}."
        )
    return tensor[..., :native_width]
