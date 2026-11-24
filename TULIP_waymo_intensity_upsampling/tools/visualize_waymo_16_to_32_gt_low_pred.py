#!/usr/bin/env python3
"""Create pure Waymo GT, Low, and TULIP Pred comparison images.

This tool operates in LiDAR range-image space. It does not use the camera
oracle mode. Low is placed on the 32-row grid at rows 0, 2, ..., 30, while
GT and Pred use the complete 32-row grid. PNG values are visualization-only;
the raw float32 arrays are saved separately in NPZ.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tulip_adapter.waymo_16_to_32_dataset import Waymo16To32Dataset
from tulip_adapter.waymo_16_to_32_tulip import build_waymo_16_to_32_model


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--tulip-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-role", default="test")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--range-scale-m", type=float, default=75.0)
    parser.add_argument("--intensity-transform", choices=("raw", "log1p"), default="log1p")
    parser.add_argument("--range-vmax", type=float, default=75.0)
    parser.add_argument("--intensity-vmax", type=float, default=20.0)
    parser.add_argument("--intensity-log-vmax", type=float, default=20000.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict:
    """Load a trusted local checkpoint across supported PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def inverse_transform(
    tensor: torch.Tensor,
    *,
    range_scale_m: float,
    intensity_transform: str,
) -> np.ndarray:
    """Convert a model tensor to raw range/intensity float32 values.

    Input shape: [2,H,W] or [B,2,H,W].
    Output shape: same rank and shape, with channel 0 in meters and channel 1
    in raw intensity units.
    """
    values = tensor.detach().cpu().numpy().astype(np.float32, copy=True)
    if values.ndim not in (3, 4) or values.shape[-3] != 2:
        raise ValueError(f"expected [2,H,W] or [B,2,H,W], got {values.shape}")
    values[..., 0, :, :] *= np.float32(range_scale_m)
    if intensity_transform == "log1p":
        values[..., 1, :, :] = np.expm1(values[..., 1, :, :])
    return values


def zero_invalid(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero both channels where a pixel is invalid.

    Input values shape: [2,H,W]. Input mask shape: [H,W].
    Output shape: [2,H,W], dtype float32.
    """
    output = np.asarray(values, dtype=np.float32).copy()
    valid = np.asarray(mask, dtype=bool)
    if output.shape[1:] != valid.shape:
        raise ValueError(f"values/mask shape mismatch: {output.shape} vs {valid.shape}")
    output[:, ~valid] = 0.0
    output[~np.isfinite(output)] = 0.0
    return output


def make_cmap(name: str):
    """Create a colormap with invalid pixels rendered black."""
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad("black")
    cmap.set_under("black")
    return cmap


def save_three_panel(
    values: list[np.ndarray],
    masks: list[np.ndarray],
    titles: list[str],
    *,
    output: Path,
    title: str,
    vmax: float,
    cmap_name: str,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> None:
    """Save a common-scale three-panel image.

    Each input value and mask has shape [H,W]. Output is a PNG image.
    """
    if len(values) != 3 or len(masks) != 3 or len(titles) != 3:
        raise ValueError("three panels are required")
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    cmap = make_cmap(cmap_name)
    image = None
    for axis, value, mask, panel_title in zip(axes, values, masks, titles):
        display = np.asarray(value, dtype=np.float32)
        if transform is not None:
            display = transform(display)
        masked = np.ma.array(display, mask=~np.asarray(mask, dtype=bool))
        image = axis.imshow(
            masked,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
        )
        axis.set_title(panel_title)
        axis.set_xlabel("native azimuth column")
        axis.set_ylabel("32-grid ring row")
    if image is not None:
        figure.colorbar(image, ax=axes, pad=0.01)
    figure.suptitle(title, fontsize=15)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def finite_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    """Calculate finite statistics over mask=True pixels."""
    selected = np.asarray(values, dtype=np.float32)[np.asarray(mask, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return {"count": 0}
    return {
        "count": int(selected.size),
        "min": float(selected.min()),
        "max": float(selected.max()),
        "mean": float(selected.mean()),
        "p50": float(np.percentile(selected, 50.0)),
        "p95": float(np.percentile(selected, 95.0)),
        "p99": float(np.percentile(selected, 99.0)),
    }


def main() -> None:
    """Generate pure GT/Low/Pred range and intensity comparisons."""
    args = parse_args()
    if args.sample_index < 0:
        raise ValueError("sample-index must be non-negative")
    if args.range_scale_m <= 0 or args.range_vmax <= 0:
        raise ValueError("range values must be positive")
    if args.intensity_vmax <= 0 or args.intensity_log_vmax <= 0:
        raise ValueError("intensity display maxima must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    dataset = Waymo16To32Dataset(
        args.index,
        dataset_role=args.dataset_role,
        intensity_transform=args.intensity_transform,
        range_scale_m=args.range_scale_m,
    )
    if args.sample_index >= len(dataset):
        raise IndexError(f"sample-index {args.sample_index} >= dataset length {len(dataset)}")
    sample = dataset[args.sample_index]

    model = build_waymo_16_to_32_model(args.tulip_root).to(device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with torch.inference_mode():
        input_signal = sample["input"].unsqueeze(0).to(device)
        target_signal = sample["target"].unsqueeze(0).to(device)
        target_mask = sample["target_mask"].unsqueeze(0).to(device)
        prediction, _ = model(input_signal, target_signal, target_mask)

    input_raw = inverse_transform(
        input_signal[0],
        range_scale_m=args.range_scale_m,
        intensity_transform=args.intensity_transform,
    )[..., :2650]
    target_raw = inverse_transform(
        target_signal[0],
        range_scale_m=args.range_scale_m,
        intensity_transform=args.intensity_transform,
    )[..., :2650]
    prediction_raw = inverse_transform(
        prediction[0],
        range_scale_m=args.range_scale_m,
        intensity_transform=args.intensity_transform,
    )[..., :2650]
    input_mask = sample["input_mask"].numpy()[..., :2650].astype(bool)
    target_mask_native = sample["target_mask"].numpy()[..., :2650].astype(bool)

    low = np.zeros_like(target_raw, dtype=np.float32)
    low_mask = np.zeros_like(target_mask_native, dtype=bool)
    low[:, 0::2, :] = input_raw
    low_mask[0::2, :] = input_mask
    generated_mask = target_mask_native.copy()
    generated_mask[0::2, :] = False

    low = zero_invalid(low, low_mask)
    target_raw = zero_invalid(target_raw, target_mask_native)
    prediction_raw = zero_invalid(prediction_raw, target_mask_native)
    range_values = [target_raw[0], low[0], prediction_raw[0]]
    intensity_values = [target_raw[1], low[1], prediction_raw[1]]
    range_masks = [target_mask_native, low_mask, target_mask_native]
    intensity_masks = range_masks

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_three_panel(
        range_values,
        range_masks,
        ["GT range (32-line)", "Low range (16-line on 32-grid)", "TULIP Pred range"],
        output=output_dir / "range_gt_low_pred.png",
        title="Waymo pure GT / Low / Pred range",
        vmax=args.range_vmax,
        cmap_name="turbo",
    )
    save_three_panel(
        intensity_values,
        intensity_masks,
        ["GT intensity", "Low intensity (16-line on 32-grid)", "TULIP Pred intensity"],
        output=output_dir / "intensity_gt_low_pred_linear.png",
        title="Waymo pure GT / Low / Pred intensity (linear display)",
        vmax=args.intensity_vmax,
        cmap_name="viridis",
        transform=lambda value: np.clip(value, 0.0, None),
    )
    save_three_panel(
        intensity_values,
        intensity_masks,
        ["GT intensity", "Low intensity (16-line on 32-grid)", "TULIP Pred intensity"],
        output=output_dir / "intensity_gt_low_pred_log1p.png",
        title="Waymo pure GT / Low / Pred intensity (log1p display)",
        vmax=float(np.log1p(args.intensity_log_vmax)),
        cmap_name="magma",
        transform=lambda value: np.log1p(np.clip(value, 0.0, None)),
    )

    range_error = np.abs(prediction_raw[0] - target_raw[0])
    intensity_error = np.abs(prediction_raw[1] - target_raw[1])
    np.savez_compressed(
        output_dir / "comparison_raw.npz",
        gt_range=target_raw[0],
        low_range=low[0],
        pred_range=prediction_raw[0],
        gt_intensity=target_raw[1],
        low_intensity=low[1],
        pred_intensity=prediction_raw[1],
        gt_mask=target_mask_native,
        low_mask=low_mask,
        pred_mask=target_mask_native,
        generated_mask=generated_mask,
        range_abs_error_generated=range_error * generated_mask,
        intensity_abs_error_generated=intensity_error * generated_mask,
    )
    metadata = {
        "checkpoint": str(args.checkpoint),
        "dataset_role": args.dataset_role,
        "sample_index": args.sample_index,
        "segment_id": sample["segment_id"],
        "frame_index": sample["frame_index"],
        "range_scale_m": args.range_scale_m,
        "intensity_transform": args.intensity_transform,
        "input_shape_native": [16, 2650, 2],
        "target_shape_native": [32, 2650, 2],
        "low_shape_32_grid": [32, 2650, 2],
        "low_rows_on_32_grid": list(range(0, 32, 2)),
        "generated_rows": list(range(1, 32, 2)),
        "comparison_scope": "pure GT / Low / TULIP Pred; no oracle range",
        "png_is_visualization_only": True,
        "display_only": {
            "range_vmax": args.range_vmax,
            "intensity_vmax": args.intensity_vmax,
            "intensity_log_vmax": args.intensity_log_vmax,
        },
        "stats": {
            "gt_range": finite_stats(target_raw[0], target_mask_native),
            "low_range": finite_stats(low[0], low_mask),
            "pred_range": finite_stats(prediction_raw[0], target_mask_native),
            "gt_intensity": finite_stats(target_raw[1], target_mask_native),
            "low_intensity": finite_stats(low[1], low_mask),
            "pred_intensity": finite_stats(prediction_raw[1], target_mask_native),
            "generated_range_mae": float(range_error[generated_mask].mean()) if generated_mask.any() else None,
            "generated_intensity_mae": float(intensity_error[generated_mask].mean()) if generated_mask.any() else None,
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "range_image": str(output_dir / "range_gt_low_pred.png"),
        "intensity_linear_image": str(output_dir / "intensity_gt_low_pred_linear.png"),
        "intensity_log1p_image": str(output_dir / "intensity_gt_low_pred_log1p.png"),
        "raw_npz": str(output_dir / "comparison_raw.npz"),
        "generated_range_mae": metadata["stats"]["generated_range_mae"],
        "generated_intensity_mae": metadata["stats"]["generated_intensity_mae"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
