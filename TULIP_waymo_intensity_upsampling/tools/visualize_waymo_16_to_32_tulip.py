#!/usr/bin/env python3
"""Create Waymo 16-to-32 range/intensity pseudocolor comparisons."""
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tulip_adapter.waymo_16_to_32_dataset import Waymo16To32Dataset
from tulip_adapter.waymo_16_to_32_tulip import build_waymo_16_to_32_model


def parse_args() -> argparse.Namespace:
    """Parse visualization arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--tulip-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-role", default="validation")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--intensity-transform", choices=("raw", "log1p"), default="raw")
    parser.add_argument("--range-vmax", type=float, default=75.0)
    parser.add_argument("--intensity-vmax", type=float, default=20.0)
    parser.add_argument("--intensity-log-vmax", type=float, default=20000.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def to_raw(
    input_signal: torch.Tensor,
    target_signal: torch.Tensor,
    prediction: torch.Tensor,
    intensity_transform: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert tensors to raw range/intensity values.

    Input shapes: input [1,2,16,2688], target/prediction [1,2,32,2688].
    Output shapes: same as inputs in raw value units.
    """
    x = input_signal.clone()
    y = target_signal.clone()
    pred = prediction.clone()
    if intensity_transform == "log1p":
        x[:, 1] = torch.expm1(x[:, 1])
        y[:, 1] = torch.expm1(y[:, 1])
        pred[:, 1] = torch.expm1(pred[:, 1]).clamp_min(0.0)
    return x, y, pred


def masked_array(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return float values with invalid pixels as NaN."""
    result = np.asarray(values, dtype=np.float32).copy()
    result[~np.asarray(mask, dtype=bool)] = np.nan
    return result


def make_cmap(name: str):
    """Create a colormap with invalid values rendered black."""
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad("black")
    cmap.set_under("black")
    return cmap


def save_panels(
    panels: list[tuple[np.ndarray, np.ndarray, str]],
    *,
    output: Path,
    title: str,
    vmax: float,
    cmap_name: str,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> None:
    """Save a six-panel pseudocolor comparison.

    Input panels contain value/mask/title with 2D arrays of [H,W].
    Output is a PNG comparison figure.
    """
    figure, axes = plt.subplots(2, 3, figsize=(19, 8), constrained_layout=True)
    cmap = make_cmap(cmap_name)
    for axis, (values, mask, panel_title) in zip(axes.flat, panels):
        display = np.asarray(values, dtype=np.float32)
        if transform is not None:
            display = transform(display)
        image = axis.imshow(
            masked_array(display, mask),
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
        )
        axis.set_title(panel_title)
        axis.set_xlabel("Azimuth column")
        axis.set_ylabel("LiDAR row")
        figure.colorbar(image, ax=axis, pad=0.01)
    figure.suptitle(title, fontsize=15)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def finite_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    """Calculate finite statistics over a boolean mask."""
    selected = np.asarray(values)[np.asarray(mask, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return {"count": 0}
    return {
        "count": int(selected.size),
        "min": float(selected.min()),
        "max": float(selected.max()),
        "mean": float(selected.mean()),
        "p50": float(np.percentile(selected, 50.0)),
        "p99": float(np.percentile(selected, 99.0)),
    }


def main() -> None:
    """Generate range and intensity pseudocolor comparisons for one sample."""
    args = parse_args()
    if args.sample_index < 0:
        raise ValueError("sample-index must be non-negative")
    if args.range_vmax <= 0 or args.intensity_vmax <= 0 or args.intensity_log_vmax <= 0:
        raise ValueError("display maxima must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    dataset = Waymo16To32Dataset(
        args.index,
        dataset_role=args.dataset_role,
        intensity_transform=args.intensity_transform,
    )
    if args.sample_index >= len(dataset):
        raise IndexError(f"sample-index {args.sample_index} >= dataset length {len(dataset)}")

    sample = dataset[args.sample_index]
    model = build_waymo_16_to_32_model(args.tulip_root).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with torch.inference_mode():
        x = sample["input"].unsqueeze(0).to(device)
        y = sample["target"].unsqueeze(0).to(device)
        target_mask = sample["target_mask"].unsqueeze(0).to(device)
        generated_mask = sample["generated_mask"].unsqueeze(0).to(device)
        prediction, _ = model(x, y, target_mask)
        x_raw, y_raw, pred_raw = to_raw(x, y, prediction, args.intensity_transform)
        nearest = F.interpolate(x_raw, size=(32, 2688), mode="nearest")
        linear = F.interpolate(
            x_raw, size=(32, 2688), mode="bilinear", align_corners=False
        )

    x_native = x_raw[..., :2650][0].detach().cpu().numpy()
    y_native = y_raw[..., :2650][0].detach().cpu().numpy()
    pred_native = pred_raw[..., :2650][0].detach().cpu().numpy()
    nearest_native = nearest[..., :2650][0].detach().cpu().numpy()
    linear_native = linear[..., :2650][0].detach().cpu().numpy()
    target_mask_native = target_mask[..., :2650][0].detach().cpu().numpy().astype(bool)
    input_mask_native = sample["input_mask"].numpy()[..., :2650].astype(bool)
    generated_mask_native = generated_mask[..., :2650][0].detach().cpu().numpy().astype(bool)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    range_panels = [
        (x_native[0], input_mask_native, "16-line input range"),
        (y_native[0], target_mask_native, "32-line GT range"),
        (pred_native[0], target_mask_native, "TULIP range"),
        (nearest_native[0], target_mask_native, "Nearest range"),
        (linear_native[0], target_mask_native, "Linear range"),
        (np.abs(pred_native[0] - y_native[0]), generated_mask_native,
         "TULIP abs error (generated rows)"),
    ]
    save_panels(
        range_panels,
        output=args.output_dir / "range_comparison.png",
        title="Waymo 16-to-32 range comparison",
        vmax=args.range_vmax,
        cmap_name="turbo",
    )

    intensity_panels = [
        (x_native[1], input_mask_native, "16-line input intensity"),
        (y_native[1], target_mask_native, "32-line GT intensity"),
        (pred_native[1], target_mask_native, "TULIP intensity"),
        (nearest_native[1], target_mask_native, "Nearest intensity"),
        (linear_native[1], target_mask_native, "Linear intensity"),
        (np.abs(pred_native[1] - y_native[1]), generated_mask_native,
         "TULIP abs error (generated rows)"),
    ]
    save_panels(
        intensity_panels,
        output=args.output_dir / "intensity_comparison_linear.png",
        title="Waymo 16-to-32 raw intensity comparison (display clipped)",
        vmax=args.intensity_vmax,
        cmap_name="viridis",
        transform=lambda values: np.clip(values, 0.0, None),
    )
    log_vmax = float(np.log1p(args.intensity_log_vmax))
    save_panels(
        intensity_panels,
        output=args.output_dir / "intensity_comparison_log1p.png",
        title="Waymo 16-to-32 intensity comparison (log1p display)",
        vmax=log_vmax,
        cmap_name="magma",
        transform=lambda values: np.log1p(np.clip(values, 0.0, None)),
    )

    generated_mask_t = generated_mask_native
    metadata = {
        "checkpoint": str(args.checkpoint),
        "dataset_role": args.dataset_role,
        "sample_index": args.sample_index,
        "segment_id": sample["segment_id"],
        "frame_index": sample["frame_index"],
        "intensity_transform": args.intensity_transform,
        "display_only": {
            "range_vmax": args.range_vmax,
            "intensity_vmax": args.intensity_vmax,
            "intensity_log_vmax": args.intensity_log_vmax,
        },
        "mask_scope_for_error": "generated_odd_rows_only",
        "generated_rows": list(range(1, 32, 2)),
        "generated_valid_pixels": int(generated_mask_t.sum()),
        "range_gt_stats_generated": finite_stats(y_native[0], generated_mask_t),
        "intensity_gt_stats_generated": finite_stats(y_native[1], generated_mask_t),
        "range_tulip_mae_generated": float(
            np.abs(pred_native[0] - y_native[0])[generated_mask_t].mean()
        ),
        "intensity_tulip_mae_generated": float(
            np.abs(pred_native[1] - y_native[1])[generated_mask_t].mean()
        ),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + chr(10),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
