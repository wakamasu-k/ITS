#!/usr/bin/env python3
"""Evaluate Waymo 16-to-32 performance on generated rows only."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tulip_adapter.waymo_16_to_32_dataset import Waymo16To32Dataset
from tulip_adapter.waymo_16_to_32_tulip import build_waymo_16_to_32_model


BINS = (
    (0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0),
    (5.0, 20.0), (20.0, 100.0), (100.0, 1000.0),
    (1000.0, float("inf")),
)


def parse_args() -> argparse.Namespace:
    """Parse generated-row evaluation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--tulip-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--intensity-transform", choices=("raw", "log1p"), default="raw")
    parser.add_argument("--dataset-role", default="validation")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--range-scale-m",
        type=float,
        help=(
            "Range normalization used during training. By default, read it "
            "from checkpoint['args']['range_scale_m']."
        ),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resolve_range_scale_m(checkpoint: dict, requested: float | None) -> float:
    """Resolve range normalization and reject checkpoint/evaluation mismatch."""
    checkpoint_args = checkpoint.get("args")
    if isinstance(checkpoint_args, dict):
        stored = checkpoint_args.get("range_scale_m")
    else:
        stored = getattr(checkpoint_args, "range_scale_m", None)

    if requested is None:
        if stored is None:
            raise ValueError(
                "checkpoint does not record range_scale_m; pass --range-scale-m"
            )
        resolved = float(stored)
    else:
        resolved = float(requested)
        if stored is not None and not math.isclose(
            resolved, float(stored), rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                f"range-scale-m {resolved} does not match checkpoint value {stored}"
            )
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError("range-scale-m must be finite and positive")
    return resolved


def generated_only_mask(mask: torch.Tensor) -> torch.Tensor:
    """Keep valid pixels on generated odd target rows only.

    Input shape: [B,32,W].
    Output shape: bool [B,32,W], with rows 1,3,...,31 retained.
    """
    if mask.ndim != 3 or mask.shape[1] % 2 != 0:
        raise ValueError(f"mask must have an even height, got {tuple(mask.shape)}")
    result = mask.to(dtype=torch.bool).clone()
    result[:, 0::2] = False
    return result


def channel_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    channel: int,
) -> float:
    """Calculate one-channel MAE on the supplied mask.

    Input shapes: prediction/target [B,2,32,W], mask [B,32,W].
    Output shape: Python float.
    """
    mask_f = mask.to(dtype=prediction.dtype)
    count = mask_f.sum()
    if count.item() == 0:
        raise ValueError("evaluation mask contains no valid pixels")
    return float(((prediction[:, channel] - target[:, channel]).abs() * mask_f).sum() / count)


def raw_signals(
    input_signal: torch.Tensor,
    target_signal: torch.Tensor,
    prediction: torch.Tensor,
    intensity_transform: str,
    range_scale_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert signals to raw range/intensity metric values.

    Input shapes: input [B,2,16,2688], target/prediction [B,2,32,2688].
    Output shapes: same shapes in raw range and raw intensity units.
    """
    x_raw = input_signal.clone()
    y_raw = target_signal.clone()
    pred_raw = prediction.clone()
    if not math.isfinite(range_scale_m) or range_scale_m <= 0:
        raise ValueError("range_scale_m must be finite and positive")
    x_raw[:, 0] *= range_scale_m
    y_raw[:, 0] *= range_scale_m
    pred_raw[:, 0] *= range_scale_m
    if intensity_transform == "log1p":
        x_raw[:, 1] = torch.expm1(x_raw[:, 1])
        y_raw[:, 1] = torch.expm1(y_raw[:, 1])
        pred_raw[:, 1] = torch.expm1(pred_raw[:, 1]).clamp_min(0.0)
    return x_raw, y_raw, pred_raw


def main() -> None:
    """Evaluate TULIP and interpolation baselines on generated rows."""
    args = parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("max_frames must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    range_scale_m = resolve_range_scale_m(checkpoint, args.range_scale_m)

    dataset = Waymo16To32Dataset(
        args.index,
        dataset_role=args.dataset_role,
        intensity_transform=args.intensity_transform,
        range_scale_m=range_scale_m,
    )
    count = len(dataset) if args.max_frames is None else min(args.max_frames, len(dataset))
    if count < 1:
        raise ValueError("no evaluation samples")

    model = build_waymo_16_to_32_model(args.tulip_root).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    args.output_root.mkdir(parents=True, exist_ok=True)

    per_frame: list[dict[str, object]] = []
    accum = {
        f"{low:g}_{high:g}": {
            "count": 0,
            "target_sum": 0.0,
            "tulip_abs_sum": 0.0,
            "nearest_abs_sum": 0.0,
            "linear_abs_sum": 0.0,
        }
        for low, high in BINS
    }

    with torch.inference_mode():
        for index in range(count):
            sample = dataset[index]
            input_signal = sample["input"].unsqueeze(0).to(device)
            target_signal = sample["target"].unsqueeze(0).to(device)
            target_mask = sample["target_mask"].unsqueeze(0).to(device)
            generated_mask = generated_only_mask(
                sample["generated_mask"].unsqueeze(0).to(device)
            )
            prediction, _ = model(input_signal, target_signal, target_mask)
            x_raw, y_raw, pred_raw = raw_signals(
                input_signal,
                target_signal,
                prediction,
                args.intensity_transform,
                range_scale_m,
            )
            nearest = F.interpolate(x_raw, size=(32, 2688), mode="nearest")
            linear = F.interpolate(
                x_raw, size=(32, 2688), mode="bilinear", align_corners=False
            )
            prediction_native = pred_raw[..., :2650]
            nearest_native = nearest[..., :2650]
            linear_native = linear[..., :2650]
            target_native = y_raw[..., :2650]
            mask_native = generated_mask[..., :2650]
            if not torch.isfinite(prediction_native).all():
                raise RuntimeError(f"non-finite prediction at frame {index}")

            row = {
                "segment_id": sample["segment_id"],
                "frame_index": sample["frame_index"],
                "tulip_range_mae": channel_mae(
                    prediction_native, target_native, mask_native, 0
                ),
                "tulip_intensity_mae": channel_mae(
                    prediction_native, target_native, mask_native, 1
                ),
                "nearest_range_mae": channel_mae(
                    nearest_native, target_native, mask_native, 0
                ),
                "nearest_intensity_mae": channel_mae(
                    nearest_native, target_native, mask_native, 1
                ),
                "linear_range_mae": channel_mae(
                    linear_native, target_native, mask_native, 0
                ),
                "linear_intensity_mae": channel_mae(
                    linear_native, target_native, mask_native, 1
                ),
                "generated_valid_pixels": int(mask_native.sum()),
            }
            per_frame.append(row)

            target_values = target_native[0, 1][mask_native[0]].detach().cpu().numpy()
            tulip_values = prediction_native[0, 1][mask_native[0]].detach().cpu().numpy()
            nearest_values = nearest_native[0, 1][mask_native[0]].detach().cpu().numpy()
            linear_values = linear_native[0, 1][mask_native[0]].detach().cpu().numpy()
            for low, high in BINS:
                key = f"{low:g}_{high:g}"
                selected = (target_values >= low) & (target_values < high)
                if not np.any(selected):
                    continue
                item = accum[key]
                item["count"] += int(selected.sum())
                item["target_sum"] += float(target_values[selected].sum())
                item["tulip_abs_sum"] += float(
                    np.abs(tulip_values[selected] - target_values[selected]).sum()
                )
                item["nearest_abs_sum"] += float(
                    np.abs(nearest_values[selected] - target_values[selected]).sum()
                )
                item["linear_abs_sum"] += float(
                    np.abs(linear_values[selected] - target_values[selected]).sum()
                )
            print(json.dumps({"evaluated": index + 1, "frames": count}), flush=True)

    metric_names = (
        "tulip_range_mae", "tulip_intensity_mae",
        "nearest_range_mae", "nearest_intensity_mae",
        "linear_range_mae", "linear_intensity_mae",
    )
    bins = {}
    total_bin_count = sum(item["count"] for item in accum.values())
    for key, item in accum.items():
        item_count = item["count"]
        if item_count:
            bins[key] = {
                "count": item_count,
                "fraction": item_count / total_bin_count,
                "target_mean": item["target_sum"] / item_count,
                "tulip_mae": item["tulip_abs_sum"] / item_count,
                "nearest_mae": item["nearest_abs_sum"] / item_count,
                "linear_mae": item["linear_abs_sum"] / item_count,
            }
        else:
            bins[key] = {"count": 0, "fraction": 0.0}

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset_role": args.dataset_role,
        "evaluated_frames": count,
        "mask_scope": "generated_odd_rows_only",
        "generated_rows": list(range(1, 32, 2)),
        "range_scale_m": range_scale_m,
        "intensity_transform": args.intensity_transform,
        "metric_value_domain": "raw_range_and_raw_intensity",
        "generated_valid_pixel_count": total_bin_count,
        **{
            name: sum(float(row[name]) for row in per_frame) / count
            for name in metric_names
        },
        "intensity_bins": bins,
    }
    with (args.output_root / "per_frame.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = list(per_frame[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_frame)
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + chr(10),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
