#!/usr/bin/env python3
"""Evaluate Waymo 16-to-32 checkpoints under one controlled contract."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tulip_adapter.waymo_16_to_32_dataset import Waymo16To32Dataset
from tulip_adapter.waymo_16_to_32_tulip import build_waymo_16_to_32_model


SCRIPT_VERSION = "1.0.0"
NATIVE_WIDTH = 2650
PADDED_WIDTH = 2688
INTENSITY_BINS = (
    ("0_0.5", 0.0, 0.5),
    ("0.5_1", 0.5, 1.0),
    ("1_2", 1.0, 2.0),
    ("2_5", 2.0, 5.0),
    ("5_20", 5.0, 20.0),
    ("20_100", 20.0, 100.0),
    ("100_1000", 100.0, 1000.0),
    ("1000_inf", 1000.0, float("inf")),
)
THRESHOLDS = (20.0, 100.0, 1000.0)
METHODS = ("tulip", "nearest", "linear")


def parse_args() -> argparse.Namespace:
    """Parse controlled evaluation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--tulip-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--dataset-role", choices=("train", "validation", "test"), required=True
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--range-scale-m", type=float)
    parser.add_argument("--intensity-transform", choices=("raw", "log1p"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load one checkpoint on CPU without changing its contents."""
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict) or "model" not in value:
        raise ValueError("checkpoint must be a dictionary containing model")
    return value


def resolve_checkpoint_contract(
    checkpoint: dict[str, Any],
    *,
    range_scale_override: float | None,
    intensity_transform_override: str | None,
) -> tuple[float, str]:
    """Resolve preprocessing from checkpoint metadata and reject mismatches.

    Input shape: checkpoint dictionary and optional scalar/string overrides.
    Output shape: ``(range_scale_m, intensity_transform)``.
    """
    saved_args = checkpoint.get("args", {})
    contract = checkpoint.get("training_contract", {})
    if not isinstance(saved_args, dict):
        saved_args = vars(saved_args) if hasattr(saved_args, "__dict__") else {}
    if not isinstance(contract, dict):
        contract = {}
    saved_scale = contract.get("range_scale_m", saved_args.get("range_scale_m"))
    saved_transform = contract.get(
        "intensity_transform", saved_args.get("intensity_transform")
    )
    if saved_scale is None and range_scale_override is None:
        raise ValueError("checkpoint does not record range_scale_m; pass it explicitly")
    if saved_transform is None and intensity_transform_override is None:
        raise ValueError(
            "checkpoint does not record intensity_transform; pass it explicitly"
        )
    scale = float(saved_scale if range_scale_override is None else range_scale_override)
    transform = str(
        saved_transform
        if intensity_transform_override is None
        else intensity_transform_override
    )
    if scale <= 0:
        raise ValueError("range_scale_m must be positive")
    if transform not in ("raw", "log1p"):
        raise ValueError(f"unsupported intensity_transform: {transform}")
    if saved_scale is not None and range_scale_override is not None:
        if not np.isclose(float(saved_scale), scale, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"range scale mismatch: checkpoint={saved_scale}, override={scale}"
            )
    if saved_transform is not None and intensity_transform_override is not None:
        if str(saved_transform) != transform:
            raise ValueError(
                "intensity transform mismatch: "
                f"checkpoint={saved_transform}, override={transform}"
            )
    return scale, transform


def generated_only_mask(mask: torch.Tensor) -> torch.Tensor:
    """Retain valid generated odd rows only.

    Input shape: bool-compatible ``[B,32,W]``.
    Output shape: bool ``[B,32,W]`` with even rows false.
    """
    if mask.ndim != 3 or mask.shape[1] != 32:
        raise ValueError(f"mask must be [B,32,W], got {tuple(mask.shape)}")
    result = mask.to(dtype=torch.bool).clone()
    result[:, 0::2] = False
    return result


def restore_raw_domain(
    tensor: torch.Tensor,
    *,
    range_scale_m: float,
    intensity_transform: str,
) -> torch.Tensor:
    """Restore model-domain range/intensity without silently clipping output.

    Input shape: ``[B,2,H,W]``.
    Output shape: ``[B,2,H,W]`` in metres and raw intensity.
    """
    if tensor.ndim != 4 or tensor.shape[1] != 2:
        raise ValueError("tensor must be [B,2,H,W]")
    if range_scale_m <= 0:
        raise ValueError("range_scale_m must be positive")
    result = tensor.clone()
    result[:, 0] *= range_scale_m
    if intensity_transform == "log1p":
        result[:, 1] = torch.expm1(result[:, 1])
    elif intensity_transform != "raw":
        raise ValueError(f"unsupported intensity_transform: {intensity_transform}")
    return result


def binary_counts(
    target: torch.Tensor,
    prediction: torch.Tensor,
    mask: torch.Tensor,
    *,
    threshold: float,
) -> dict[str, int]:
    """Count threshold-classification outcomes on one mask.

    Input shapes: target/prediction/mask ``[B,H,W]``.
    Output shape: dictionary of scalar integer counts.
    """
    if target.shape != prediction.shape or target.shape != mask.shape:
        raise ValueError("target, prediction, and mask shapes differ")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    selected = mask.to(dtype=torch.bool)
    truth = target >= threshold
    detected = prediction >= threshold
    return {
        "target_positive": int((selected & truth).sum()),
        "predicted_positive": int((selected & detected).sum()),
        "true_positive": int((selected & truth & detected).sum()),
        "false_positive": int((selected & ~truth & detected).sum()),
        "false_negative": int((selected & truth & ~detected).sum()),
    }


def _empty_metric_accumulator() -> dict[str, float | int]:
    """Create one pixel-weighted metric accumulator."""
    return {
        "count": 0,
        "range_abs_sum": 0.0,
        "intensity_raw_abs_sum": 0.0,
        "intensity_log1p_abs_sum": 0.0,
    }


def _empty_bin_accumulator() -> dict[str, dict[str, float | int]]:
    """Create raw-intensity band accumulators."""
    return {
        label: {
            "count": 0,
            "target_sum": 0.0,
            **{f"{method}_abs_sum": 0.0 for method in METHODS},
        }
        for label, _, _ in INTENSITY_BINS
    }


def _empty_threshold_accumulator() -> dict[str, dict[str, dict[str, int]]]:
    """Create high-intensity detection accumulators."""
    return {
        f"{threshold:g}": {
            method: {
                "target_positive": 0,
                "predicted_positive": 0,
                "true_positive": 0,
                "false_positive": 0,
                "false_negative": 0,
            }
            for method in METHODS
        }
        for threshold in THRESHOLDS
    }


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    """Return a finite ratio or null when the denominator is zero."""
    return float(numerator / denominator) if denominator else None


def _atomic_text(path: Path, text: str) -> None:
    """Write UTF-8 text through an atomic rename."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write per-frame rows through an atomic rename."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    """Evaluate one checkpoint on generated odd rows in raw and log1p domains."""
    args = parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("max_frames must be positive")
    if args.output_root.exists() and not args.overwrite:
        existing = [args.output_root / "summary.json", args.output_root / "per_frame.csv"]
        if any(path.exists() for path in existing):
            raise FileExistsError(
                f"evaluation output exists in {args.output_root}; pass --overwrite"
            )
    checkpoint = load_checkpoint(args.checkpoint)
    range_scale_m, intensity_transform = resolve_checkpoint_contract(
        checkpoint,
        range_scale_override=args.range_scale_m,
        intensity_transform_override=args.intensity_transform,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    dataset = Waymo16To32Dataset(
        args.index,
        dataset_role=args.dataset_role,
        intensity_transform=intensity_transform,
        range_scale_m=range_scale_m,
    )
    count = len(dataset) if args.max_frames is None else min(args.max_frames, len(dataset))
    if count < 1:
        raise ValueError("no evaluation samples")
    model = build_waymo_16_to_32_model(args.tulip_root)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()
    args.output_root.mkdir(parents=True, exist_ok=True)

    metric_accum = {method: _empty_metric_accumulator() for method in METHODS}
    bin_accum = _empty_bin_accumulator()
    threshold_accum = _empty_threshold_accumulator()
    per_frame: list[dict[str, Any]] = []
    tulip_negative_range = 0
    tulip_negative_intensity = 0
    total_valid = 0

    with torch.inference_mode():
        for sample_index in range(count):
            sample = dataset[sample_index]
            input_signal = sample["input"].unsqueeze(0).to(device)
            target_signal = sample["target"].unsqueeze(0).to(device)
            target_mask = sample["target_mask"].unsqueeze(0).to(device)
            generated_mask = generated_only_mask(
                sample["generated_mask"].unsqueeze(0).to(device)
            ) & target_mask.to(dtype=torch.bool)
            prediction, _ = model(input_signal, target_signal, target_mask)
            if not torch.isfinite(prediction).all():
                raise RuntimeError(f"non-finite prediction at sample {sample_index}")

            input_raw = restore_raw_domain(
                input_signal,
                range_scale_m=range_scale_m,
                intensity_transform=intensity_transform,
            )
            target_raw = restore_raw_domain(
                target_signal,
                range_scale_m=range_scale_m,
                intensity_transform=intensity_transform,
            )
            tulip_unclamped = restore_raw_domain(
                prediction,
                range_scale_m=range_scale_m,
                intensity_transform=intensity_transform,
            )
            nearest_raw = F.interpolate(input_raw, size=(32, PADDED_WIDTH), mode="nearest")
            linear_raw = F.interpolate(
                input_raw,
                size=(32, PADDED_WIDTH),
                mode="bilinear",
                align_corners=False,
            )

            target_raw = target_raw[..., :NATIVE_WIDTH]
            target_log = target_signal[:, 1, :, :NATIVE_WIDTH]
            mask = generated_mask[..., :NATIVE_WIDTH]
            tulip_unclamped = tulip_unclamped[..., :NATIVE_WIDTH]
            tulip_raw = tulip_unclamped.clone()
            tulip_raw[:, 1].clamp_min_(0.0)
            nearest_raw = nearest_raw[..., :NATIVE_WIDTH]
            linear_raw = linear_raw[..., :NATIVE_WIDTH]
            methods_raw = {
                "tulip": tulip_raw,
                "nearest": nearest_raw,
                "linear": linear_raw,
            }
            methods_log = {
                "tulip": prediction[:, 1, :, :NATIVE_WIDTH],
                "nearest": torch.log1p(nearest_raw[:, 1].clamp_min(0.0)),
                "linear": torch.log1p(linear_raw[:, 1].clamp_min(0.0)),
            }
            valid_count = int(mask.sum())
            if valid_count == 0:
                raise ValueError(f"sample {sample_index} has no generated valid pixels")
            total_valid += valid_count
            tulip_negative_range += int((mask & (tulip_unclamped[:, 0] < 0)).sum())
            tulip_negative_intensity += int((mask & (tulip_unclamped[:, 1] < 0)).sum())

            frame_row: dict[str, Any] = {
                "segment_id": sample["segment_id"],
                "frame_index": sample["frame_index"],
                "timestamp_micros": sample["timestamp_micros"],
                "generated_valid_pixels": valid_count,
            }
            for method in METHODS:
                raw_value = methods_raw[method]
                log_value = methods_log[method]
                if not torch.isfinite(raw_value).all() or not torch.isfinite(log_value).all():
                    raise RuntimeError(
                        f"non-finite {method} metric value at sample {sample_index}"
                    )
                range_abs = (raw_value[:, 0] - target_raw[:, 0]).abs()[mask]
                intensity_raw_abs = (raw_value[:, 1] - target_raw[:, 1]).abs()[mask]
                intensity_log_abs = (log_value - target_log).abs()[mask]
                accumulator = metric_accum[method]
                accumulator["count"] += valid_count
                accumulator["range_abs_sum"] += float(range_abs.sum())
                accumulator["intensity_raw_abs_sum"] += float(intensity_raw_abs.sum())
                accumulator["intensity_log1p_abs_sum"] += float(intensity_log_abs.sum())
                frame_row[f"{method}_range_mae_m"] = float(range_abs.mean())
                frame_row[f"{method}_intensity_raw_mae"] = float(intensity_raw_abs.mean())
                frame_row[f"{method}_intensity_log1p_mae"] = float(intensity_log_abs.mean())

            target_values = target_raw[:, 1][mask]
            for label, lower, upper in INTENSITY_BINS:
                selected = target_values >= lower
                if np.isfinite(upper):
                    selected &= target_values < upper
                selected_count = int(selected.sum())
                if selected_count == 0:
                    continue
                item = bin_accum[label]
                item["count"] += selected_count
                item["target_sum"] += float(target_values[selected].sum())
                for method in METHODS:
                    predicted_values = methods_raw[method][:, 1][mask]
                    item[f"{method}_abs_sum"] += float(
                        (predicted_values[selected] - target_values[selected]).abs().sum()
                    )

            for threshold in THRESHOLDS:
                threshold_key = f"{threshold:g}"
                for method in METHODS:
                    counts = binary_counts(
                        target_raw[:, 1],
                        methods_raw[method][:, 1],
                        mask,
                        threshold=threshold,
                    )
                    for name, value in counts.items():
                        threshold_accum[threshold_key][method][name] += value
            per_frame.append(frame_row)
            print(json.dumps({"evaluated": sample_index + 1, "frames": count}), flush=True)

    aggregate: dict[str, dict[str, float | int]] = {}
    frame_mean: dict[str, dict[str, float]] = {}
    for method in METHODS:
        item = metric_accum[method]
        method_count = int(item["count"])
        aggregate[method] = {
            "count": method_count,
            "range_mae_m": float(item["range_abs_sum"]) / method_count,
            "intensity_raw_mae": float(item["intensity_raw_abs_sum"]) / method_count,
            "intensity_log1p_mae": float(item["intensity_log1p_abs_sum"]) / method_count,
        }
        frame_mean[method] = {
            "range_mae_m": sum(float(row[f"{method}_range_mae_m"]) for row in per_frame) / count,
            "intensity_raw_mae": sum(
                float(row[f"{method}_intensity_raw_mae"]) for row in per_frame
            ) / count,
            "intensity_log1p_mae": sum(
                float(row[f"{method}_intensity_log1p_mae"]) for row in per_frame
            ) / count,
        }

    intensity_bins: dict[str, dict[str, float | int]] = {}
    for label, _, _ in INTENSITY_BINS:
        item = bin_accum[label]
        bin_count = int(item["count"])
        result: dict[str, float | int] = {"count": bin_count}
        if bin_count:
            result["fraction"] = bin_count / total_valid
            result["target_mean"] = float(item["target_sum"]) / bin_count
            for method in METHODS:
                result[f"{method}_raw_mae"] = (
                    float(item[f"{method}_abs_sum"]) / bin_count
                )
        else:
            result["fraction"] = 0.0
        intensity_bins[label] = result

    threshold_metrics: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for threshold_key, methods in threshold_accum.items():
        threshold_metrics[threshold_key] = {}
        for method, values in methods.items():
            tp = values["true_positive"]
            fp = values["false_positive"]
            fn = values["false_negative"]
            precision = _safe_ratio(tp, tp + fp)
            recall = _safe_ratio(tp, tp + fn)
            f1 = None
            if precision is not None and recall is not None and precision + recall > 0:
                f1 = 2.0 * precision * recall / (precision + recall)
            threshold_metrics[threshold_key][method] = {
                **values,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

    saved_args = checkpoint.get("args", {})
    checkpoint_profile = checkpoint.get("training_contract", {}).get(
        "profile", saved_args.get("profile", "historical") if isinstance(saved_args, dict) else "historical"
    )
    summary = {
        "script_version": SCRIPT_VERSION,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_task": checkpoint.get("task"),
        "checkpoint_profile": checkpoint_profile,
        "dataset_role": args.dataset_role,
        "evaluated_frames": count,
        "mask_scope": "valid_generated_odd_rows_only",
        "generated_rows": list(range(1, 32, 2)),
        "range_scale_m": range_scale_m,
        "intensity_transform": intensity_transform,
        "raw_intensity_prediction_policy": "clamp_negative_to_zero_for_metrics",
        "interpolation_baseline_domain": "raw_range_and_raw_intensity",
        "aggregate_pixel_weighted": aggregate,
        "aggregate_frame_mean": frame_mean,
        "tulip_negative_prediction": {
            "range_count": tulip_negative_range,
            "range_fraction": tulip_negative_range / total_valid,
            "intensity_count_before_metric_clamp": tulip_negative_intensity,
            "intensity_fraction_before_metric_clamp": tulip_negative_intensity / total_valid,
        },
        "intensity_bins_raw_target": intensity_bins,
        "high_intensity_detection": threshold_metrics,
    }
    _write_csv(args.output_root / "per_frame.csv", per_frame)
    _atomic_text(
        args.output_root / "summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
