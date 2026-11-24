#!/usr/bin/env python3
"""Calibrate a raw-intensity decision threshold on one validation split.

This tool does not retrain or alter a checkpoint.  The target is declared
high at one fixed raw threshold (normally 1000), while only the prediction
threshold is swept.  It uses the same generated-odd-row mask and raw-domain
restoration contract as the controlled 16-to-32 evaluator.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.evaluate_waymo_16_to_32_controlled import (
    generated_only_mask,
    load_checkpoint,
    resolve_checkpoint_contract,
    restore_raw_domain,
)
from tulip_adapter.waymo_16_to_32_dataset import Waymo16To32Dataset
from tulip_adapter.waymo_16_to_32_tulip import build_waymo_16_to_32_model


NATIVE_WIDTH = 2650
DEFAULT_PREDICTION_THRESHOLDS = (100.0, 200.0, 300.0, 500.0, 750.0, 1000.0,
                                 1500.0, 2000.0, 5000.0)


def parse_thresholds(value: str) -> Tuple[float, ...]:
    """Parse sorted, unique, positive comma-separated raw thresholds."""
    values: List[float] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        threshold = float(token)
        if threshold < 0:
            raise ValueError("prediction thresholds must be non-negative")
        values.append(threshold)
    if not values:
        raise ValueError("at least one prediction threshold is required")
    return tuple(sorted(set(values)))


def threshold_counts(
    target_raw: torch.Tensor,
    prediction_raw: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    target_threshold: float,
    prediction_threshold: float,
) -> Dict[str, int]:
    """Count a fixed target event against a candidate prediction cutoff."""
    if target_raw.shape != prediction_raw.shape or target_raw.shape != valid_mask.shape:
        raise ValueError("target, prediction, and valid mask must have matching shapes")
    if target_threshold < 0 or prediction_threshold < 0:
        raise ValueError("thresholds must be non-negative")
    selected = valid_mask.to(dtype=torch.bool)
    truth = target_raw >= target_threshold
    predicted = prediction_raw >= prediction_threshold
    return {
        "target_positive": int((selected & truth).sum()),
        "predicted_positive": int((selected & predicted).sum()),
        "true_positive": int((selected & truth & predicted).sum()),
        "false_positive": int((selected & ~truth & predicted).sum()),
        "false_negative": int((selected & truth & ~predicted).sum()),
    }


def metrics_from_counts(values: Dict[str, int]) -> Dict[str, Optional[float]]:
    """Compute precision, recall and F1 from accumulated binary counts."""
    tp = values["true_positive"]
    fp = values["false_positive"]
    fn = values["false_negative"]
    precision = float(tp / (tp + fp)) if tp + fp else None
    recall = float(tp / (tp + fn)) if tp + fn else None
    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = float(2.0 * precision * recall / (precision + recall))
    return {"precision": precision, "recall": recall, "f1": f1}


def best_f1_candidate(rows: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Select the deterministic maximum-F1 candidate for validation review."""
    candidates = [row for row in rows if row.get("f1") is not None]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            float(row["f1"]),
            float(row["precision"]),
            -float(row["prediction_threshold"]),
        ),
    )


def atomic_json(path: Path, value: Dict[str, Any]) -> None:
    """Write one JSON report atomically without leaving partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    """Parse calibration inputs; callers should use validation data only."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--tulip-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--dataset-role", choices=("train", "validation", "test"),
                        default="validation")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--target-threshold", type=float, default=1000.0)
    parser.add_argument(
        "--prediction-thresholds",
        default=",".join(str(value) for value in DEFAULT_PREDICTION_THRESHOLDS),
        help="Comma-separated raw prediction thresholds to sweep.",
    )
    parser.add_argument("--range-scale-m", type=float)
    parser.add_argument("--intensity-transform", choices=("raw", "log1p"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run one read-only checkpoint threshold sweep and save a JSON report."""
    args = parse_args()
    if args.dataset_role != "validation":
        raise ValueError("threshold calibration is validation-only; use the controlled evaluator for test")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("max-frames must be positive")
    if args.target_threshold < 0:
        raise ValueError("target-threshold must be non-negative")
    prediction_thresholds = parse_thresholds(args.prediction_thresholds)
    if args.output_json.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {args.output_json}; pass --overwrite")

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
    frame_count = len(dataset) if args.max_frames is None else min(args.max_frames, len(dataset))
    if frame_count < 1:
        raise ValueError("no validation samples")

    model = build_waymo_16_to_32_model(args.tulip_root)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()
    accumulated = {
        threshold: {
            "target_positive": 0,
            "predicted_positive": 0,
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
        }
        for threshold in prediction_thresholds
    }
    total_valid = 0
    positive_abs_error_sum = 0.0
    positive_count = 0

    with torch.inference_mode():
        for sample_index in range(frame_count):
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
            target_raw = restore_raw_domain(
                target_signal,
                range_scale_m=range_scale_m,
                intensity_transform=intensity_transform,
            )[:, 1, :, :NATIVE_WIDTH]
            prediction_raw = restore_raw_domain(
                prediction,
                range_scale_m=range_scale_m,
                intensity_transform=intensity_transform,
            )[:, 1, :, :NATIVE_WIDTH].clamp_min(0.0)
            mask = generated_mask[..., :NATIVE_WIDTH]
            total_valid += int(mask.sum())
            positive = mask & (target_raw >= args.target_threshold)
            positive_count += int(positive.sum())
            positive_abs_error_sum += float((prediction_raw - target_raw).abs()[positive].sum())
            for threshold in prediction_thresholds:
                counts = threshold_counts(
                    target_raw,
                    prediction_raw,
                    mask,
                    target_threshold=args.target_threshold,
                    prediction_threshold=threshold,
                )
                for name, value in counts.items():
                    accumulated[threshold][name] += value
            print(json.dumps({"evaluated": sample_index + 1, "frames": frame_count}), flush=True)

    rows: List[Dict[str, Any]] = []
    for threshold in prediction_thresholds:
        counts = accumulated[threshold]
        rows.append({
            "prediction_threshold": threshold,
            **counts,
            **metrics_from_counts(counts),
        })
    suggested = best_f1_candidate(rows)
    summary: Dict[str, Any] = {
        "purpose": "validation-only calibration of prediction threshold for a fixed raw GT event",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset_role": args.dataset_role,
        "evaluated_frames": frame_count,
        "mask_scope": "valid_generated_odd_rows_only",
        "range_scale_m": range_scale_m,
        "intensity_transform": intensity_transform,
        "target_threshold_raw": args.target_threshold,
        "target_positive_count": positive_count,
        "target_positive_fraction": positive_count / total_valid,
        "raw_intensity_mae_on_target_positive": (
            positive_abs_error_sum / positive_count if positive_count else None
        ),
        "prediction_threshold_candidates": rows,
        "suggested_by_max_validation_f1": suggested,
        "warning": "Selection is based on validation only. Do not tune from test results.",
    }
    atomic_json(args.output_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
