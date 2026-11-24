#!/usr/bin/env python3
"""Evaluate high-intensity recovery by observed-ring support on validation.

For each generated 32-line pixel, the support group is derived solely from
the observed 16-line rows: exact same-column support, ±1-column-only support,
or no local support.  Each group retains both target positives and negatives,
so precision, recall and F1 remain meaningful.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.audit_waymo_16_to_32_high_intensity import _neighbor_support
from tools.calibrate_waymo_16_to_32_high_intensity_threshold import (
    metrics_from_counts,
    threshold_counts,
)
from tools.evaluate_waymo_16_to_32_controlled import (
    generated_only_mask,
    load_checkpoint,
    resolve_checkpoint_contract,
    restore_raw_domain,
)
from tulip_adapter.waymo_16_to_32_dataset import Waymo16To32Dataset
from tulip_adapter.waymo_16_to_32_tulip import build_waymo_16_to_32_model


NATIVE_WIDTH = 2650
GROUPS = ("exact_support", "local_only_support", "no_local_support")


def support_groups(
    target_32: np.ndarray,
    target_valid_mask_32: np.ndarray,
    observed_rows: np.ndarray,
    generated_rows: np.ndarray,
    *,
    high_threshold: float,
) -> Dict[str, np.ndarray]:
    """Return three mutually exclusive generated-pixel support masks.

    The raw GT intensity from observed rows is used only to identify information
    already present in the 16-line input.  The target intensity of a generated
    pixel is not used to assign its support group.
    """
    target = np.asarray(target_32, dtype=np.float32)
    valid = np.asarray(target_valid_mask_32, dtype=bool)
    if target.ndim != 3 or target.shape[2] != 2:
        raise ValueError("target_32 must be [32,W,2]")
    if valid.shape != target.shape[:2]:
        raise ValueError("target_valid_mask_32 shape mismatch")
    if not np.isfinite(target).all():
        raise ValueError("target_32 contains NaN or Inf")
    if high_threshold < 0:
        raise ValueError("high_threshold must be non-negative")
    observed = np.asarray(observed_rows, dtype=np.int64)
    generated = np.asarray(generated_rows, dtype=np.int64)
    if set(observed.tolist()) & set(generated.tolist()):
        raise ValueError("observed and generated rows overlap")
    if sorted(np.concatenate((observed, generated)).tolist()) != list(range(target.shape[0])):
        raise ValueError("observed and generated rows do not cover target height")
    exact, local = _neighbor_support(target[..., 1], valid, generated, high_threshold)
    generated_domain = np.zeros_like(valid)
    generated_domain[generated] = True
    exact &= generated_domain
    local &= generated_domain
    return {
        "exact_support": exact,
        "local_only_support": local & ~exact,
        "no_local_support": generated_domain & ~local,
    }


def merge_counts(accumulated: Dict[str, int], current: Dict[str, int]) -> None:
    """Add one binary-count dictionary in place."""
    for key in ("target_positive", "predicted_positive", "true_positive", "false_positive", "false_negative"):
        accumulated[key] += int(current[key])


def group_result(
    counts: Dict[str, int], *, raw_abs_sum_on_target_positive: float
) -> Dict[str, Any]:
    """Attach detection scores and positive-target raw MAE to one group."""
    positives = counts["target_positive"]
    return {
        **counts,
        **metrics_from_counts(counts),
        "raw_intensity_mae_on_target_positive": (
            raw_abs_sum_on_target_positive / positives if positives else None
        ),
    }


def atomic_json(path: Path, value: Dict[str, Any]) -> None:
    """Atomically write a JSON evaluation report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    """Parse a validation-only support-stratified evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--tulip-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--dataset-role", choices=("validation",), default="validation")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--target-threshold", type=float, default=1000.0)
    parser.add_argument("--prediction-threshold", type=float, default=500.0)
    parser.add_argument("--range-scale-m", type=float)
    parser.add_argument("--intensity-transform", choices=("raw", "log1p"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_pair_support(path: Path, *, high_threshold: float) -> Dict[str, np.ndarray]:
    """Load raw support masks from a pair NPZ without modifying it."""
    with np.load(path, allow_pickle=False) as data:
        required = {
            "target_32", "target_valid_mask_32", "observed_target_rows",
            "generated_target_rows",
        }
        missing = required - set(data.files)
        if missing:
            raise KeyError("pair is missing arrays: {}".format(sorted(missing)))
        return support_groups(
            data["target_32"],
            data["target_valid_mask_32"],
            data["observed_target_rows"],
            data["generated_target_rows"],
            high_threshold=high_threshold,
        )


def main() -> None:
    """Evaluate a checkpoint separately on the three observed-support groups."""
    args = parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("max-frames must be positive")
    if args.target_threshold < 0 or args.prediction_threshold < 0:
        raise ValueError("thresholds must be non-negative")
    if args.output_json.exists() and not args.overwrite:
        raise FileExistsError("output exists: {}; pass --overwrite".format(args.output_json))
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
        dataset_role="validation",
        intensity_transform=intensity_transform,
        range_scale_m=range_scale_m,
    )
    frame_count = len(dataset) if args.max_frames is None else min(args.max_frames, len(dataset))
    if frame_count < 1:
        raise ValueError("no validation samples")
    model = build_waymo_16_to_32_model(args.tulip_root)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()
    counts = {
        group: {name: 0 for name in (
            "target_positive", "predicted_positive", "true_positive",
            "false_positive", "false_negative",
        )}
        for group in GROUPS
    }
    positive_abs_sums = {group: 0.0 for group in GROUPS}
    group_pixel_counts = {group: 0 for group in GROUPS}

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
                raise RuntimeError("non-finite prediction at sample {}".format(sample_index))
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
            pair_value = Path(dataset.rows[sample_index]["pair_npz"])
            pair_path = pair_value if pair_value.is_absolute() else ROOT / pair_value
            raw_groups = _load_pair_support(pair_path, high_threshold=args.target_threshold)
            for group in GROUPS:
                support = torch.from_numpy(raw_groups[group]).to(device=device, dtype=torch.bool)
                selected = mask & support.unsqueeze(0)
                group_pixel_counts[group] += int(selected.sum())
                current = threshold_counts(
                    target_raw,
                    prediction_raw,
                    selected,
                    target_threshold=args.target_threshold,
                    prediction_threshold=args.prediction_threshold,
                )
                merge_counts(counts[group], current)
                positives = selected & (target_raw >= args.target_threshold)
                positive_abs_sums[group] += float((prediction_raw - target_raw).abs()[positives].sum())
            print(json.dumps({"evaluated": sample_index + 1, "frames": frame_count}), flush=True)

    groups = {
        group: {
            "evaluation_pixels": group_pixel_counts[group],
            **group_result(counts[group], raw_abs_sum_on_target_positive=positive_abs_sums[group]),
        }
        for group in GROUPS
    }
    positive_total = sum(counts[group]["target_positive"] for group in GROUPS)
    summary: Dict[str, Any] = {
        "purpose": "validation-only high-reflection detection separated by observed-ring support",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset_role": "validation",
        "evaluated_frames": frame_count,
        "mask_scope": "valid_generated_odd_rows_only",
        "target_threshold_raw": args.target_threshold,
        "prediction_threshold_raw": args.prediction_threshold,
        "support_definition": {
            "exact_support": "adjacent observed row at the same azimuth column is >= target threshold",
            "local_only_support": "not exact; adjacent observed row within azimuth column ±1 is >= target threshold",
            "no_local_support": "no adjacent observed row within azimuth column ±1 is >= target threshold",
        },
        "groups": groups,
        "high_target_positive_count": positive_total,
        "warning": "Validation only. Use these results to decide whether another training condition is justified; do not select from test.",
    }
    if positive_total == 0:
        raise ValueError("no target high-intensity pixels found")
    if positive_total != sum(groups[group]["target_positive"] for group in GROUPS):
        raise RuntimeError("support groups do not partition high-intensity targets")
    atomic_json(args.output_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
