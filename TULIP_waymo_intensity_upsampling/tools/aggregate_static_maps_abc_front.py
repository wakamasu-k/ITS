#!/usr/bin/env python3
"""Aggregate per-anchor A/B/C FRONT comparison JSON reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = []
    for frame in args.frames:
        path = args.input_root / f"comparison_front_frame{frame}" / "comparison.json"
        reports.append(json.loads(path.read_text(encoding="utf-8")))

    metrics = (
        "coverage_recall_vs_reference", "coverage_iou_vs_reference",
        "depth_mae_m", "depth_rmse_m", "log_intensity_mae", "log_intensity_rmse",
        "additional_pixels_vs_reference", "missing_pixels_vs_reference",
    )
    result = {
        "frames": args.frames,
        "num_anchors": len(reports),
        "per_frame": [
            {
                "frame_index": report["frame_index"],
                "A_unique_pixels": report["variants"]["A"]["unique_pixels"],
                "B_unique_pixels": report["variants"]["B"]["unique_pixels"],
                "C_unique_pixels": report["variants"]["C"]["unique_pixels"],
                "A_vs_C": report["A_vs_C"],
                "B_vs_C": report["B_vs_C"],
            }
            for report in reports
        ],
        "mean_unique_pixels": {
            key: float(np.mean([r["variants"][key]["unique_pixels"] for r in reports]))
            for key in ("A", "B", "C")
        },
        "mean_A_vs_C": {
            metric: float(np.mean([r["A_vs_C"][metric] for r in reports]))
            for metric in metrics
        },
        "mean_B_vs_C": {
            metric: float(np.mean([r["B_vs_C"][metric] for r in reports]))
            for metric in metrics
        },
    }
    tolerant_reports = [r.get("B_vs_C_tolerant_3px") for r in reports]
    if all(item is not None for item in tolerant_reports):
        count_fields = (
            "candidate_pixels", "exact_same_pixel", "matched_within_radius",
            "good_depth_le_0_5m", "caution_depth_0_5_to_1m",
            "poor_depth_1_to_2m", "severe_depth_gt_2m",
            "no_reference_within_radius", "reference_only_exact_pixel",
        )
        result["mean_B_vs_C_tolerant_3px"] = {
            field: float(np.mean([item[field] for item in tolerant_reports]))
            for field in count_fields
        }
        total_candidate = sum(item["candidate_pixels"] for item in tolerant_reports)
        total_matched = sum(item["matched_within_radius"] for item in tolerant_reports)
        result["mean_B_vs_C_tolerant_3px"]["aggregate_matched_fraction"] = float(
            total_matched / max(1, total_candidate)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
