#!/usr/bin/env python3
"""Compare aligned A/B/C anchor databases using saved pixel-depth indices."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-root", required=True, type=Path)
    parser.add_argument("--b-root", required=True, type=Path)
    parser.add_argument("--c-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--radius-px", type=int, default=3)
    return parser.parse_args()


def load_index(root: Path, anchor_id: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(root / str(anchor_id) / "anchor_index.npz")
    return data["point_index"] >= 0, data["depth"].astype(np.float32)


def nearest_reference_depth(
    candidate: np.ndarray, reference: np.ndarray, reference_depth: np.ndarray, radius: int
) -> tuple[np.ndarray, np.ndarray]:
    height, width = candidate.shape
    best_distance = np.full((height, width), np.inf, dtype=np.float32)
    best_depth = np.full((height, width), np.nan, dtype=np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            distance = float(dx * dx + dy * dy)
            if distance > radius * radius:
                continue
            y0, y1 = max(0, -dy), min(height, height - dy)
            x0, x1 = max(0, -dx), min(width, width - dx)
            shifted_valid = reference[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
            target = candidate[y0:y1, x0:x1]
            improve = target & shifted_valid & (distance < best_distance[y0:y1, x0:x1])
            current_distance = best_distance[y0:y1, x0:x1]
            current_depth = best_depth[y0:y1, x0:x1]
            current_distance[improve] = distance
            source_depth = reference_depth[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
            current_depth[improve] = source_depth[improve]
    matched = candidate & np.isfinite(best_distance)
    return matched, best_depth


def load_pose(root: Path, anchor_id: int) -> np.ndarray:
    meta = json.loads((root / str(anchor_id) / "anchor.json").read_text(encoding="utf-8"))
    if "anchor_pose_xyyaw" in meta:
        pose = meta["anchor_pose_xyyaw"]
        return np.asarray([pose["x"], pose["y"], pose["yaw_deg"]], dtype=np.float64)
    return np.asarray(meta["anchor_pose_T_wv"], dtype=np.float64)


def main() -> None:
    args = parse_args()
    ids = sorted(int(path.name) for path in args.a_root.iterdir() if path.is_dir() and path.name.isdigit())
    if ids != sorted(int(path.name) for path in args.b_root.iterdir() if path.is_dir() and path.name.isdigit()):
        raise ValueError("A and B anchor IDs differ")
    if ids != sorted(int(path.name) for path in args.c_root.iterdir() if path.is_dir() and path.name.isdigit()):
        raise ValueError("A and C anchor IDs differ")

    rows: list[dict[str, object]] = []
    for anchor_id in ids:
        a_mask, _ = load_index(args.a_root, anchor_id)
        b_mask, b_depth = load_index(args.b_root, anchor_id)
        c_mask, c_depth = load_index(args.c_root, anchor_id)
        matched, nearest_c_depth = nearest_reference_depth(
            b_mask, c_mask, c_depth, args.radius_px
        )
        depth_error = np.abs(b_depth[matched] - nearest_c_depth[matched])
        finite = np.isfinite(depth_error)
        depth_error = depth_error[finite]
        b_count, c_count = int(b_mask.sum()), int(c_mask.sum())
        overlap = int((b_mask & c_mask).sum())
        union = int((b_mask | c_mask).sum())
        matched_count = int(matched.sum())
        poses = [load_pose(root, anchor_id) for root in (args.a_root, args.b_root, args.c_root)]
        rows.append({
            "anchor_id": anchor_id,
            "a_pixels": int(a_mask.sum()),
            "b_pixels": b_count,
            "c_pixels": c_count,
            "b_added_vs_a": int((b_mask & ~a_mask).sum()),
            "b_c_exact_overlap": overlap,
            "b_c_exact_iou": overlap / union if union else 0.0,
            "b_c_matched_within_radius": matched_count,
            "b_c_matched_fraction": matched_count / b_count if b_count else 0.0,
            "depth_le_0_5m": int((depth_error <= 0.5).sum()),
            "depth_0_5_to_1m": int(((depth_error > 0.5) & (depth_error <= 1.0)).sum()),
            "depth_1_to_2m": int(((depth_error > 1.0) & (depth_error <= 2.0)).sum()),
            "depth_gt_2m": int((depth_error > 2.0).sum()),
            "depth_mae_m": float(depth_error.mean()) if depth_error.size else None,
            "pose_max_abs_delta": float(max(
                np.max(np.abs(poses[0] - poses[1])), np.max(np.abs(poses[0] - poses[2]))
            )),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_anchor.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    numeric = [key for key in rows[0] if key != "anchor_id" and rows[0][key] is not None]
    summary = {
        "anchors": len(rows),
        "radius_px": args.radius_px,
        "mean_per_anchor": {key: float(np.mean([float(row[key]) for row in rows])) for key in numeric},
        "min_matched_fraction": float(min(float(row["b_c_matched_fraction"]) for row in rows)),
        "max_pose_abs_delta": float(max(float(row["pose_max_abs_delta"]) for row in rows)),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
