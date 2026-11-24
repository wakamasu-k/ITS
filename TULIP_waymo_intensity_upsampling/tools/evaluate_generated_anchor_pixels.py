#!/usr/bin/env python3
"""Visualize and evaluate visible TULIP-generated anchor pixels against GT."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b-root", required=True, type=Path)
    parser.add_argument("--c-root", required=True, type=Path)
    parser.add_argument("--b-map", required=True, type=Path)
    parser.add_argument("--c-map", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--radius-px", type=int, default=3)
    return parser.parse_args()


def nearest_reference(
    candidate: np.ndarray,
    reference_index: np.ndarray,
    reference_depth: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return matched mask plus depth/index of the nearest GT pixel."""
    height, width = candidate.shape
    best_distance = np.full((height, width), np.inf, dtype=np.float32)
    best_depth = np.full((height, width), np.nan, dtype=np.float32)
    best_index = np.full((height, width), -1, dtype=np.int64)
    reference_valid = reference_index >= 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            distance = float(dx * dx + dy * dy)
            if distance > radius * radius:
                continue
            y0, y1 = max(0, -dy), min(height, height - dy)
            x0, x1 = max(0, -dx), min(width, width - dx)
            ref_slice = np.s_[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
            dst_slice = np.s_[y0:y1, x0:x1]
            improve = (
                candidate[dst_slice]
                & reference_valid[ref_slice]
                & (distance < best_distance[dst_slice])
            )
            best_distance[dst_slice][improve] = distance
            best_depth[dst_slice][improve] = reference_depth[ref_slice][improve]
            best_index[dst_slice][improve] = reference_index[ref_slice][improve]
    matched = candidate & np.isfinite(best_distance)
    return matched, best_depth, best_index


def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def main() -> None:
    args = parse_args()
    if args.radius_px < 0:
        raise ValueError("--radius-px must be non-negative")
    b_map = np.load(args.b_map)
    c_map = np.load(args.c_map)
    if "source_is_generated" not in b_map.files:
        raise ValueError("B map has no source_is_generated array")
    b_intensity = b_map["intensity"].astype(np.float64)
    c_intensity = c_map["intensity"].astype(np.float64)
    source_is_generated = b_map["source_is_generated"].astype(bool)
    if source_is_generated.shape != b_intensity.shape:
        raise ValueError("B provenance and intensity shapes differ")

    b_ids = sorted(int(p.name) for p in args.b_root.iterdir() if p.is_dir() and p.name.isdigit())
    c_ids = sorted(int(p.name) for p in args.c_root.iterdir() if p.is_dir() and p.name.isdigit())
    if b_ids != c_ids:
        raise ValueError("B and C anchor IDs differ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    all_depth_error: list[np.ndarray] = []
    all_b_log_intensity: list[np.ndarray] = []
    all_c_log_intensity: list[np.ndarray] = []

    for anchor_id in b_ids:
        b_dir, c_dir = args.b_root / str(anchor_id), args.c_root / str(anchor_id)
        b_index_data = np.load(b_dir / "anchor_index.npz")
        c_index_data = np.load(c_dir / "anchor_index.npz")
        b_index = b_index_data["point_index"].astype(np.int64)
        b_depth = b_index_data["depth"].astype(np.float32)
        c_index = c_index_data["point_index"].astype(np.int64)
        c_depth = c_index_data["depth"].astype(np.float32)
        b_valid = b_index >= 0
        generated = np.zeros_like(b_valid)
        generated[b_valid] = source_is_generated[b_index[b_valid]]
        observed = b_valid & ~generated

        matched, nearest_c_depth, nearest_c_index = nearest_reference(
            generated, c_index, c_depth, args.radius_px
        )
        depth_error = np.abs(b_depth[matched] - nearest_c_depth[matched]).astype(np.float64)
        b_raw = b_intensity[b_index[matched]]
        c_raw = c_intensity[nearest_c_index[matched]]
        b_log, c_log = np.log1p(np.maximum(b_raw, 0.0)), np.log1p(np.maximum(c_raw, 0.0))
        all_depth_error.append(depth_error)
        all_b_log_intensity.append(b_log)
        all_c_log_intensity.append(c_log)

        gray = np.asarray(Image.open(b_dir / "anchor.png").convert("L"))
        overlay = np.repeat(gray[:, :, None], 3, axis=2)
        observed_gray = (gray[observed].astype(np.float32) * 0.55).astype(np.uint8)
        overlay[observed] = np.repeat(observed_gray[:, None], 3, axis=1)
        overlay[generated] = np.array([255, 32, 32], dtype=np.uint8)
        Image.fromarray(overlay).save(args.output_dir / f"anchor_{anchor_id:03d}_generated_red.png")

        rows.append({
            "anchor_id": anchor_id,
            "b_visible_pixels": int(b_valid.sum()),
            "observed_visible_pixels": int(observed.sum()),
            "generated_visible_pixels": int(generated.sum()),
            "generated_fraction": float(generated.sum() / b_valid.sum()) if b_valid.any() else 0.0,
            "generated_matched_gt_pixels": int(matched.sum()),
            "generated_matched_fraction": float(matched.sum() / generated.sum()) if generated.any() else 0.0,
            "depth_mae_m": float(depth_error.mean()) if depth_error.size else None,
            "depth_median_m": float(np.median(depth_error)) if depth_error.size else None,
            "depth_le_0_5m_fraction": float((depth_error <= 0.5).mean()) if depth_error.size else None,
            "depth_gt_2m_fraction": float((depth_error > 2.0).mean()) if depth_error.size else None,
            "log1p_intensity_mae": float(np.mean(np.abs(b_log - c_log))) if b_log.size else None,
            "log1p_intensity_rmse": float(np.sqrt(np.mean((b_log - c_log) ** 2))) if b_log.size else None,
            "log1p_intensity_correlation": correlation(b_log, c_log),
        })

    fields = list(rows[0])
    with (args.output_dir / "per_anchor.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    depth_error = np.concatenate(all_depth_error)
    b_log = np.concatenate(all_b_log_intensity)
    c_log = np.concatenate(all_c_log_intensity)
    total_generated = sum(int(row["generated_visible_pixels"]) for row in rows)
    total_matched = sum(int(row["generated_matched_gt_pixels"]) for row in rows)
    summary = {
        "anchors": len(rows),
        "radius_px": args.radius_px,
        "total_generated_visible_pixels": total_generated,
        "total_generated_matched_gt_pixels": total_matched,
        "generated_matched_fraction": total_matched / total_generated if total_generated else 0.0,
        "depth_mae_m": float(depth_error.mean()),
        "depth_median_m": float(np.median(depth_error)),
        "depth_le_0_5m_fraction": float((depth_error <= 0.5).mean()),
        "depth_gt_2m_fraction": float((depth_error > 2.0).mean()),
        "log1p_intensity_mae": float(np.mean(np.abs(b_log - c_log))),
        "log1p_intensity_rmse": float(np.sqrt(np.mean((b_log - c_log) ** 2))),
        "log1p_intensity_correlation": correlation(b_log, c_log),
        "overlay_legend": {"red": "visible TULIP-generated point", "gray": "visible measured point"},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
