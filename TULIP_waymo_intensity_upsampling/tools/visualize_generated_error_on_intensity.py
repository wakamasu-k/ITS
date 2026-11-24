#!/usr/bin/env python3
"""Overlay generated-point 3D error classes on B/reference intensity images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def describe_values(values: np.ndarray) -> dict:
    values = values.astype(np.float64)
    return {
        "pixels": int(values.size),
        "nonzero_fraction": float(np.mean(values > 0)) if values.size else None,
        "mean_gray_0_255": float(values.mean()) if values.size else None,
        "median_gray_0_255": float(np.median(values)) if values.size else None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--segment", required=True)
    p.add_argument("--anchor", required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument(
        "--reference-root",
        type=Path,
        default=Path("/mnt/w/32line/datasets/submaps/cross/original_val_test_all/db/training"),
    )
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")

    project = Path(__file__).resolve().parents[1]
    abc = project / "outputs/shimizu_protocol_507q_754db_abc"
    seg, aid = args.segment, str(args.anchor)
    a_dir = abc / "anchors_A/training" / seg / aid
    b_dir = abc / "anchors_B/training" / seg / aid
    ref_dir = args.reference_root / seg / aid
    map_path = abc / "maps/B_tulip32/training" / seg / "map_static.npz"

    with np.load(a_dir / "anchor_index.npz", allow_pickle=False) as z:
        a_idx = z["point_index"]
    with np.load(b_dir / "anchor_index.npz", allow_pickle=False) as z:
        b_idx = z["point_index"]
    with np.load(ref_dir / "anchor_index.npz", allow_pickle=False) as z:
        ref_idx, ref_depth = z["point_index"], z["depth"]
    with np.load(map_path, allow_pickle=False) as z:
        b_xyz, is_generated = z["xyz"], z["source_is_generated"].astype(bool)
    b_meta = json.loads((b_dir / "anchor.json").read_text(encoding="utf-8"))
    ref_meta = json.loads((ref_dir / "anchor.json").read_text(encoding="utf-8"))
    if not np.allclose(b_meta["intrinsic"], ref_meta["intrinsic"], rtol=0, atol=1e-5):
        raise ValueError("B/reference camera intrinsics differ")
    if not np.allclose(b_meta["world_to_cam"], ref_meta["world_to_cam"], rtol=0, atol=1e-5):
        raise ValueError("B/reference camera poses differ")

    b_img = cv2.imread(str(b_dir / "anchor.png"), cv2.IMREAD_GRAYSCALE)
    ref_img = cv2.imread(str(ref_dir / "anchor.png"), cv2.IMREAD_GRAYSCALE)
    if b_img is None or ref_img is None:
        raise FileNotFoundError("Could not load B or reference intensity image")
    b_valid = b_idx >= 0
    generated_pixels = np.zeros_like(b_valid)
    generated_pixels[b_valid] = is_generated[b_idx[b_valid]]
    newly = generated_pixels & (a_idx < 0)
    ref_valid = (ref_idx >= 0) & np.isfinite(ref_depth) & (ref_depth > 0)
    matched = newly & ref_valid

    y, x = np.nonzero(matched)
    depth = ref_depth[y, x].astype(np.float64)
    fx, fy, cx, cy = map(float, ref_meta["intrinsic"])
    camera_points = np.stack(
        [depth, -(x-cx)*depth/fx, -(y-cy)*depth/fy, np.ones_like(depth)], axis=1
    )
    world_points = (camera_points @ np.linalg.inv(np.asarray(ref_meta["world_to_cam"], dtype=np.float64)).T)[:, :3]
    b_points = b_xyz[b_idx[y, x].astype(np.int64)].astype(np.float64)
    xyz_error = np.linalg.norm(b_points - world_points, axis=1)
    pixel_error = np.abs(b_img[y, x].astype(np.int16) - ref_img[y, x].astype(np.int16))

    high = xyz_error > 2.0
    far = depth >= 40.0
    far_high = high & far
    good = ~high

    h, w = b_idx.shape
    region_names = [["top_left", "top_center", "top_right"],
                    ["middle_left", "center", "middle_right"],
                    ["bottom_left", "bottom_center", "bottom_right"]]
    region_rows = np.minimum(y * 3 // h, 2)
    region_cols = np.minimum(x * 3 // w, 2)
    regions = {}
    for row in range(3):
        for col in range(3):
            m = (region_rows == row) & (region_cols == col)
            regions[region_names[row][col]] = {
                "matched_new_pixels": int(m.sum()),
                "xyz_error_over_2m_pixels": int(np.sum(high[m])),
                "xyz_error_over_2m_fraction": float(np.mean(high[m])) if m.any() else None,
                "over_2m_and_depth_at_least_40m_pixels": int(np.sum(far_high[m])),
            }

    by_depth = {}
    b_gray = b_img[y, x]
    ref_gray = ref_img[y, x]
    for lo, hi in [(0, 10), (10, 20), (20, 40), (40, 80), (80, float("inf"))]:
        m = (depth >= lo) & (depth < hi)
        name = f"{lo}_{hi:g}m" if np.isfinite(hi) else f"{lo}m_plus"
        by_depth[name] = {
            "matched_pixels": int(m.sum()),
            "xyz_error_over_2m_pixels": int(np.sum(high[m])),
            "xyz_error_over_2m_fraction": float(np.mean(high[m])) if m.any() else None,
            "b_intensity": describe_values(b_gray[m]),
            "reference_intensity": describe_values(ref_gray[m]),
            "mean_abs_png_intensity_difference": float(pixel_error[m].mean()) if m.any() else None,
        }

    # Build RGB overlay on the B anchor image. Colors identify geometry classes;
    # source intensity remains visible as a dim grayscale background.
    base = (b_img.astype(np.float32) * 0.40).clip(0, 255).astype(np.uint8)
    overlay = np.repeat(base[:, :, None], 3, axis=2)
    overlay[y[good], x[good]] = (40, 210, 70)          # green: <=2 m error
    overlay[y[high & ~far], x[high & ~far]] = (255, 110, 0)  # orange: >2 m, <40 m
    overlay[y[far_high], x[far_high]] = (255, 0, 210)  # magenta: >2 m and >=40 m

    # A second panel displays the reference intensity; the third is the B overlay.
    ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2RGB)
    b_rgb = cv2.cvtColor(b_img, cv2.COLOR_GRAY2RGB)
    panel = np.concatenate([b_rgb, ref_rgb, overlay], axis=1)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    image_path = args.output_dir / "intensity_and_geometry_error_overlay.png"
    cv2.imwrite(str(image_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    result = {
        "segment": seg,
        "anchor": aid,
        "new_generated_pixels": int(newly.sum()),
        "reference_matched_new_pixels": int(matched.sum()),
        "unmatched_new_pixels_not_scored": int((newly & ~ref_valid).sum()),
        "high_geometry_error_threshold_m": 2.0,
        "far_depth_threshold_m": 40.0,
        "xyz_error_over_2m_pixels": int(high.sum()),
        "xyz_error_over_2m_fraction": float(high.mean()) if high.size else None,
        "far_and_xyz_error_over_2m_pixels": int(far_high.sum()),
        "intensity_on_error_classes": {
            "geometry_error_at_most_2m": {
                "B_png": describe_values(b_gray[good]),
                "reference_png": describe_values(ref_gray[good]),
                "mean_abs_B_vs_reference_png_difference": float(pixel_error[good].mean()) if good.any() else None,
            },
            "geometry_error_over_2m": {
                "B_png": describe_values(b_gray[high]),
                "reference_png": describe_values(ref_gray[high]),
                "mean_abs_B_vs_reference_png_difference": float(pixel_error[high].mean()) if high.any() else None,
            },
            "far_and_geometry_error_over_2m": {
                "B_png": describe_values(b_gray[far_high]),
                "reference_png": describe_values(ref_gray[far_high]),
                "mean_abs_B_vs_reference_png_difference": float(pixel_error[far_high].mean()) if far_high.any() else None,
            },
        },
        "by_depth": by_depth,
        "by_image_region_3x3": regions,
        "overlay": str(image_path),
        "legend": {
            "panel_order": ["B intensity PNG", "Shimizu reference intensity PNG", "B image with geometry classes"],
            "green": "new generated pixel with 3D error <=2 m",
            "orange": "new generated pixel with 3D error >2 m and reference depth <40 m",
            "magenta": "new generated pixel with 3D error >2 m and reference depth >=40 m",
            "background_black": "no matched newly covered generated point",
        },
        "caveat": "B/reference PNG values use the same projection and postprocess settings but different map intensity normalization upper values; pixel brightness differences are descriptive, not raw reflectance errors.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
