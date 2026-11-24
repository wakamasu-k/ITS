#!/usr/bin/env python3
"""Compare newly covered B/TULIP pixels to the reference static-map surface.

This audit does not modify maps or rendered images. It uses the saved pixel-to-
map-point indices and depth buffers, and reports 3D distances only where the
reference image has a valid depth at the exact same pixel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment", required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("/mnt/w/32line/datasets/submaps/cross/original_val_test_all/db/training"),
        help="Root of Shimizu reference DB anchors, grouped by segment/anchor.",
    )
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    abc = project / "outputs/shimizu_protocol_507q_754db_abc"
    segment = args.segment
    anchor = str(args.anchor)
    a_dir = abc / "anchors_A/training" / segment / anchor
    b_dir = abc / "anchors_B/training" / segment / anchor
    ref_dir = args.reference_root / segment / anchor
    b_map_path = abc / "maps/B_tulip32/training" / segment / "map_static.npz"

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    for path in (a_dir / "anchor_index.npz", b_dir / "anchor_index.npz",
                 ref_dir / "anchor_index.npz", b_map_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(a_dir / "anchor_index.npz", allow_pickle=False) as z:
        a_index = z["point_index"]
    with np.load(b_dir / "anchor_index.npz", allow_pickle=False) as z:
        b_index = z["point_index"]
    with np.load(ref_dir / "anchor_index.npz", allow_pickle=False) as z:
        ref_index, ref_depth = z["point_index"], z["depth"]
    with np.load(b_map_path, allow_pickle=False) as z:
        b_xyz = z["xyz"]
        is_generated = z["source_is_generated"].astype(bool)

    b_meta = load_json(b_dir / "anchor.json")
    ref_meta = load_json(ref_dir / "anchor.json")
    if not (a_index.shape == b_index.shape == ref_index.shape == ref_depth.shape):
        raise ValueError("A, B, and reference raster dimensions do not match")
    if not np.allclose(b_meta["intrinsic"], ref_meta["intrinsic"], rtol=0, atol=1e-5):
        raise ValueError("B and reference camera intrinsics differ")
    if not np.allclose(b_meta["world_to_cam"], ref_meta["world_to_cam"], rtol=0, atol=1e-5):
        raise ValueError("B and reference camera poses differ")

    b_valid = b_index >= 0
    if np.any(b_index[b_valid] >= len(is_generated)):
        raise ValueError("B pixel index is outside the B map")
    generated_pixels = np.zeros_like(b_valid)
    generated_pixels[b_valid] = is_generated[b_index[b_valid]]
    newly_covered = generated_pixels & (a_index < 0)
    ref_valid = (ref_index >= 0) & np.isfinite(ref_depth) & (ref_depth > 0)
    matched = newly_covered & ref_valid

    yy, xx = np.nonzero(matched)
    depth = ref_depth[yy, xx].astype(np.float64)
    fx, fy, cx, cy = map(float, ref_meta["intrinsic"])
    # The renderer uses forward depth X and camera coordinates
    # [X, -(u-cx)X/fx, -(v-cy)X/fy].
    xyz_cam = np.stack(
        [depth, -(xx - cx) * depth / fx, -(yy - cy) * depth / fy,
         np.ones_like(depth)], axis=1,
    )
    cam_to_world = np.linalg.inv(np.asarray(ref_meta["world_to_cam"], dtype=np.float64))
    xyz_ref_world = (xyz_cam @ cam_to_world.T)[:, :3]
    xyz_b = b_xyz[b_index[yy, xx].astype(np.int64)].astype(np.float64)
    error = np.linalg.norm(xyz_b - xyz_ref_world, axis=1)

    result = {
        "segment": segment,
        "anchor": anchor,
        "reference": str(ref_dir),
        "comparison": "B generated points at pixels newly covered relative to A; same-pixel reference depth",
        "new_generated_pixels": int(newly_covered.sum()),
        "same_pixel_reference_covered": int(matched.sum()),
        "same_pixel_reference_coverage_fraction": float(matched.sum() / newly_covered.sum()) if newly_covered.any() else None,
        "unmatched_new_pixels": int((newly_covered & ~ref_valid).sum()),
        "xyz_error_m": {
            "mean": float(error.mean()) if error.size else None,
            "median": float(np.median(error)) if error.size else None,
            "p90": float(np.percentile(error, 90)) if error.size else None,
            "p95": float(np.percentile(error, 95)) if error.size else None,
            "p99": float(np.percentile(error, 99)) if error.size else None,
            "max": float(error.max()) if error.size else None,
            "within_0.25m_fraction": float(np.mean(error <= 0.25)) if error.size else None,
            "within_0.50m_fraction": float(np.mean(error <= 0.50)) if error.size else None,
            "within_1.00m_fraction": float(np.mean(error <= 1.00)) if error.size else None,
            "over_2m_fraction": float(np.mean(error > 2.0)) if error.size else None,
        },
        "limitations": [
            "Only new generated pixels with a valid reference depth at the exact same pixel are scored.",
            "Unmatched pixels are unknown, not automatically incorrect.",
            "Reference is the Shimizu static-map rendering, not independent survey ground truth.",
            "Raster pixels are weighted; a rendered source point may occupy multiple pixels.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
