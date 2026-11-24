#!/usr/bin/env python3
"""Break down new B-point XYZ errors by reference range and image region."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def stats(errors: np.ndarray) -> dict:
    if not errors.size:
        return {"pixels": 0, "median_m": None, "p90_m": None, "over_2m_fraction": None}
    return {
        "pixels": int(errors.size),
        "median_m": float(np.median(errors)),
        "p90_m": float(np.percentile(errors, 90)),
        "over_2m_fraction": float(np.mean(errors > 2.0)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--segment", required=True)
    p.add_argument("--anchor", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--reference-root",
        type=Path,
        default=Path("/mnt/w/32line/datasets/submaps/cross/original_val_test_all/db/training"),
    )
    p.add_argument(
        "--retrieval-jsonl",
        type=Path,
        default=Path("outputs/shimizu_protocol_507q_754db_abc/retrieval_B/shimizu_507q_754db_B_topk_top50.jsonl"),
    )
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")

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
        xyz, generated = z["xyz"], z["source_is_generated"].astype(bool)
    b_meta = json.loads((b_dir / "anchor.json").read_text(encoding="utf-8"))
    ref_meta = json.loads((ref_dir / "anchor.json").read_text(encoding="utf-8"))
    if not np.allclose(b_meta["intrinsic"], ref_meta["intrinsic"], rtol=0, atol=1e-5):
        raise ValueError("B/reference camera intrinsics differ")
    if not np.allclose(b_meta["world_to_cam"], ref_meta["world_to_cam"], rtol=0, atol=1e-5):
        raise ValueError("B/reference camera poses differ")

    b_valid = b_idx >= 0
    generated_pixels = np.zeros_like(b_valid)
    generated_pixels[b_valid] = generated[b_idx[b_valid]]
    newly = generated_pixels & (a_idx < 0)
    ref_valid = (ref_idx >= 0) & np.isfinite(ref_depth) & (ref_depth > 0)
    matched = newly & ref_valid
    y, x = np.nonzero(matched)
    d = ref_depth[y, x].astype(np.float64)
    fx, fy, cx, cy = map(float, ref_meta["intrinsic"])
    xyz_cam = np.stack([d, -(x-cx)*d/fx, -(y-cy)*d/fy, np.ones_like(d)], axis=1)
    xyz_gt = (xyz_cam @ np.linalg.inv(np.asarray(ref_meta["world_to_cam"], dtype=np.float64)).T)[:, :3]
    xyz_b = xyz[b_idx[y, x].astype(np.int64)].astype(np.float64)
    error = np.linalg.norm(xyz_b - xyz_gt, axis=1)

    # Range bins are forward depth in the reference camera.
    range_edges = [0, 10, 20, 40, 80, 120, float("inf")]
    by_range = {}
    for lo, hi in zip(range_edges[:-1], range_edges[1:]):
        m = (d >= lo) & (d < hi)
        label = f"{lo:g}_{hi:g}m" if np.isfinite(hi) else f"{lo:g}m_plus"
        by_range[label] = stats(error[m])

    h, w = b_idx.shape
    row = np.minimum(y * 3 // h, 2)
    col = np.minimum(x * 3 // w, 2)
    by_region = {}
    names = [["top_left", "top_center", "top_right"],
             ["middle_left", "center", "middle_right"],
             ["bottom_left", "bottom_center", "bottom_right"]]
    for r in range(3):
        for c in range(3):
            m = (row == r) & (col == c)
            region = names[r][c]
            values = stats(error[m])
            values["over_2m_pixels"] = int(np.sum(error[m] > 2.0))
            by_region[region] = values

    image_key = f"/{seg}/{aid}/anchor.png"
    retrieval = None
    if args.retrieval_jsonl.is_file():
        hits = []
        with args.retrieval_jsonl.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                strict_path = rec.get("strict_gt", {}).get("anchor_png", "")
                if image_key in strict_path:
                    top1 = rec.get("topk", [{}])[0]
                    hits.append({
                        "query_index": rec.get("q_idx"),
                        "strict_gt_rank": rec.get("strict_gt", {}).get("strict_rank"),
                        "strict_gt_score": rec.get("strict_gt", {}).get("score"),
                        "top1_score": top1.get("score"),
                        "top1_anchor": top1.get("anchor_png"),
                        "top1_distance_m": top1.get("dist_m"),
                    })
        retrieval = hits

    result = {
        "segment": seg,
        "anchor": aid,
        "new_generated_pixels": int(newly.sum()),
        "reference_matched_pixels": int(matched.sum()),
        "reference_unmatched_new_pixels": int((newly & ~ref_valid).sum()),
        "overall_xyz_error": stats(error),
        "by_reference_forward_depth": by_range,
        "by_image_region_3x3": by_region,
        "retrieval_query_records_whose_gt_is_this_anchor": retrieval,
        "interpretation_note": "Only same-pixel reference-depth matches are scored; unmatched pixels are unknown, not errors.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
