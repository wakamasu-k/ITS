#!/usr/bin/env python3
"""Create Shimizu-retrieval CSV manifests from real anchor/query metadata."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam-dir", required=True, type=Path)
    parser.add_argument("--anchor-root", required=True, type=Path)
    parser.add_argument(
        "--pose-root", type=Path,
        help="optional aligned anchor root containing anchor_pose_T_wv metadata",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pair-id", required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    anchor_rows: list[dict[str, object]] = []
    anchor_xy: list[tuple[float, float]] = []
    for meta_path in sorted(args.anchor_root.glob("*/anchor.json"), key=lambda p: int(p.parent.name)):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        pose = meta["anchor_pose_xyyaw"]
        image = meta_path.with_name("anchor.png").resolve()
        if not image.exists():
            raise FileNotFoundError(image)
        pose_meta = meta
        if "anchor_pose_T_wv" not in pose_meta and args.pose_root is not None:
            pose_meta = json.loads(
                (args.pose_root / meta_path.parent.name / "anchor.json").read_text(encoding="utf-8")
            )
        if "anchor_pose_T_wv" not in pose_meta:
            raise ValueError(
                f"anchor_pose_T_wv missing in {meta_path}; provide --pose-root with aligned metadata"
            )
        transform = pose_meta["anchor_pose_T_wv"]
        anchor_xy.append((float(pose["x"]), float(pose["y"])))
        anchor_rows.append({
            "pair_id": args.pair_id,
            "db_subset": meta.get("subset", "training"),
            "db_seg": meta["segment_id"],
            "anchor_id": int(meta["submap_id"]),
            "anchor_png": str(image),
            "is_virtual": 0,
            "anchor_T_wv_rowmajor": json.dumps(transform, separators=(",", ":")),
            "anchor_x_db": float(pose["x"]),
            "anchor_y_db": float(pose["y"]),
            "anchor_z_db": float(transform[11]),
        })
    if not anchor_rows:
        raise ValueError(f"no anchor.json found under {args.anchor_root}")

    xy = np.asarray(anchor_xy, dtype=np.float64)
    pair_rows: list[dict[str, object]] = []
    distances: list[float] = []
    for cam_meta_path in sorted(args.cam_dir.glob("*_cam.json")):
        meta = json.loads(cam_meta_path.read_text(encoding="utf-8"))
        query_xy = np.asarray([meta["pose_xyyaw"]["x"], meta["pose_xyyaw"]["y"]])
        distance = np.linalg.norm(xy - query_xy[None, :], axis=1)
        nearest = int(np.argmin(distance))
        distances.append(float(distance[nearest]))
        cam_png = cam_meta_path.with_suffix(".png").resolve()
        if not cam_png.exists():
            raise FileNotFoundError(cam_png)
        pair_rows.append({
            "cam_png": str(cam_png),
            "anchor_png": anchor_rows[nearest]["anchor_png"],
            "gt_anchor_png": anchor_rows[nearest]["anchor_png"],
            "label": 1,
            "pair_id": args.pair_id,
            "q_subset": "training",
            "q_seg": meta["segment_id"],
            "anchor_id": anchor_rows[nearest]["anchor_id"],
            "frame_index": int(meta["frame_index"]),
            "gt_distance_m": float(distance[nearest]),
        })

    if not pair_rows:
        raise ValueError(f"no *_cam.json found under {args.cam_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = args.output_dir / "pairs.csv"
    db_path = args.output_dir / "db.csv"
    write_csv(pairs_path, pair_rows)
    write_csv(db_path, anchor_rows)
    summary = {
        "pair_id": args.pair_id,
        "queries": len(pair_rows),
        "anchors": len(anchor_rows),
        "nearest_anchor_distance_m": {
            "min": float(np.min(distances)),
            "median": float(np.median(distances)),
            "max": float(np.max(distances)),
        },
        "pairs_csv": str(pairs_path.resolve()),
        "db_csv": str(db_path.resolve()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
