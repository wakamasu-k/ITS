#!/usr/bin/env python3
"""Convert a retrieval top-K JSONL dump into the Shimizu LoFTR/PnP manifest."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


FIELDS = [
    "pair_id", "lane_tag", "db_seg", "q_seg", "db_subset", "q_subset",
    "q_frame_index", "anchor_id", "q_image_png", "q_meta_json", "q_depth_npz",
    "anchor_png", "anchor_json", "anchor_index_npz", "gt_npz", "dist_xy_m",
    "is_strict_gt", "is_gt_t1m", "is_gt_t2m", "is_gt_t5m", "is_gt_t10m",
    "n_matches", "n_fine", "w0", "h0", "w1", "h1", "error",
    "retrieval_q_idx", "retrieval_rank", "retrieval_score", "retrieval_db_index",
    "retrieval_is_same_pair", "retrieval_is_strict_gt", "retrieval_dist_m",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk-jsonl", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-mode", choices=("topk", "strict"), default="topk")
    parser.add_argument(
        "--loftr-anchor-root", type=Path,
        help="optional aligned anchor root whose images/indexes are used by LoFTR",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")

    rows: list[dict[str, object]] = []
    query_count = 0
    for line in args.topk_jsonl.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        q_png = Path(item["q_cam_png"]).resolve()
        q_json = q_png.with_suffix(".json")
        q_meta = json.loads(q_json.read_text(encoding="utf-8"))
        q_pose = q_meta["pose_xyyaw"]
        q_xy = np.asarray([float(q_pose["x"]), float(q_pose["y"])])
        q_seg = str(q_meta["segment_id"])
        strict_db_index = int(item["strict_gt"]["db_index"])
        pair_id = str(item["pair_id"])
        if args.candidate_mode == "strict":
            strict = item["strict_gt"]
            candidates = [{
                "rank": int(strict["strict_rank"]),
                "db_index": int(strict["db_index"]),
                "score": float(strict["score"]),
                "anchor_png": strict["anchor_png"],
                "is_same_pair": True,
            }]
        else:
            candidates = item["topk"][: args.top_k]
        expected = 1 if args.candidate_mode == "strict" else args.top_k
        if len(candidates) != expected:
            raise ValueError(f"query {item['q_idx']} has only {len(candidates)} candidates")
        for candidate in candidates:
            retrieval_anchor_png = Path(candidate["anchor_png"]).resolve()
            if args.loftr_anchor_root is None:
                anchor_png = retrieval_anchor_png
            else:
                anchor_png = (args.loftr_anchor_root / retrieval_anchor_png.parent.name / "anchor.png").resolve()
            anchor_json = anchor_png.with_name("anchor.json")
            anchor_index = anchor_png.with_name("anchor_index.npz")
            a_meta = json.loads(anchor_json.read_text(encoding="utf-8"))
            a_pose = a_meta["anchor_pose_xyyaw"]
            distance = float(np.linalg.norm(q_xy - np.asarray([a_pose["x"], a_pose["y"]])))
            db_index = int(candidate["db_index"])
            rows.append({
                "pair_id": pair_id,
                "lane_tag": "same_scene",
                "db_seg": str(a_meta["segment_id"]),
                "q_seg": q_seg,
                "db_subset": str(a_meta.get("subset", "training")),
                "q_subset": "training",
                "q_frame_index": int(q_meta["frame_index"]),
                "anchor_id": int(a_meta["submap_id"]),
                "q_image_png": str(q_png),
                "q_meta_json": str(q_json),
                "q_depth_npz": "",
                "anchor_png": str(anchor_png),
                "anchor_json": str(anchor_json),
                "anchor_index_npz": str(anchor_index),
                "gt_npz": "",
                "dist_xy_m": distance,
                "is_strict_gt": int(db_index == strict_db_index),
                "is_gt_t1m": int(distance <= 1.0),
                "is_gt_t2m": int(distance <= 2.0),
                "is_gt_t5m": int(distance <= 5.0),
                "is_gt_t10m": int(distance <= 10.0),
                "n_matches": "", "n_fine": "",
                "w0": int(a_meta["width"]), "h0": int(a_meta["height"]),
                "w1": int(q_meta["width"]), "h1": int(q_meta["height"]),
                "error": "",
                "retrieval_q_idx": int(item["q_idx"]),
                "retrieval_rank": int(candidate["rank"]),
                "retrieval_score": float(candidate["score"]),
                "retrieval_db_index": db_index,
                "retrieval_is_same_pair": int(bool(candidate.get("is_same_pair", True))),
                "retrieval_is_strict_gt": int(db_index == strict_db_index),
                "retrieval_dist_m": distance,
            })
        query_count += 1

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "queries": query_count,
        "top_k": 1 if args.candidate_mode == "strict" else args.top_k,
        "candidate_mode": args.candidate_mode,
        "rows": len(rows),
        "output_csv": str(args.output_csv.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
