#!/usr/bin/env python3
"""Compute spatially tolerant Recall@K from an existing retrieval top-K dump."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk-jsonl", required=True, type=Path)
    parser.add_argument("--db-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--thresholds-m", default="1,2,5,10")
    parser.add_argument("--ks", default="1,5,10")
    args = parser.parse_args()
    thresholds = [float(v) for v in args.thresholds_m.split(",")]
    ks = [int(v) for v in args.ks.split(",")]
    with args.db_csv.open(newline="", encoding="utf-8") as handle:
        db = list(csv.DictReader(handle))
    db_xy = np.asarray([[float(r["anchor_x_db"]), float(r["anchor_y_db"])] for r in db])
    hits = {(t, k): [] for t in thresholds for k in ks}
    top1_distances: list[float] = []
    for line in args.topk_jsonl.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        cam_json = Path(row["q_cam_png"]).with_suffix(".json")
        cam_meta = json.loads(cam_json.read_text(encoding="utf-8"))
        pose = cam_meta["pose_xyyaw"]
        q_xy = np.asarray([float(pose["x"]), float(pose["y"])])
        ranked = np.asarray([int(item["db_index"]) for item in row["topk"]])
        distances = np.linalg.norm(db_xy[ranked] - q_xy[None, :], axis=1)
        top1_distances.append(float(distances[0]))
        for threshold in thresholds:
            for k in ks:
                hits[(threshold, k)].append(bool(np.any(distances[:k] <= threshold)))
    top1 = np.asarray(top1_distances)
    summary = {
        "queries": int(top1.size),
        "top1_distance_m": {
            "mean": float(top1.mean()), "median": float(np.median(top1)),
            "p90": float(np.percentile(top1, 90)), "max": float(top1.max()),
        },
        "spatial_recall": {
            f"within_{t:g}m_R@{k}": float(np.mean(hits[(t, k)]))
            for t in thresholds for k in ks
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
