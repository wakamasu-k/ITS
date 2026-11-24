#!/usr/bin/env python3
"""Compare B/GT anchor grayscale values without counting black background."""
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
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def corr(a: np.ndarray, b: np.ndarray) -> float | None:
    return None if a.size < 2 or np.std(a) == 0 or np.std(b) == 0 else float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    args = parse_args()
    ids = sorted(int(p.name) for p in args.b_root.iterdir() if p.is_dir() and p.name.isdigit())
    if ids != sorted(int(p.name) for p in args.c_root.iterdir() if p.is_dir() and p.name.isdigit()):
        raise ValueError("B and C anchor IDs differ")
    rows: list[dict[str, object]] = []
    all_b: list[np.ndarray] = []
    all_c: list[np.ndarray] = []
    for anchor_id in ids:
        b_idx = np.load(args.b_root / str(anchor_id) / "anchor_index.npz")["point_index"]
        c_idx = np.load(args.c_root / str(anchor_id) / "anchor_index.npz")["point_index"]
        overlap = (b_idx >= 0) & (c_idx >= 0)
        b = np.asarray(Image.open(args.b_root / str(anchor_id) / "anchor.png").convert("L"), dtype=np.float64)[overlap] / 255.0
        c = np.asarray(Image.open(args.c_root / str(anchor_id) / "anchor.png").convert("L"), dtype=np.float64)[overlap] / 255.0
        all_b.append(b)
        all_c.append(c)
        rows.append({
            "anchor_id": anchor_id,
            "overlap_pixels": int(overlap.sum()),
            "intensity_mae": float(np.mean(np.abs(b - c))) if b.size else None,
            "intensity_rmse": float(np.sqrt(np.mean((b - c) ** 2))) if b.size else None,
            "intensity_correlation": corr(b, c),
            "b_mean": float(b.mean()) if b.size else None,
            "c_mean": float(c.mean()) if c.size else None,
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_anchor.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    b, c = np.concatenate(all_b), np.concatenate(all_c)
    summary = {
        "anchors": len(ids),
        "comparison_mask": "exact B/GT covered-pixel overlap only",
        "overlap_pixels": int(b.size),
        "intensity_mae": float(np.mean(np.abs(b - c))),
        "intensity_rmse": float(np.sqrt(np.mean((b - c) ** 2))),
        "intensity_correlation": corr(b, c),
        "b_mean": float(b.mean()),
        "c_mean": float(c.mean()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
