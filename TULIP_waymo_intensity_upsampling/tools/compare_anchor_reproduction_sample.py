#!/usr/bin/env python3
"""Compare the common anchor IDs between a reproduced and reference DB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def anchor_images(root: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in root.glob("*/anchor.png"):
        try:
            result[int(path.parent.name)] = path
        except ValueError:
            continue
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reproduced-root", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    reproduced = anchor_images(args.reproduced_root)
    reference = anchor_images(args.reference_root)
    common = sorted(set(reproduced) & set(reference))
    if not common:
        raise ValueError("No common numeric anchor IDs")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for anchor_id in common:
        rep = cv2.imread(str(reproduced[anchor_id]), cv2.IMREAD_GRAYSCALE)
        ref = cv2.imread(str(reference[anchor_id]), cv2.IMREAD_GRAYSCALE)
        if rep is None or ref is None:
            raise ValueError(f"Failed to read anchor {anchor_id}")
        if rep.shape != ref.shape:
            raise ValueError(f"Shape mismatch for anchor {anchor_id}: {rep.shape} vs {ref.shape}")

        rep_mask = rep > 0
        ref_mask = ref > 0
        union = rep_mask | ref_mask
        intersection = rep_mask & ref_mask
        absdiff = cv2.absdiff(rep, ref)
        rows.append(
            {
                "anchor_id": anchor_id,
                "height": int(rep.shape[0]),
                "width": int(rep.shape[1]),
                "reproduced_nonzero": int(rep_mask.sum()),
                "reference_nonzero": int(ref_mask.sum()),
                "occupancy_iou": float(intersection.sum() / max(1, union.sum())),
                "exact_pixel_fraction": float(np.mean(rep == ref)),
                "mae_all": float(absdiff.mean()),
                "mae_union": float(absdiff[union].mean()) if union.any() else 0.0,
                "max_abs_error": int(absdiff.max()),
            }
        )
        montage = np.concatenate((ref, rep, absdiff), axis=1)
        cv2.imwrite(str(args.output_dir / f"anchor_{anchor_id:04d}_ref_rep_diff.png"), montage)

    summary = {
        "reproduced_root": str(args.reproduced_root.resolve()),
        "reference_root": str(args.reference_root.resolve()),
        "common_anchor_ids": common,
        "count": len(rows),
        "mean_occupancy_iou": float(np.mean([row["occupancy_iou"] for row in rows])),
        "mean_exact_pixel_fraction": float(np.mean([row["exact_pixel_fraction"] for row in rows])),
        "mean_mae_all": float(np.mean([row["mae_all"] for row in rows])),
        "mean_mae_union": float(np.mean([row["mae_union"] for row in rows])),
        "per_anchor": rows,
    }
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] wrote: {path}")


if __name__ == "__main__":
    main()
