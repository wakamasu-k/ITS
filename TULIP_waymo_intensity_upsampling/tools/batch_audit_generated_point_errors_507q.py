#!/usr/bin/env python3
"""Audit B-generated newly covered pixels for every strict-GT retrieval query.

Groups work by segment so each large B map is loaded once. The script is
read-only with respect to maps and retrieval inputs; it creates a new report
directory and refuses to overwrite one that already exists.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    rx, ry = rankdata(np.asarray(x)), rankdata(np.asarray(y))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {
        "queries": int(a.size),
        "median": float(np.median(a)) if a.size else None,
        "mean": float(a.mean()) if a.size else None,
    }


def anchor_from_path(value: str) -> tuple[str, str]:
    parts = Path(value).parts
    for i in range(len(parts) - 3):
        if parts[i:i + 2] == ("anchors_B", "training"):
            return parts[i + 2], parts[i + 3]
    raise ValueError(f"Cannot parse B anchor path: {value}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--retrieval-jsonl",
        type=Path,
        default=Path("outputs/shimizu_protocol_507q_754db_abc/retrieval_B/shimizu_507q_754db_B_topk_top50.jsonl"),
    )
    p.add_argument(
        "--reference-root",
        type=Path,
        default=Path("/mnt/w/32line/datasets/submaps/cross/original_val_test_all/db/training"),
    )
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {args.output_dir}")

    project = Path(__file__).resolve().parents[1]
    abc = project / "outputs/shimizu_protocol_507q_754db_abc"
    records = [json.loads(line) for line in args.retrieval_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("No retrieval records found")

    keyed = []
    for rec in records:
        strict = rec.get("strict_gt") or {}
        if not strict.get("anchor_png"):
            raise ValueError(f"Query {rec.get('q_idx')} has no strict-GT anchor path")
        seg, anchor = anchor_from_path(strict["anchor_png"])
        keyed.append((seg, anchor, rec))

    # Cache each query anchor's metrics; retrieval files can contain repeated GT anchors.
    per_anchor: dict[tuple[str, str], dict] = {}
    by_segment: dict[str, list[tuple[str, str, dict]]] = {}
    for item in keyed:
        by_segment.setdefault(item[0], []).append(item)

    range_edges = [0, 10, 20, 40, 80, 120, float("inf")]
    pooled_ranges = {f"{lo:g}_{hi:g}m" if np.isfinite(hi) else f"{lo:g}m_plus": []
                     for lo, hi in zip(range_edges[:-1], range_edges[1:])}

    for seg, seg_items in by_segment.items():
        map_path = abc / "maps/B_tulip32/training" / seg / "map_static.npz"
        with np.load(map_path, allow_pickle=False) as z:
            b_xyz = z["xyz"]
            generated = z["source_is_generated"].astype(bool)
        for _, anchor, _ in seg_items:
            key = (seg, anchor)
            if key in per_anchor:
                continue
            a_dir = abc / "anchors_A/training" / seg / anchor
            b_dir = abc / "anchors_B/training" / seg / anchor
            ref_dir = args.reference_root / seg / anchor
            with np.load(a_dir / "anchor_index.npz", allow_pickle=False) as z:
                a_idx = z["point_index"]
            with np.load(b_dir / "anchor_index.npz", allow_pickle=False) as z:
                b_idx = z["point_index"]
            with np.load(ref_dir / "anchor_index.npz", allow_pickle=False) as z:
                ref_idx, ref_depth = z["point_index"], z["depth"]
            b_meta = json.loads((b_dir / "anchor.json").read_text(encoding="utf-8"))
            ref_meta = json.loads((ref_dir / "anchor.json").read_text(encoding="utf-8"))
            if not np.allclose(b_meta["intrinsic"], ref_meta["intrinsic"], rtol=0, atol=1e-5):
                raise ValueError(f"Camera intrinsics mismatch at {seg}/{anchor}")
            if not np.allclose(b_meta["world_to_cam"], ref_meta["world_to_cam"], rtol=0, atol=1e-5):
                raise ValueError(f"Camera pose mismatch at {seg}/{anchor}")
            b_valid = b_idx >= 0
            gen_pix = np.zeros_like(b_valid)
            gen_pix[b_valid] = generated[b_idx[b_valid]]
            newly = gen_pix & (a_idx < 0)
            ref_valid = (ref_idx >= 0) & np.isfinite(ref_depth) & (ref_depth > 0)
            match = newly & ref_valid
            y, x = np.nonzero(match)
            d = ref_depth[y, x].astype(np.float64)
            fx, fy, cx, cy = map(float, ref_meta["intrinsic"])
            cam = np.stack([d, -(x-cx)*d/fx, -(y-cy)*d/fy, np.ones_like(d)], axis=1)
            world = (cam @ np.linalg.inv(np.asarray(ref_meta["world_to_cam"], dtype=np.float64)).T)[:, :3]
            pred = b_xyz[b_idx[y, x].astype(np.int64)].astype(np.float64)
            error = np.linalg.norm(pred - world, axis=1)
            per_anchor[key] = {
                "new_generated_pixels": int(newly.sum()),
                "reference_matched_pixels": int(match.sum()),
                "reference_unmatched_new_pixels": int((newly & ~ref_valid).sum()),
                "median_error_m": float(np.median(error)) if error.size else None,
                "p90_error_m": float(np.percentile(error, 90)) if error.size else None,
                "over_2m_fraction": float(np.mean(error > 2.0)) if error.size else None,
            }
            for lo, hi in zip(range_edges[:-1], range_edges[1:]):
                label = f"{lo:g}_{hi:g}m" if np.isfinite(hi) else f"{lo:g}m_plus"
                m = (d >= lo) & (d < hi)
                if m.any():
                    pooled_ranges[label].append(error[m].copy())

    query_rows = []
    for seg, anchor, rec in keyed:
        metric = per_anchor[(seg, anchor)]
        strict = rec["strict_gt"]
        top1 = rec.get("topk", [{}])[0]
        query_rows.append({
            "query_index": rec.get("q_idx"),
            "segment": seg,
            "anchor": anchor,
            "strict_gt_rank": strict.get("strict_rank"),
            "top1_score": top1.get("score"),
            "new_generated_pixels": metric["new_generated_pixels"],
            "reference_matched_pixels": metric["reference_matched_pixels"],
            "reference_unmatched_new_pixels": metric["reference_unmatched_new_pixels"],
            "median_error_m": metric["median_error_m"],
            "p90_error_m": metric["p90_error_m"],
            "over_2m_fraction": metric["over_2m_fraction"],
        })

    def corr_for(field: str) -> float | None:
        valid = [r for r in query_rows if r[field] is not None]
        return spearman([float(r["strict_gt_rank"]) for r in valid], [float(r[field]) for r in valid])

    rank_bands = {}
    for label, keep in {
        "rank_1": lambda r: r["strict_gt_rank"] == 1,
        "rank_2_to_5": lambda r: 2 <= r["strict_gt_rank"] <= 5,
        "rank_6_to_20": lambda r: 6 <= r["strict_gt_rank"] <= 20,
        "rank_21_plus": lambda r: r["strict_gt_rank"] >= 21,
    }.items():
        group = [r for r in query_rows if keep(r) and r["over_2m_fraction"] is not None]
        rank_bands[label] = {
            "queries": len(group),
            "median_query_median_error_m": float(np.median([r["median_error_m"] for r in group])) if group else None,
            "median_query_over_2m_fraction": float(np.median([r["over_2m_fraction"] for r in group])) if group else None,
        }

    pooled = {}
    for key, chunks in pooled_ranges.items():
        a = np.concatenate(chunks) if chunks else np.asarray([], dtype=np.float64)
        pooled[key] = {
            "matched_pixels": int(a.size),
            "median_error_m": float(np.median(a)) if a.size else None,
            "p90_error_m": float(np.percentile(a, 90)) if a.size else None,
            "over_2m_fraction": float(np.mean(a > 2.0)) if a.size else None,
        }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "per_query.jsonl").open("w", encoding="utf-8") as f:
        for row in query_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "query_records": len(query_rows),
        "unique_gt_anchors": len(per_anchor),
        "segments": len(by_segment),
        "spearman_strict_rank_vs_over_2m_fraction": corr_for("over_2m_fraction"),
        "spearman_strict_rank_vs_median_xyz_error_m": corr_for("median_error_m"),
        "retrieval_rank_groups": rank_bands,
        "pooled_error_by_reference_depth": pooled,
        "limits": [
            "Same-pixel Shimizu reference depth is used; unmatched new pixels are not scored.",
            "Correlation is descriptive and does not establish that geometric error causes retrieval rank.",
            "Repeated queries sharing a GT anchor reuse the same geometric audit result.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
