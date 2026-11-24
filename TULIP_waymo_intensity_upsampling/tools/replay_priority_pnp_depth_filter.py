"""Replay cached LoFTR matches with a local measured-depth consistency filter."""
import argparse
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from audit_loftr_generated_contribution import load_evaluator
from audit_priority_pnp_sources import solve
from evaluate_retrieval_topk_pipeline import key, read_csv, summarize, write_csv
from inspect_priority_depth_rejections import plane_distances, world_points


def supported_pixels(u, v, depth, ids, observed, radius, min_points, spread, tolerance,
                     xyz=None, plane_rms=.05, plane_delta=.10):
    """Require distinct measured point IDs near the pixel and a coherent depth band."""
    height, width = depth.shape
    keep = np.zeros(len(u), dtype=bool)
    restored = np.zeros(len(u), dtype=bool)
    for i, (x, y) in enumerate(zip(u, v)):
        region = (slice(max(0, y-radius), min(height, y+radius+1)),
                  slice(max(0, x-radius), min(width, x+radius+1)))
        valid = observed[region]
        support_ids, unique = np.unique(ids[region][valid], return_index=True)
        if len(unique) < min_points:
            continue
        reference = depth[region][valid][unique]
        if np.ptp(reference) <= spread:
            keep[i] = abs(float(depth[y, x]) - float(np.median(reference))) <= tolerance
        elif xyz is not None:
            rms, residual, _ = plane_distances(
                xyz[support_ids].astype(np.float64), xyz[ids[y, x]].astype(np.float64))
            restored[i] = rms != "" and rms <= plane_rms and residual <= plane_delta
    return keep | restored, restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--min-points", type=int, default=3)
    parser.add_argument("--max-spread-m", type=float, default=1.)
    parser.add_argument("--max-delta-m", type=float, default=.5)
    parser.add_argument("--restore-plane", action="store_true")
    parser.add_argument("--plane-rms-m", type=float, default=.05)
    parser.add_argument("--plane-delta-m", type=float, default=.10)
    parser.add_argument("--limit-queries", type=int, default=0, help="0 uses all saved queries")
    args = parser.parse_args()
    if args.radius < 1 or args.min_points < 1 or args.limit_queries < 0 or not all(
        math.isfinite(v) and v >= 0 for v in (
            args.max_spread_m, args.max_delta_m, args.plane_rms_m, args.plane_delta_m)
    ):
        parser.error("Invalid filter settings")
    source = args.run / "all507/all"
    rows = read_csv(source / "all507.csv")
    pairs = {key(row): row for row in read_csv(source / "all507_pnp/per_pair.csv")}
    if len(pairs) != len(rows) or pairs.keys() != {key(row) for row in rows}:
        raise ValueError("Cached pairs do not match the manifest")
    if args.limit_queries:
        ids = set(sorted({int(row["retrieval_q_idx"]) for row in rows})[:args.limit_queries])
        rows = [row for row in rows if int(row["retrieval_q_idx"]) in ids]
    protocol = json.loads((args.run / "protocol.json").read_text())
    if protocol["ransac_seed"] != 0:
        raise ValueError("Expected cached baseline with RANSAC seed 0")
    ev = load_evaluator()
    icp = ev.load_icp_db_to_q(protocol["icp"], db_col=None, q_col=None, mat_col=None, mat_cols=None)

    @lru_cache(maxsize=8)
    def load_depth(path):
        index = ev.load_anchor_index_npz(path)
        with np.load(path, allow_pickle=False) as data:
            point_ids = data["point_index"]
            generated = data["source_is_generated"].astype(bool)
        depth = index.depth_x
        if depth.shape != point_ids.shape or depth.shape != generated.shape:
            raise ValueError(f"Invalid depth or provenance shape: {path}")
        observed = (index.valid_mask & (point_ids >= 0) & ~generated
                    & np.isfinite(depth) & (depth >= .5) & (depth <= 200.))
        return index, point_ids, generated, observed

    @lru_cache(maxsize=128)
    def metadata(path):
        return json.loads(Path(path).read_text())

    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "all507_pnp"
    output.mkdir()
    (args.output / "filter_settings.json").write_text(
        json.dumps(vars(args), default=str, indent=2))
    records = []
    for number, row in enumerate(rows, 1):
        base = pairs[key(row)]
        cache = args.run / "matches" / Path(base["match_cache"]).name
        with np.load(cache, allow_pickle=False) as data:
            mk0, mk1 = data["mkpts0"], data["mkpts1"]
            valid, gen = data["valid_3d"], data["source_is_generated"]
        index, point_ids, generated, observed = load_depth(row["anchor_index_npz"])
        depth = index.depth_x
        height, width = depth.shape
        u = np.clip(np.round(mk0[:, 0]).astype(np.int64), 0, width-1)
        v = np.clip(np.round(mk0[:, 1]).astype(np.int64), 0, height-1)
        values = depth[v, u]
        valid_now = index.valid_mask[v, u] & np.isfinite(values) & (values >= .5) & (values <= 200.)
        if not np.array_equal(valid, valid_now) or not np.array_equal(gen, generated[v, u] & valid_now):
            raise ValueError(f"Cached labels differ from anchor provenance: {cache}")
        mask = valid & ~gen
        anchor = metadata(row["anchor_json"])
        xyz = (world_points(anchor["diagnostic_composition"]["point_index_map"])
               if args.restore_plane and gen.any() else None)
        mask[gen], restored = supported_pixels(
            u[gen], v[gen], depth, point_ids, observed, args.radius, args.min_points,
            args.max_spread_m, args.max_delta_m, xyz, args.plane_rms_m, args.plane_delta_m)
        restored_indices = np.flatnonzero(gen)[restored]
        transform = icp.get(row["db_seg"], row["q_seg"])
        pose, inliers, error = solve(ev, mk0, mk1, mask, valid, index,
                                    anchor, metadata(row["q_meta_json"]), transform)
        records.append(dict(base, pnp_success=bool(pose.success), comparable=transform is not None,
                            t_err_xy=error, n_match_in=int(pose.n_match_in), n_3d=int(pose.n_3d),
                            pred_Cx=float(pose.C_pred[0]) if pose.success else "",
                            pred_Cy=float(pose.C_pred[1]) if pose.success else "",
                            n_inlier=int(pose.n_inlier), reproj_med=float(pose.reproj_med),
                            generated_valid_used=int((gen & mask).sum()),
                            generated_inliers=int(gen[inliers].sum()),
                            restored_generated=int(restored.sum()),
                            restored_inliers=int(np.isin(inliers, restored_indices).sum())))
        if number == 1 or number % 100 == 0 or number == len(rows):
            print(f"{number}/{len(rows)}", flush=True)
    write_csv(args.output / "all507.csv", rows)
    write_csv(output / "per_pair.csv", records)
    summarize(args.output, "all507")
    print("Generated correspondences retained:",
          sum(int(row["generated_valid_used"]) for row in records),
          "/", sum(int(row["generated_valid_available"]) for row in records))
    print("Plane-restored correspondences:", sum(row["restored_generated"] for row in records))


if __name__ == "__main__":
    main()
