"""Locate rejected generated matches and inspect local measured surfaces."""
import argparse
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from evaluate_retrieval_topk_pipeline import key, read_csv, write_csv


@lru_cache(maxsize=2)
def world_points(path):
    with np.load(path, allow_pickle=False) as data:
        return data["xyz"]


def plane_distances(points, generated_point):
    if len(points) < 6:
        return "", "", ""
    center = points.mean(axis=0)
    _, singular, axes = np.linalg.svd(points - center, full_matrices=False)
    if singular[1] / np.sqrt(len(points)) < .025:
        return "", "", ""
    normal = axes[-1]
    rms = float(singular[-1] / np.sqrt(len(points)))
    residual = float(abs((generated_point - center) @ normal))
    return rms, residual, float(abs(normal[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "filtered", "changed", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads((args.filtered / "filter_settings.json").read_text())
    source = args.run / "all507/all"
    manifest = {key(r): r for r in read_csv(source / "all507.csv")}
    baseline = {key(r): r for r in read_csv(source / "all507_pnp/per_pair.csv")}
    filtered = {key(r): r for r in read_csv(args.filtered / "all507_pnp/per_pair.csv")}
    cases = [r for r in read_csv(args.changed) if r["selection"] == "same_anchor"]
    if not cases or manifest.keys() != baseline.keys() or baseline.keys() != filtered.keys():
        raise ValueError("Candidate results do not match")
    args.output.mkdir(parents=True)
    points, summaries = [], []
    for case in cases:
        k = (case["q_seg"], int(case["q_frame_index"]), case["old_db_seg"], int(case["old_anchor_id"]))
        row, before, after = manifest[k], baseline[k], filtered[k]
        path = args.run / "matches" / Path(before["match_cache"]).name
        with np.load(path, allow_pickle=False) as data:
            xy, generated = data["mkpts0"], data["source_is_generated"]
            inlier = np.zeros(len(xy), dtype=bool)
            inlier[data["all_inlier_indices"]] = True
        with np.load(row["anchor_index_npz"], allow_pickle=False) as data:
            depth, ids, provenance = data["depth"], data["point_index"], data["source_is_generated"]
        meta = json.loads(Path(row["anchor_json"]).read_text())
        xyz = world_points(meta["diagnostic_composition"]["point_index_map"])
        observed = (ids >= 0) & ~provenance & np.isfinite(depth) & (depth >= .5) & (depth <= 200.)
        image = cv2.imread(row["anchor_png"], cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != depth.shape:
            raise ValueError(f"Invalid anchor image: {row['anchor_png']}")
        h, w = depth.shape
        radius = settings["radius"]
        counts = Counter()
        added = []
        for i in np.flatnonzero(generated):
            x, y = np.clip(np.round(xy[i]).astype(int), (0, 0), (w-1, h-1))
            region = (slice(max(0, y-radius), min(h, y+radius+1)),
                      slice(max(0, x-radius), min(w, x+radius+1)))
            valid = observed[region]
            support_ids, unique = np.unique(ids[region][valid], return_index=True)
            depths = depth[region][valid][unique]
            spread = float(np.ptp(depths)) if len(depths) else ""
            delta = abs(float(depth[y, x]) - float(np.median(depths))) if len(depths) else ""
            if len(depths) < settings["min_points"]:
                reason = "too_few_points"
            elif spread > settings["max_spread_m"]:
                reason = "depth_spread"
            elif delta > settings["max_delta_m"]:
                reason = "depth_delta"
            else:
                reason = "retained"
            counts[reason] += 1
            if inlier[i]:
                counts[reason + "_inlier"] += 1
            rms, residual, vertical = plane_distances(
                xyz[support_ids].astype(np.float64), xyz[ids[y, x]].astype(np.float64))
            plane_supported = (rms != "" and rms <= .05 and residual <= .10)
            if plane_supported and inlier[i] and reason != "retained":
                counts["plane_supported_rejected_inliers"] += 1
            points.append(dict(query_index=case["query_index"], change=case["change"],
                               match_index=int(i), x=int(x), y=int(y), full_inlier=bool(inlier[i]),
                               reason=reason, support_points=len(depths), depth_spread_m=spread,
                               depth_delta_m=delta, plane_rms_m=rms, plane_distance_m=residual,
                               normal_abs_z=vertical, plane_supported=plane_supported))
            if inlier[i]:
                added.append((int(x), int(y), reason))
        if counts["retained"] != int(after["generated_valid_used"]):
            raise ValueError(f"Original filter did not reproduce for {k}")
        for x, y, reason in sorted(added, key=lambda p: p[2] != "retained"):
            color = {"retained": (0, 220, 0), "depth_spread": (0, 0, 255),
                     "depth_delta": (0, 170, 255), "too_few_points": (255, 180, 0)}[reason]
            cv2.circle(image, (x, y), 5, color, 2)
        filename = args.output / f'{case["change"]}_q{int(case["query_index"]):03d}.png'
        if not cv2.imwrite(str(filename), image):
            raise OSError(filename)
        names = ("retained", "too_few_points", "depth_spread", "depth_delta")
        summaries.append(dict(
            query_index=case["query_index"], change=case["change"],
            all_xy_m=before["t_err_xy"], filtered_xy_m=after["t_err_xy"],
            **{name: counts[name] for name in names},
            **{name + "_inlier": counts[name + "_inlier"] for name in names},
            plane_supported_rejected_inliers=counts["plane_supported_rejected_inliers"],
            image=str(filename)))
    write_csv(args.output / "matches.csv", points)
    write_csv(args.output / "queries.csv", summaries)
    for group in ("improved", "worsened"):
        subset = [r for r in summaries if r["change"] == group]
        print(group, len(subset), {name: sum(r[name] for r in subset) for name in (
            "retained_inlier", "too_few_points_inlier", "depth_spread_inlier", "depth_delta_inlier",
            "plane_supported_rejected_inliers")})
    print(args.output)


if __name__ == "__main__":
    main()
