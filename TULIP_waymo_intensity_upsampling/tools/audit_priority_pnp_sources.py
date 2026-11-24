"""Evaluate fixed LoFTR matches with and without TULIP-derived depth."""
import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from audit_loftr_generated_contribution import load_evaluator
from evaluate_retrieval_topk_pipeline import CKPT, ICP, key, read_csv, summarize, write_csv


def solve(ev, mk0, mk1, mask, valid, index, anchor, query, transform, seed=0):
    original = ev.cv2.solvePnPRansac
    captured = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result[3])
        return result

    ev.cv2.setRNGSeed(seed)
    ev.cv2.solvePnPRansac = capture
    try:
        pose = ev.estimate_pose_pnp_xforward(
            mk0[mask], mk1[mask], index, ev._parse_Twc(anchor),
            ev._parse_intrinsic(anchor), ev._parse_intrinsic(query), transform,
            min_depth=.5, max_depth=200., ransac_reproj_th=4.,
            ransac_max_iter=2000, ransac_conf=.999, refine=False,
        )
    finally:
        ev.cv2.solvePnPRansac = original
    if pose.n_3d != np.count_nonzero(mask & valid):
        raise ValueError("Depth validity differs from evaluator")
    inliers = (np.flatnonzero(mask & valid)[np.asarray(captured[0]).reshape(-1)]
               if pose.success else np.empty(0, dtype=np.int64))
    if len(inliers) != pose.n_inlier:
        raise ValueError("Unexpected RANSAC inlier count")
    error = ""
    if pose.success and transform is not None:
        truth = ev._camera_center_world(ev._parse_Twc(query))
        error = float(np.linalg.norm((pose.C_pred - truth)[:2]))
    return pose, inliers, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--changed", type=Path)
    parser.add_argument("--expected-queries", type=int, default=507)
    parser.add_argument("--all-only", action="store_true",
                        help="Evaluate all valid matches without provenance attribution")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("cases", "all507"), required=True)
    args = parser.parse_args()
    rows = read_csv(args.manifest)
    expected = args.expected_queries * 10
    if len(rows) != expected or len({key(row) for row in rows}) != expected or len(
        {row["retrieval_q_idx"] for row in rows}
    ) != args.expected_queries:
        raise ValueError("Expected ten unique candidate pairs per query")
    if args.scope == "cases":
        if args.changed is None:
            parser.error("--changed is required for cases")
        cases = [r for r in read_csv(args.changed) if r["selection"] == "changed_anchor"]
        wanted = {
            (r["q_seg"], int(r["q_frame_index"]), r[f"{side}_db_seg"], int(r[f"{side}_anchor_id"]))
            for r in cases for side in ("A", "priority")
        }
        if len(cases) != 33 or len(wanted) != 66:
            raise ValueError("Expected 33 changed-anchor cases and 66 pairs")
    else:
        wanted = {key(row) for row in rows}
    jobs = [(i, row) for i, row in enumerate(rows) if key(row) in wanted]
    if len(jobs) != len(wanted):
        raise ValueError("Requested candidates are missing from the manifest")
    cache = args.output / "matches"
    cache.mkdir(parents=True, exist_ok=True)
    protocol = args.output / "protocol.json"
    source = dict(manifest=str(args.manifest.resolve()),
                  manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  checkpoint=CKPT, icp=ICP,
                  precision="fp16 CUDA", ransac_seed=0)
    if args.all_only:
        source["all_only"] = True
    if protocol.exists():
        if json.loads(protocol.read_text()) != source:
            raise ValueError("Output cache belongs to a different protocol")
    else:
        protocol.write_text(json.dumps(source, indent=2))
    export = args.output / args.scope
    if export.exists():
        raise FileExistsError(export)
    ev = load_evaluator()
    device = torch.device("cuda")
    matcher = None
    icp = ev.load_icp_db_to_q(ICP, db_col=None, q_col=None, mat_col=None, mat_cols=None)

    @lru_cache(maxsize=8)
    def load_index(path):
        index = ev.load_anchor_index_npz(path)
        if args.all_only:
            return index, None
        with np.load(path, allow_pickle=False) as data:
            generated = data["source_is_generated"].astype(bool)
        if generated.shape != index.depth_x.shape:
            raise ValueError(f"Provenance dimensions differ: {path}")
        return index, generated

    results = {"all": []}
    if not args.all_only:
        results["measured_only"] = []
    for number, (i, row) in enumerate(jobs, 1):
        path = cache / f"{i:05d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                records = json.loads(data["records"].item())
        else:
            if matcher is None:
                matcher = ev.load_matcher(CKPT, device=device, enable_fine=True,
                                          enable_repeatability=False)
            anchor = json.loads(Path(row["anchor_json"]).read_text())
            query = json.loads(Path(row["q_meta_json"]).read_text())
            index, generated = load_index(row["anchor_index_npz"])
            transform = icp.get(row["db_seg"], row["q_seg"])
            matches = ev.infer_matches(
                matcher, ev._read_gray_u8(row["anchor_png"]), ev._read_gray_u8(row["q_image_png"]),
                resize_long=840, stride=8, cell_point="center", device=device,
                amp_enabled=True, amp_dtype=torch.float16, match_source="auto",
                conf_th=0., max_matches=4096,
            )
            mk0, mk1 = matches["mkpts0"], matches["mkpts1"]
            h, w = index.depth_x.shape
            u = np.clip(np.round(mk0[:, 0]).astype(np.int64), 0, w - 1)
            v = np.clip(np.round(mk0[:, 1]).astype(np.int64), 0, h - 1)
            depth = index.depth_x[v, u]
            valid = index.valid_mask[v, u] & np.isfinite(depth) & (depth >= .5) & (depth <= 200.)
            gen = generated[v, u] & valid if generated is not None else None
            saved = dict(mkpts0=mk0, mkpts1=mk1, valid_3d=valid)
            if gen is not None:
                saved["source_is_generated"] = gen
            records = {}
            for name in results:
                mask = np.ones(len(mk0), dtype=bool) if name == "all" else ~gen
                pose, inliers, error = solve(
                    ev, mk0, mk1, mask, valid, index, anchor, query, transform)
                saved[name + "_inlier_indices"] = inliers
                records[name] = dict(
                    {field: row[field] for field in (
                        "pair_id", "q_seg", "q_frame_index", "db_seg", "anchor_id", "retrieval_rank")},
                    match_cache=str(path), pnp_success=bool(pose.success),
                    comparable=transform is not None, t_err_xy=error,
                    pred_Cx=float(pose.C_pred[0]) if pose.success else "",
                    pred_Cy=float(pose.C_pred[1]) if pose.success else "",
                    n_match_in=int(pose.n_match_in), n_3d=int(pose.n_3d),
                    n_inlier=int(pose.n_inlier), reproj_med=float(pose.reproj_med),
                    generated_valid_available=int(gen.sum()) if gen is not None else "",
                    generated_valid_used=int((gen & mask).sum()) if gen is not None else "",
                    generated_inliers=int(gen[inliers].sum()) if gen is not None else "",
                )
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **saved, records=json.dumps(records))
            temporary.replace(path)
        for name in results:
            results[name].append(records[name])
        if number == 1 or number % 50 == 0 or number == len(jobs):
            print(f"{number}/{len(jobs)}", flush=True)

    export.mkdir()
    for name, records in results.items():
        root = export / name
        root.mkdir()
        output = root / "all507_pnp" if args.scope == "all507" else root
        output.mkdir(exist_ok=True)
        write_csv(output / "per_pair.csv", records)
        if args.scope == "all507":
            write_csv(root / "all507.csv", rows)
            print(name, flush=True)
            summarize(root, "all507")


if __name__ == "__main__":
    main()
