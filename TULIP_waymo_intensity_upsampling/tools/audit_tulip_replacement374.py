"""Inspect the same near anchor before and after enabling pixel replacement."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from run_tulip_pixel_ablation374 import BASE, B_MAPS, VALID


def rankings(path):
    return {r["q_cam_png"]: r for r in map(json.loads, path.read_text().splitlines())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-retained", action="store_true")
    args = parser.parse_args()
    before = rankings(VALID / "retrieval/observed_priority_topk_top50.jsonl")
    after = rankings(args.run / "replacement_and_fill/replacement_and_fill_topk_top50.jsonl")
    if before.keys() != after.keys() or len(before) != 374:
        raise ValueError("Expected identical sets of 374 queries")
    cases = []
    for camera, old in before.items():
        new = after[camera]
        if (old["pair_id"], old["strict_gt"]["db_index"]) != (
                new["pair_id"], new["strict_gt"]["db_index"]):
            raise ValueError(f"Ground truth differs for {camera}")
        near = [c for c in old["topk"][:10] if c["soft_gt"]["t2m"]]
        retained = any(c["soft_gt"]["t2m"] for c in new["topk"][:10])
        if not near or (retained and not args.compare_retained):
            continue
        candidate = near[0]
        idx = candidate["db_index"]
        match = next((c for c in new["topk"] if c["db_index"] == idx), {})
        cases.append(dict(outcome="retained" if retained else "lost", query=old["q_idx"], pair_id=old["pair_id"], db_index=idx,
                          distance_m=candidate["dist_m"], rank_before=candidate["rank"],
                          rank_after=match.get("rank"), score_before=candidate["score"],
                          score_after=match.get("score"), camera=camera))
    lost = sum(row["outcome"] == "lost" for row in cases)
    if lost != 48 or (args.compare_retained and len(cases) != 322):
        raise ValueError(f"Unexpected group counts: {lost} lost, {len(cases) - lost} retained")
    a = pd.read_parquet(BASE / "A/db.parquet")
    b = pd.read_parquet(BASE / "B/db.parquet")
    p = pd.read_parquet(VALID / "database/db.parquet")
    selected = a.loc[sorted({row["db_index"] for row in cases})]
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for (subset, segment), group in selected.groupby(["db_subset", "db_seg"]):
        with np.load(B_MAPS / subset / segment / "map_static.npz") as data:
            generated = data["source_is_generated"].reshape(-1).astype(bool)
        for idx, row in group.iterrows():
            ap, bp = Path(row.anchor_png), Path(b.iloc[idx].anchor_png)
            with np.load(ap.with_name("anchor_index.npz")) as az, np.load(
                    bp.with_name("anchor_index.npz")) as bz:
                ai, bi = az["point_index"], bz["point_index"]
                ad, bd = az["depth"], bz["depth"]
            replaced = (ai >= 0) & (bi >= 0) & generated[np.maximum(bi, 0)]
            valid = replaced & np.isfinite(ad) & np.isfinite(bd) & (ad > 0) & (bd > 0)
            delta = np.zeros(ad.shape, dtype=np.float32)
            np.subtract(ad, bd, out=delta, where=valid)
            np.abs(delta, out=delta)
            near = valid & (delta <= 0.25)
            mid = valid & (delta > 0.25) & (delta <= 2)
            far = valid & (delta > 2)
            old_image = cv2.imread(p.iloc[idx].anchor_png, cv2.IMREAD_GRAYSCALE)
            new_image = cv2.imread(str(args.run / "replacement_and_fill/images" /
                                       f"{idx:04d}.png"), cv2.IMREAD_GRAYSCALE)
            if old_image is None or new_image is None or old_image.shape != ai.shape or new_image.shape != ai.shape:
                raise ValueError(f"Missing or incompatible images for DB index {idx}")
            difference = np.abs(new_image.astype(np.int16) - old_image)
            target = args.output / f"db_{idx:04d}.png"
            if not args.compare_retained:
                overlay = cv2.cvtColor(old_image, cv2.COLOR_GRAY2BGR)
                overlay[replaced & ~valid] = (255, 0, 255)
                overlay[near] = (0, 200, 0)
                overlay[mid] = (0, 200, 255)
                overlay[far] = (0, 0, 255)
                panels = [cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) for im in (old_image, new_image)]
                if not cv2.imwrite(str(target), np.hstack([*panels, overlay])):
                    raise OSError(target)
            results[idx] = dict(anchor_id=row.anchor_id, replaced=int(replaced.sum()),
                                valid_depth=int(valid.sum()), near=int(near.sum()),
                                intermediate=int(mid.sum()), far=int(far.sum()),
                                near_png_mae=float(difference[near].mean()) if near.any() else None,
                                far_png_mae=float(difference[far].mean()) if far.any() else None,
                                image=None if args.compare_retained else str(target))
    table = pd.DataFrame([{**case, **results[case["db_index"]]} for case in cases])
    table["rank_after_display"] = table["rank_after"].map(
        lambda rank: ">50" if pd.isna(rank) else str(int(rank)))
    if args.compare_retained:
        table["far_pct"] = 100 * table["far"] / table["valid_depth"].replace(0, np.nan)
        table.to_csv(args.output / "query_comparison.csv", index=False)
        summary = table.groupby(["pair_id", "outcome"]).agg(
            queries=("query", "size"),
            unique_anchors=("db_index", "nunique"),
            median_far_pct=("far_pct", "median"),
            median_near_png_mae=("near_png_mae", "median"),
        )
        summary.to_csv(args.output / "by_segment.csv")
        print(summary.to_string())
        print(args.output)
        return
    table.to_csv(args.output / "worsened_queries.csv", index=False)
    anchors = table.drop_duplicates("db_index")
    anchors.to_csv(args.output / "unique_anchors.csv", index=False)
    print(f"Queries: {len(table)}; unique anchors: {len(anchors)}")
    print(anchors[["replaced", "valid_depth", "near", "intermediate", "far"]].sum().to_string())
    print(args.output)


if __name__ == "__main__":
    main()
