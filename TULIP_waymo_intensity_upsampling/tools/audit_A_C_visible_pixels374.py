"""Compare rendered A/C pixels with saved depth and verified world point coordinates."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from run_tulip_pixel_ablation374 import BASE, VALID


def read_raster(image):
    image = Path(image)
    meta = json.loads(image.with_name("anchor.json").read_text())
    with np.load(image.with_name("anchor_index.npz")) as data:
        index, depth = data["point_index"].copy(), data["depth"].copy()
    return meta, index, depth


def verify_map(xyz, index, depth, transform):
    valid = index >= 0
    ids = index[valid]
    if not len(ids) or ids.min() < 0 or ids.max() >= len(xyz):
        raise ValueError("Point indices exceed map bounds")
    # Forward camera depth is the first coordinate in this renderer.
    projected = xyz[ids] @ transform[0, :3] + transform[0, 3]
    if not np.allclose(projected, depth[valid], atol=0.02, rtol=1e-5):
        raise ValueError("Map does not reproduce saved pixel depths")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3,
                        help="Anchors per segment; 0 processes all 443 anchors")
    parser.add_argument("--same-point-mm", type=float, default=1.)
    args = parser.parse_args()
    if args.limit < 0 or not np.isfinite(args.same_point_mm) or args.same_point_mm <= 0:
        parser.error("Invalid limit or coordinate tolerance")
    a = pd.read_parquet(BASE / "A/db.parquet")
    c = pd.read_parquet(BASE / "C/db.parquet")
    fields = ["pair_id", "db_subset", "db_seg", "anchor_id"]
    if not a[fields].astype(str).equals(c[fields].astype(str)):
        raise ValueError("A/C anchor identities differ")
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for (subset, segment), group in a.groupby(["db_subset", "db_seg"], sort=False):
        map_a = VALID / "maps/A_observed16" / subset / segment / "map_static.npz"
        split = "original_val" if segment.startswith("segment-832302") else "original_test"
        map_c = Path("/mnt/w/32line/datasets/maps/cross") / split / "db" / subset / segment / "map_static.npz"
        with np.load(map_a) as data:
            xyz_a = data["xyz"]
        with np.load(map_c) as data:
            xyz_c = data["xyz"]
        if args.limit:
            positions = np.unique(np.linspace(0, len(group) - 1, min(args.limit, len(group))).astype(int))
            group = group.iloc[positions]
        for idx, row in group.iterrows():
            am, ai, ad = read_raster(row.anchor_png)
            cm, ci, cd = read_raster(c.iloc[idx].anchor_png)
            for field in ("width", "height", "intrinsic", "world_to_cam"):
                if not np.allclose(am[field], cm[field], atol=1e-8, rtol=0):
                    raise ValueError(f"Anchor geometry differs: {idx}/{field}")
            transform = np.array(am["world_to_cam"])
            verify_map(xyz_a, ai, ad, transform)
            verify_map(xyz_c, ci, cd, transform)
            av, cv = ai >= 0, ci >= 0
            common = av & cv
            same = np.zeros(ai.shape, bool)
            distance = np.linalg.norm(xyz_a[ai[common]] - xyz_c[ci[common]], axis=1)
            same[common] = distance <= args.same_point_mm / 1000
            changed = common & ~same
            delta = np.full(ad.shape, np.nan, np.float32)
            np.subtract(cd, ad, out=delta, where=common)
            # 0 empty, 1 same coordinates, 2 C-only, 3 changed point, 4 A-only.
            labels = np.zeros(ai.shape, np.uint8)
            labels[same], labels[~av & cv], labels[changed], labels[av & ~cv] = 1, 2, 3, 4
            prefix = args.output / f"db_{idx:04d}"
            np.savez_compressed(str(prefix) + "_pixels.npz", category=labels,
                                A_point_index=ai, C_point_index=ci,
                                A_depth=ad, C_depth=cd, C_minus_A_depth=delta)
            colors = np.array([[0, 0, 0], [130, 130, 130], [0, 220, 0],
                               [0, 0, 255], [255, 100, 0]], np.uint8)
            left = cv2.imread(row.anchor_png)
            right = cv2.imread(c.iloc[idx].anchor_png)
            if left is None or right is None:
                raise ValueError("Missing rendered PNG")
            dest = str(prefix) + "_comparison.png"
            if not cv2.imwrite(dest, np.hstack([left, right, colors[labels]])):
                raise OSError(dest)
            records.append(dict(db_index=idx, segment=segment, anchor_id=row.anchor_id,
                                A_valid=int(av.sum()), C_valid=int(cv.sum()),
                                C_new=int((~av & cv).sum()), A_only=int((av & ~cv).sum()),
                                same_point=int(same.sum()), changed_point=int(changed.sum()),
                                changed_depth_abs_median=float(np.median(np.abs(delta[changed]))) if changed.any() else None,
                                map_A=str(map_a), map_C=str(map_c), image=dest))
            print(f"Checked DB {idx}, anchor {row.anchor_id}", flush=True)
    pd.DataFrame(records).to_csv(args.output / "summary.csv", index=False)
    (args.output / "settings.json").write_text(json.dumps(
        {"same_point_tolerance_mm": args.same_point_mm, "limit_per_segment": args.limit,
         "category": {"0": "empty", "1": "same_world_coordinate", "2": "C_only",
                      "3": "different_world_coordinate", "4": "A_only"}}, indent=2))
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
