"""Measure B recovery at the same pixels where C differs from A."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from audit_A_C_visible_pixels374 import read_raster, verify_map
from run_tulip_pixel_ablation374 import BASE, B_MAPS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = pd.read_csv(args.source / "summary.csv")
    b = pd.read_parquet(BASE / "B/db.parquet")
    c = pd.read_parquet(BASE / "C/db.parquet")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for segment, group in cases.groupby("segment", sort=False):
        selected = b[b.db_seg == segment]
        if len(selected.db_subset.unique()) != 1:
            raise ValueError("Ambiguous subset")
        subset = selected.db_subset.iloc[0]
        with np.load(B_MAPS / subset / segment / "map_static.npz") as data:
            xyz = data["xyz"]
            intensity_b = data["intensity"].reshape(-1)
            generated = data["source_is_generated"].reshape(-1).astype(bool)
        c_maps = group.map_C.unique()
        if len(c_maps) != 1:
            raise ValueError("Ambiguous C map")
        with np.load(c_maps[0]) as data:
            intensity_c = data["intensity"].reshape(-1)
        for case in group.itertuples():
            idx = case.db_index
            if str(b.iloc[idx].anchor_id) != str(case.anchor_id) or b.iloc[idx].db_seg != segment:
                raise ValueError("Anchor identity mismatch")
            bm, bi, bd = read_raster(b.iloc[idx].anchor_png)
            cm, ci, cd = read_raster(c.iloc[idx].anchor_png)
            for key in ("width", "height", "intrinsic", "world_to_cam"):
                if not np.allclose(bm[key], cm[key], atol=1e-8, rtol=0):
                    raise ValueError(f"Geometry mismatch: {key}")
            verify_map(xyz, bi, bd, np.array(bm["world_to_cam"]))
            with np.load(args.source / f"db_{idx:04d}_pixels.npz") as data:
                category = data["category"].copy()
                if not np.array_equal(data["C_point_index"], ci):
                    raise ValueError("C raster differs from source audit")
            bv = (bi >= 0) & np.isfinite(bd) & (bd > 0)
            cv = (ci >= 0) & np.isfinite(cd) & (cd > 0)
            both = bv & cv
            difference = np.zeros(bd.shape, np.float32)
            np.subtract(bd, cd, out=difference, where=both)
            difference = np.abs(difference)
            png_b = cv2.imread(b.iloc[idx].anchor_png, 0)
            png_c = cv2.imread(c.iloc[idx].anchor_png, 0)
            if png_b is None or png_c is None or png_b.shape != bd.shape or png_c.shape != bd.shape:
                raise ValueError("Missing or incompatible PNG")
            png_difference = np.abs(png_b.astype(np.int16) - png_c)
            for region_name, region in (("C_new", category == 2), ("C_changed", category == 3)):
                usable = region & both
                near = usable & (difference <= .25)
                far = usable & (difference > 2)
                gen = np.zeros(bd.shape, bool)
                gen[usable] = generated[bi[usable]]
                raw_error = np.abs(intensity_b[bi[near]] - intensity_c[ci[near]])
                rows.append(dict(db_index=idx, segment=segment, anchor_id=case.anchor_id,
                                 region=region_name, region_pixels=int(region.sum()),
                                 B_comparable=int(usable.sum()), B_missing=int((region & ~bv).sum()),
                                 B_generated=int((gen & usable).sum()), depth_le025=int(near.sum()),
                                 depth_over2=int(far.sum()),
                                 depth_abs_error_sum=float(difference[usable].sum(dtype=np.float64)),
                                 raw_near_count=len(raw_error),
                                 raw_near_abs_error_median=float(np.median(raw_error)) if len(raw_error) else None,
                                 raw_near_abs_error_p95=float(np.percentile(raw_error, 95)) if len(raw_error) else None,
                                 raw_near_abs_error_sum=float(raw_error.sum(dtype=np.float64)),
                                 png_near_abs_error_sum=float(png_difference[near].sum(dtype=np.float64))))
            # C-new pixels: green <=25cm, yellow intermediate, red >2m, blue missing B.
            mask = category == 2
            overlay = cv2.cvtColor(png_c, cv2.COLOR_GRAY2BGR)
            overlay[mask & ~bv] = (255, 100, 0)
            overlay[mask & both & (difference <= .25)] = (0, 220, 0)
            overlay[mask & both & (difference > .25) & (difference <= 2)] = (0, 220, 255)
            overlay[mask & both & (difference > 2)] = (0, 0, 255)
            image = np.hstack([cv2.cvtColor(png_b, cv2.COLOR_GRAY2BGR),
                               cv2.cvtColor(png_c, cv2.COLOR_GRAY2BGR), overlay])
            target = args.output / f"db_{idx:04d}_B_C_recovery.png"
            if not cv2.imwrite(str(target), image):
                raise OSError(target)
            print(f"Checked B/C DB {idx}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(args.output / "per_anchor.csv", index=False)
    sums = table.groupby("region").sum(numeric_only=True)
    summary = sums[["region_pixels", "B_comparable", "B_missing", "B_generated",
                    "depth_le025", "depth_over2"]].copy()
    summary["B_coverage_pct"] = 100 * sums.B_comparable / sums.region_pixels
    summary["depth_le025_pct_of_comparable"] = 100 * sums.depth_le025 / sums.B_comparable.replace(0, np.nan)
    summary["depth_over2_pct_of_comparable"] = 100 * sums.depth_over2 / sums.B_comparable.replace(0, np.nan)
    summary["depth_mae_m"] = sums.depth_abs_error_sum / sums.B_comparable.replace(0, np.nan)
    summary["raw_intensity_mae_near"] = sums.raw_near_abs_error_sum / sums.raw_near_count.replace(0, np.nan)
    summary["png_mae_near"] = sums.png_near_abs_error_sum / sums.raw_near_count.replace(0, np.nan)
    summary.to_csv(args.output / "summary.csv")
    print(summary.round(4).to_string())
    (args.output / "settings.json").write_text(json.dumps(
        {"source": str(args.source.resolve()), "near_depth_m": .25, "far_depth_m": 2,
         "note": "Same-pixel discrepancies include occlusion and rasterization. Raw intensities use stored map units. PNG intensities include postprocessing. Pixel-weighted representative anchors."}, indent=2))
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
