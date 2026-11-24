"""Prepare a Closing comparison using the Retrieval evaluator's default resize."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import distance_transform_edt
from torchvision import transforms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    db = pd.read_parquet(args.manifests / "A_db.parquet")
    pairs = pd.read_parquet(args.manifests / "A_pairs.parquet")
    if not set(pairs.anchor_png).issubset(db.anchor_png):
        raise ValueError("Some ground-truth anchor paths are absent from the DB")

    images = args.output.resolve() / "images"
    images.mkdir(parents=True)
    resize = transforms.Resize((512, 864))
    kernel = np.ones((3, 3), dtype=np.uint8)
    paths, coverage = {}, []
    for index, original in enumerate(db.anchor_png.unique(), start=1):
        with Image.open(original) as image:
            rgb = np.asarray(resize(image.convert("RGB")))
        valid = np.any(rgb != 0, axis=2)
        closed = cv2.morphologyEx(
            valid.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)
        added = closed & ~valid
        filled = rgb.copy()
        if added.any():
            y, x = distance_transform_edt(
                ~valid, return_distances=False, return_indices=True
            )
            filled[added] = rgb[y[added], x[added]]
        path = images / f"{index:04d}.png"
        Image.fromarray(filled).save(path)
        paths[original] = str(path)
        coverage.append({"anchor_png": original, "valid_pixels": int(valid.sum()),
                         "added_pixels": int(added.sum())})
        if index % 100 == 0:
            print(f"Prepared {index} anchors", flush=True)

    for frame, name in [(db, "db.parquet"), (pairs, "pairs.parquet")]:
        frame["anchor_png"] = frame["anchor_png"].map(paths)
        frame.to_parquet(args.output / name, index=False)
    pd.DataFrame(coverage).to_csv(args.output / "coverage.csv", index=False)
    print(f"Prepared {len(paths)} anchors in {args.output}", flush=True)


if __name__ == "__main__":
    main()
