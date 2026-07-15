import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from project_xyzi_to_gray_intensity import load_points, project_to_image


def compare(a_img, a_mask, gt_img, gt_mask):
    common = a_mask & gt_mask
    union = a_mask | gt_mask

    a_valid = int(np.count_nonzero(a_mask))
    gt_valid = int(np.count_nonzero(gt_mask))
    common_valid = int(np.count_nonzero(common))
    union_valid = int(np.count_nonzero(union))

    mask_iou = common_valid / max(union_valid, 1)

    if common_valid == 0:
        return {
            "a_valid": a_valid,
            "gt_valid": gt_valid,
            "common_valid": common_valid,
            "union_valid": union_valid,
            "mask_iou": mask_iou,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
        }

    diff = a_img[common] - gt_img[common]

    return {
        "a_valid": a_valid,
        "gt_valid": gt_valid,
        "common_valid": common_valid,
        "union_valid": union_valid,
        "mask_iou": mask_iou,
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "bias": float(np.mean(diff)),
    }


def project_ply_to_camera(points_path, calib_path, image_path, camera):
    points = load_points(points_path)
    intensity_img, mask_img, _ = project_to_image(
        points_xyzi=points,
        calib_path=calib_path,
        image_path=image_path,
        camera=camera,
    )

    intensity_img = np.clip(intensity_img, 0.0, 1.0).astype(np.float32)
    mask = mask_img > 0

    return intensity_img, mask


def mean_ignore_nan(values):
    values = np.array(values, dtype=np.float32)
    if np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmean(values))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pcd_dir",
        default="./experiment/kitti/tulip_base_intensity/pcd",
        help="directory containing low_XXXXXX.ply, pred_XXXXXX.ply, gt_XXXXXX.ply",
    )
    parser.add_argument(
        "--calib",
        default="/mnt/e/data_odometry_calib/dataset/sequences/00/calib.txt",
    )
    parser.add_argument(
        "--image",
        default="/mnt/e/data_odometry_gray/dataset/sequences/00/image_0/000000.png",
    )
    parser.add_argument(
        "--camera",
        default="P0",
        choices=["P0", "P1", "P2", "P3"],
    )
    parser.add_argument(
        "--out_csv",
        default="./debug_outputs/camera_intensity_multi_metrics.csv",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="limit number of frames for quick test",
    )

    args = parser.parse_args()

    pcd_dir = Path(args.pcd_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(pcd_dir.glob("pred_*.ply"))

    frame_ids = []
    for pred_path in pred_files:
        frame_id = pred_path.stem.replace("pred_", "")
        gt_path = pcd_dir / f"gt_{frame_id}.ply"
        low_path = pcd_dir / f"low_{frame_id}.ply"

        if gt_path.exists() and low_path.exists():
            frame_ids.append(frame_id)

    if args.max_frames is not None:
        frame_ids = frame_ids[:args.max_frames]

    print("matched frames:", len(frame_ids))

    if len(frame_ids) == 0:
        print("No matched low/pred/gt ply triplets found.")
        return

    rows = []

    for i, frame_id in enumerate(frame_ids):
        print(f"[{i+1}/{len(frame_ids)}] frame {frame_id}")

        low_path = pcd_dir / f"low_{frame_id}.ply"
        pred_path = pcd_dir / f"pred_{frame_id}.ply"
        gt_path = pcd_dir / f"gt_{frame_id}.ply"

        low_img, low_mask = project_ply_to_camera(
            low_path, args.calib, args.image, args.camera
        )
        pred_img, pred_mask = project_ply_to_camera(
            pred_path, args.calib, args.image, args.camera
        )
        gt_img, gt_mask = project_ply_to_camera(
            gt_path, args.calib, args.image, args.camera
        )

        low_metrics = compare(low_img, low_mask, gt_img, gt_mask)
        pred_metrics = compare(pred_img, pred_mask, gt_img, gt_mask)

        row = {
            "frame_id": frame_id,

            "low_valid": low_metrics["a_valid"],
            "pred_valid": pred_metrics["a_valid"],
            "gt_valid": pred_metrics["gt_valid"],

            "low_common": low_metrics["common_valid"],
            "pred_common": pred_metrics["common_valid"],

            "low_mask_iou": low_metrics["mask_iou"],
            "pred_mask_iou": pred_metrics["mask_iou"],

            "low_mae": low_metrics["mae"],
            "pred_mae": pred_metrics["mae"],

            "low_rmse": low_metrics["rmse"],
            "pred_rmse": pred_metrics["rmse"],

            "low_bias": low_metrics["bias"],
            "pred_bias": pred_metrics["bias"],
        }

        rows.append(row)

    fieldnames = list(rows[0].keys())

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("saved csv:", out_csv)
    print()
    print("===== Average over frames =====")
    print("frames:", len(rows))

    for key in [
        "low_valid",
        "pred_valid",
        "gt_valid",
        "low_common",
        "pred_common",
        "low_mask_iou",
        "pred_mask_iou",
        "low_mae",
        "pred_mae",
        "low_rmse",
        "pred_rmse",
        "low_bias",
        "pred_bias",
    ]:
        print(f"{key}: {mean_ignore_nan([r[key] for r in rows])}")

    low_mae = mean_ignore_nan([r["low_mae"] for r in rows])
    pred_mae = mean_ignore_nan([r["pred_mae"] for r in rows])
    low_rmse = mean_ignore_nan([r["low_rmse"] for r in rows])
    pred_rmse = mean_ignore_nan([r["pred_rmse"] for r in rows])

    print()
    print("===== Improvement =====")
    if not np.isnan(low_mae) and low_mae != 0:
        print("MAE reduction:", (low_mae - pred_mae) / low_mae)
    if not np.isnan(low_rmse) and low_rmse != 0:
        print("RMSE reduction:", (low_rmse - pred_rmse) / low_rmse)


if __name__ == "__main__":
    main()