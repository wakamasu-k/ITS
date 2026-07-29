#!/usr/bin/env python3
"""Create traceable KITTI GT/low range images for one odometry frame.

This tool deliberately runs before TULIP inference.  It first proves that the
raw XYZI -> range image -> reconstructed XYZI path is internally consistent.
Every output name and metadata record includes the source sequence and frame,
so an evaluation step can no longer be mistaken for a KITTI frame number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tulip.util.kitti_range_image import (  # noqa: E402
    DEFAULT_LOW_ROW_INDICES,
    KITTI_COLS,
    KITTI_ROWS,
    load_kitti_xyzi,
    project_xyzi_to_range_image,
    restore_low_rows,
    select_low_rows,
    unproject_range_image_to_xyzi,
    validate_range_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose one traceable KITTI odometry LiDAR frame."
    )
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--camera", choices=("P0", "P1", "P2", "P3"), default="P0")
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--frame", default="000000")
    parser.add_argument("--rows", type=int, default=KITTI_ROWS)
    parser.add_argument("--cols", type=int, default=KITTI_COLS)
    parser.add_argument("--max_range_m", type=float, default=80.0)
    parser.add_argument(
        "--low_rows",
        type=int,
        nargs="+",
        default=list(DEFAULT_LOW_ROW_INDICES),
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    return parser.parse_args()


def write_ascii_xyzi_ply(path: Path, points: np.ndarray) -> None:
    """Write an ASCII XYZI PLY that existing camera tools can read."""

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(f"Expected [N,4] XYZI points, got {points.shape}")
    with path.open("w", encoding="utf-8") as stream:
        stream.write("ply\n")
        stream.write("format ascii 1.0\n")
        stream.write(f"element vertex {points.shape[0]}\n")
        stream.write("property float x\n")
        stream.write("property float y\n")
        stream.write("property float z\n")
        stream.write("property float intensity\n")
        stream.write("end_header\n")
        np.savetxt(stream, points, fmt="%.7f %.7f %.7f %.7f")


def save_range_image_visuals(
    prefix: Path,
    image: np.ndarray,
    *,
    max_range_m: float,
) -> None:
    """Save fixed-scale range/intensity/mask images for comparable viewing."""

    radial_range = image[..., 0]
    intensity = image[..., 1]
    mask = radial_range > 0.0

    range_u8 = np.rint(
        np.clip(radial_range / max_range_m, 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    intensity_u8 = np.rint(
        np.clip(intensity, 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    mask_u8 = mask.astype(np.uint8) * 255

    cv2.imwrite(str(prefix.with_name(prefix.name + "_range.png")), range_u8)
    cv2.imwrite(
        str(prefix.with_name(prefix.name + "_intensity.png")), intensity_u8
    )
    cv2.imwrite(str(prefix.with_name(prefix.name + "_mask.png")), mask_u8)
    cv2.imwrite(
        str(prefix.with_name(prefix.name + "_range_color.png")),
        cv2.applyColorMap(range_u8, cv2.COLORMAP_TURBO),
    )
    cv2.imwrite(
        str(prefix.with_name(prefix.name + "_intensity_color.png")),
        cv2.applyColorMap(intensity_u8, cv2.COLORMAP_TURBO),
    )


def roundtrip_metrics(
    raw_points: np.ndarray,
    reconstructed: np.ndarray,
    source_index_image: np.ndarray,
) -> dict[str, float | int]:
    """Compare reconstructed points with raw returns selected by the z-buffer."""

    source_indices = source_index_image[source_index_image >= 0]
    selected_raw = raw_points[source_indices]
    if selected_raw.shape != reconstructed.shape:
        raise AssertionError(
            f"Round-trip point mismatch: raw={selected_raw.shape}, "
            f"reconstructed={reconstructed.shape}"
        )

    xyz_error = np.linalg.norm(
        reconstructed[:, :3].astype(np.float64)
        - selected_raw[:, :3].astype(np.float64),
        axis=1,
    )
    raw_range = np.linalg.norm(selected_raw[:, :3].astype(np.float64), axis=1)
    reconstructed_range = np.linalg.norm(
        reconstructed[:, :3].astype(np.float64), axis=1
    )
    intensity_error = np.abs(
        reconstructed[:, 3].astype(np.float64)
        - selected_raw[:, 3].astype(np.float64)
    )
    return {
        "compared_point_count": int(reconstructed.shape[0]),
        "xyz_quantization_mae_m": float(xyz_error.mean()),
        "xyz_quantization_rmse_m": float(np.sqrt(np.mean(xyz_error**2))),
        "xyz_quantization_max_m": float(xyz_error.max()),
        "range_roundtrip_mae_m": float(
            np.mean(np.abs(reconstructed_range - raw_range))
        ),
        "intensity_roundtrip_mae": float(intensity_error.mean()),
        "intensity_roundtrip_max": float(intensity_error.max()),
    }


def main() -> None:
    args = parse_args()
    if args.rows != KITTI_ROWS or args.cols != KITTI_COLS:
        raise ValueError(
            "The current TULIP KITTI geometry is fixed to "
            f"{KITTI_ROWS}x{KITTI_COLS}; got {args.rows}x{args.cols}"
        )
    for path in (args.points, args.image, args.calib):
        if not path.is_file():
            raise FileNotFoundError(path)

    # Reading the image here verifies the named camera frame exists and records
    # its dimensions even though camera overlay generation remains a later step.
    camera_image = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
    if camera_image is None:
        raise ValueError(f"Could not read camera image: {args.image}")

    output_dir = args.out_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_id = f"seq{args.sequence}_frame{args.frame}"

    raw_points = load_kitti_xyzi(args.points)
    gt, source_index_image, projection_stats = project_xyzi_to_range_image(
        raw_points,
        rows=args.rows,
        cols=args.cols,
        max_range_m=args.max_range_m,
    )
    low_rows = tuple(args.low_rows)
    low = select_low_rows(gt, low_rows)
    low_restored = restore_low_rows(low, low_rows, full_rows=args.rows)

    gt_points = unproject_range_image_to_xyzi(
        gt,
        full_rows=args.rows,
        cols=args.cols,
        range_unit="meter",
        max_range_m=args.max_range_m,
    )
    low_points = unproject_range_image_to_xyzi(
        low,
        row_indices=low_rows,
        full_rows=args.rows,
        cols=args.cols,
        range_unit="meter",
        max_range_m=args.max_range_m,
    )

    np.save(output_dir / f"{scene_id}_gt_64x1024x2.npy", gt)
    np.save(output_dir / f"{scene_id}_low_16x1024x2.npy", low)
    np.save(
        output_dir / f"{scene_id}_low_restored_64x1024x2.npy", low_restored
    )
    write_ascii_xyzi_ply(output_dir / f"{scene_id}_raw_xyzi.ply", raw_points)
    write_ascii_xyzi_ply(
        output_dir / f"{scene_id}_gt_reconstructed_xyzi.ply", gt_points
    )
    write_ascii_xyzi_ply(
        output_dir / f"{scene_id}_low_reconstructed_xyzi.ply", low_points
    )

    save_range_image_visuals(
        output_dir / f"{scene_id}_gt", gt, max_range_m=args.max_range_m
    )
    save_range_image_visuals(
        output_dir / f"{scene_id}_low_restored",
        low_restored,
        max_range_m=args.max_range_m,
    )

    metadata = {
        "scene_id": scene_id,
        "sequence": args.sequence,
        "frame": args.frame,
        "source_lidar": str(args.points.resolve()),
        "source_camera": str(args.image.resolve()),
        "source_calibration": str(args.calib.resolve()),
        "camera_projection": args.camera,
        "camera_image_shape": list(camera_image.shape),
        "range_image_shape": list(gt.shape),
        "range_unit": "meter",
        "maximum_range_m": args.max_range_m,
        "intensity_expected_range": [0.0, 1.0],
        "low_row_indices": list(low_rows),
        "invalid_pixel_policy": "range=0 and intensity=0",
        "collision_policy": "nearest radial range",
    }
    metrics = {
        "projection": projection_stats.as_json_dict(),
        "gt": validate_range_image(gt, max_range_m=args.max_range_m),
        "low": validate_range_image(low, max_range_m=args.max_range_m),
        "low_restored": validate_range_image(
            low_restored, max_range_m=args.max_range_m
        ),
        "gt_roundtrip": roundtrip_metrics(
            raw_points, gt_points, source_index_image
        ),
        "low_is_exact_gt_subset": bool(np.array_equal(low, gt[list(low_rows)])),
        "low_restoration_is_exact": bool(
            np.array_equal(low_restored[list(low_rows)], low)
        ),
    }

    (output_dir / f"{scene_id}_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / f"{scene_id}_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"metadata": metadata, "metrics": metrics}, indent=2))
    print(f"Saved diagnostic outputs to: {output_dir}")


if __name__ == "__main__":
    main()
