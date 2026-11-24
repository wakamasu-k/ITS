#!/usr/bin/env python3
"""Project Waymo 16-to-32 LiDAR outputs into a camera image.

This diagnostic keeps the TULIP model unchanged and compares two geometries:
GT range + TULIP intensity, and TULIP range + TULIP intensity.  Raw projected
values are saved in NPZ; PNG files are visualization-only percentile maps.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tulip_adapter.waymo_16_to_32_dataset import Waymo16To32Dataset
from tulip_adapter.waymo_16_to_32_tulip import build_waymo_16_to_32_model
from tulip_adapter.waymo_pointcloud import (
    build_top_geometry,
    paired_vehicle_points,
)
from tools.reconstruct_waymo_16_to_32_pointcloud import (
    _raw_signals,
    _read_frame,
    _source_from_pair_json,
    _top_pose,
)


def parse_args() -> argparse.Namespace:
    """Parse camera projection arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--tulip-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dataset-role", default="validation")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--camera", default="FRONT", choices=("FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"))
    parser.add_argument("--range-scale-m", type=float, default=1.0)
    parser.add_argument("--intensity-transform", choices=("raw", "log1p"), default="raw")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _camera_name(camera: str) -> int:
    """Resolve a Waymo camera enum name."""
    from waymo_open_dataset import dataset_pb2

    if not hasattr(dataset_pb2.CameraName, camera):
        raise ValueError(f"unknown camera name: {camera}")
    return int(getattr(dataset_pb2.CameraName, camera))


def _matrix4(values: Iterable[float]) -> np.ndarray:
    """Decode 16 row-major values to a 4x4 float64 matrix."""
    matrix = np.asarray(list(values), dtype=np.float64)
    if matrix.size != 16:
        raise ValueError(f"expected 16 matrix values, got {matrix.size}")
    return matrix.reshape(4, 4)


def project_vehicle_to_camera(
    points_vehicle: np.ndarray,
    calibration: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project Waymo vehicle-frame points to camera pixels.

    Input shape: ``points_vehicle`` is ``[N,3]``.  ``calibration`` is a Waymo
    CameraCalibration protobuf.  Output shapes are ``u [N]``, ``v [N]``,
    ``depth [N]`` and ``front [N]``.  The Waymo convention used here is
    ``X_c`` forward, ``u = fx*(-Y_c)/X_c + cx`` and
    ``v = fy*(-Z_c)/X_c + cy``.
    """
    points = np.asarray(points_vehicle, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_vehicle must be [N,3], got {points.shape}")
    intrinsic = list(calibration.intrinsic)
    if len(intrinsic) < 4:
        raise ValueError("camera intrinsic must contain fx, fy, cx, cy")
    fx, fy, cx, cy = (float(value) for value in intrinsic[:4])
    vehicle_from_camera = _matrix4(calibration.extrinsic.transform)
    camera_from_vehicle = np.linalg.inv(vehicle_from_camera)
    homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1
    )
    camera = (homogeneous @ camera_from_vehicle.T)[:, :3]
    depth = camera[:, 0]
    front = np.isfinite(depth) & (depth > 0.1)
    safe_depth = np.where(front, depth, 1.0)
    u = fx * (-camera[:, 1]) / safe_depth + cx
    v = fy * (-camera[:, 2]) / safe_depth + cy
    return u.astype(np.float32), v.astype(np.float32), depth.astype(np.float32), front


def _decode_camera_image(frame: Any, camera_id: int) -> np.ndarray:
    """Decode one Waymo JPEG image to an RGB uint8 array [H,W,3]."""
    record = next((item for item in frame.images if item.name == camera_id), None)
    if record is None:
        raise ValueError(f"camera image {camera_id} is missing from frame")
    encoded = np.frombuffer(record.image, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("failed to decode Waymo camera JPEG")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _camera_calibration(frame: Any, camera_id: int) -> Any:
    """Find one camera calibration by enum value."""
    matches = [item for item in frame.context.camera_calibrations if item.name == camera_id]
    if len(matches) != 1:
        raise ValueError(f"expected one camera calibration, found {len(matches)}")
    return matches[0]


def _in_image(u: np.ndarray, v: np.ndarray, width: int, height: int) -> np.ndarray:
    """Return finite pixel coordinates inside an image."""
    return (
        np.isfinite(u) & np.isfinite(v)
        & (u >= 0.0) & (u < float(width))
        & (v >= 0.0) & (v < float(height))
    )


def _zbuffer(
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    values: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a nearest-point projected map and its occupancy mask."""
    result = np.zeros((height, width), dtype=np.float32)
    occupancy = np.zeros((height, width), dtype=bool)
    nearest = np.full((height, width), np.inf, dtype=np.float32)
    order = np.argsort(depth)[::-1]
    for index in order:
        px = int(u[index])
        py = int(v[index])
        if depth[index] < nearest[py, px]:
            nearest[py, px] = depth[index]
            result[py, px] = values[index]
            occupancy[py, px] = True
    return result, occupancy


def _visualize(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Percentile-normalize a raw projected map to uint8 for PNG only."""
    output = np.zeros(values.shape, dtype=np.uint8)
    finite = mask & np.isfinite(values) & (values > 0.0)
    if not finite.any():
        return output
    upper = float(np.percentile(values[finite], 99.5))
    upper = max(upper, 1e-6)
    output[finite] = np.clip(values[finite] / upper * 255.0, 0.0, 255.0).astype(np.uint8)
    return output


def _overlay(image_rgb: np.ndarray, projected_u8: np.ndarray) -> np.ndarray:
    """Overlay a grayscale projected map on an RGB camera image."""
    color = cv2.applyColorMap(projected_u8, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    mask = projected_u8 > 0
    output = image_rgb.copy()
    output[mask] = (0.55 * output[mask] + 0.45 * color[mask]).astype(np.uint8)
    return output


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    """Load a trusted local checkpoint across PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def evaluate_one(
    dataset: Waymo16To32Dataset,
    model: torch.nn.Module,
    index: int,
    device: torch.device,
    camera: str,
    intensity_transform: str,
    range_scale_m: float,
    output_root: Path,
) -> Dict[str, Any]:
    """Project one sample in oracle and predicted range modes.

    Input shapes: dataset input ``[2,16,2688]`` and target ``[2,32,2688]``.
    Output: JSON-compatible projection metrics and NPZ/PNG artifacts.
    """
    if range_scale_m <= 0:
        raise ValueError("range_scale_m must be positive")
    sample = dataset[index]
    x = sample["input"].unsqueeze(0).to(device)
    y = sample["target"].unsqueeze(0).to(device)
    target_mask = sample["target_mask"].unsqueeze(0).to(device)
    with torch.inference_mode():
        prediction, _ = model(x, y, target_mask)
        _, target_raw, prediction_raw = _raw_signals(
            x, y, prediction, intensity_transform
        )
        target_raw[:, 0] = target_raw[:, 0] * range_scale_m
        prediction_raw[:, 0] = prediction_raw[:, 0] * range_scale_m
    target_native = target_raw[..., :2650][0].detach().cpu().numpy()
    prediction_native = prediction_raw[..., :2650][0].detach().cpu().numpy()
    mask = target_mask[..., :2650][0].detach().cpu().numpy().astype(bool)
    generated = np.zeros_like(mask)
    generated[1::2] = mask[1::2]
    target_range, target_intensity = target_native
    predicted_range, predicted_intensity = prediction_native
    common = (
        generated
        & np.isfinite(target_range) & (target_range > 0.0)
        & np.isfinite(predicted_range) & (predicted_range > 0.0)
        & np.isfinite(target_intensity) & np.isfinite(predicted_intensity)
    )
    if not common.any():
        raise ValueError(f"no common valid generated pixels at sample {index}")

    pair_json = Path(dataset.rows[index]["pair_json"])
    source_tfrecord, source_frame_index = _source_from_pair_json(pair_json)
    frame = _read_frame(source_tfrecord, source_frame_index)
    camera_id = _camera_name(camera)
    calibration = _camera_calibration(frame, camera_id)
    camera_rgb = _decode_camera_image(frame, camera_id)
    height, width = camera_rgb.shape[:2]
    geometry = build_top_geometry(
        frame, _top_pose(frame), np.arange(0, 64, 2, dtype=np.int32), native_width=2650
    )
    gt_points, gt_intensity = paired_vehicle_points(
        target_range, target_intensity, geometry, common
    )
    oracle_points, oracle_intensity = paired_vehicle_points(
        target_range, predicted_intensity, geometry, common
    )
    predicted_points, predicted_intensity_values = paired_vehicle_points(
        predicted_range, predicted_intensity, geometry, common
    )
    del gt_points, gt_intensity

    projections: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    all_projections: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, points, values in (
        ("oracle", oracle_points, oracle_intensity),
        ("predicted", predicted_points, predicted_intensity_values),
    ):
        u, v, depth, front = project_vehicle_to_camera(points, calibration)
        in_image = front & _in_image(u, v, width, height)
        all_projections[name] = (u, v, depth, front, in_image)
        projections[name] = (u[in_image], v[in_image], depth[in_image], values[in_image])

    oracle_u, oracle_v, oracle_depth, oracle_values = projections["oracle"]
    predicted_u, predicted_v, predicted_depth, predicted_values = projections["predicted"]
    oracle_map, oracle_mask = _zbuffer(
        oracle_u, oracle_v, oracle_depth, oracle_values, width, height
    )
    predicted_map, predicted_mask = _zbuffer(
        predicted_u, predicted_v, predicted_depth, predicted_values, width, height
    )
    output_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_root / "projection_raw.npz",
        oracle_u=oracle_u,
        oracle_v=oracle_v,
        oracle_depth=oracle_depth,
        oracle_intensity=oracle_values,
        predicted_u=predicted_u,
        predicted_v=predicted_v,
        predicted_depth=predicted_depth,
        predicted_intensity=predicted_values,
        oracle_map=oracle_map,
        oracle_mask=oracle_mask,
        predicted_map=predicted_map,
        predicted_mask=predicted_mask,
    )
    oracle_u8 = _visualize(oracle_map, oracle_mask)
    predicted_u8 = _visualize(predicted_map, predicted_mask)
    cv2.imwrite(str(output_root / "camera_rgb.png"), cv2.cvtColor(camera_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_root / "oracle_intensity_projection.png"), oracle_u8)
    cv2.imwrite(str(output_root / "predicted_intensity_projection.png"), predicted_u8)
    cv2.imwrite(str(output_root / "oracle_overlay.png"), cv2.cvtColor(_overlay(camera_rgb, oracle_u8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_root / "predicted_overlay.png"), cv2.cvtColor(_overlay(camera_rgb, predicted_u8), cv2.COLOR_RGB2BGR))

    oracle_all_u, oracle_all_v, oracle_all_depth, _, oracle_all_in = all_projections["oracle"]
    predicted_all_u, predicted_all_v, predicted_all_depth, _, predicted_all_in = all_projections["predicted"]
    paired_in_image = oracle_all_in & predicted_all_in
    if paired_in_image.any():
        pixel_shift = np.sqrt(
            (oracle_all_u[paired_in_image] - predicted_all_u[paired_in_image]) ** 2
            + (oracle_all_v[paired_in_image] - predicted_all_v[paired_in_image]) ** 2
        )
        depth_shift = np.abs(
            oracle_all_depth[paired_in_image] - predicted_all_depth[paired_in_image]
        )
    else:
        pixel_shift = np.empty((0,), dtype=np.float32)
        depth_shift = np.empty((0,), dtype=np.float32)
    return {
        "sample_index": int(index),
        "segment_id": sample["segment_id"],
        "frame_index": int(sample["frame_index"]),
        "source_tfrecord": str(source_tfrecord),
        "source_frame_index": int(source_frame_index),
        "camera": camera,
        "camera_shape": [int(height), int(width), 3],
        "mask_scope": "generated rows with common GT/predicted finite positive range",
        "common_lidar_pixels": int(common.sum()),
        "oracle_projected_points": int(oracle_u.size),
        "predicted_projected_points": int(predicted_u.size),
        "oracle_projection_rate": float(oracle_u.size / common.sum()),
        "predicted_projection_rate": float(predicted_u.size / common.sum()),
        "oracle_mask_pixels": int(oracle_mask.sum()),
        "predicted_mask_pixels": int(predicted_mask.sum()),
        "oracle_vs_predicted_pixel_shift_mean": float(pixel_shift.mean()) if pixel_shift.size else None,
        "oracle_vs_predicted_pixel_shift_p95": float(np.percentile(pixel_shift, 95)) if pixel_shift.size else None,
        "oracle_vs_predicted_depth_shift_mean_m": float(depth_shift.mean()) if depth_shift.size else None,
    }


def main() -> None:
    """Run camera projection evaluation and save per-frame artifacts."""
    args = parse_args()
    if args.range_scale_m <= 0:
        raise ValueError("range-scale-m must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("max_frames must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    dataset = Waymo16To32Dataset(
        args.index,
        dataset_role=args.dataset_role,
        intensity_transform=args.intensity_transform,
        range_scale_m=args.range_scale_m,
    )
    count = len(dataset) if args.max_frames is None else min(args.max_frames, len(dataset))
    if count < 1:
        raise ValueError("no evaluation samples")
    model = build_waymo_16_to_32_model(args.tulip_root).to(device)
    checkpoint = _load_checkpoint(args.checkpoint, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for index in range(count):
        row = evaluate_one(
            dataset, model, index, device, args.camera,
            args.intensity_transform,
            args.range_scale_m,
            args.output_root / f"sample_{index:06d}"
        )
        rows.append(row)
        print(json.dumps({"evaluated": index + 1, "frames": count}), flush=True)
    fields = list(rows[0].keys())
    with (args.output_root / "per_frame.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset_role": args.dataset_role,
        "evaluated_frames": count,
        "camera": args.camera,
        "range_scale_m": args.range_scale_m,
        "intensity_transform": args.intensity_transform,
        "coordinate_convention": "vehicle<-camera extrinsic inverse; Xc forward, u=-Yc/Xc, v=-Zc/Xc",
        "aggregate": {
            "common_lidar_pixels": int(sum(row["common_lidar_pixels"] for row in rows)),
            "oracle_projected_points": int(sum(row["oracle_projected_points"] for row in rows)),
            "predicted_projected_points": int(sum(row["predicted_projected_points"] for row in rows)),
            "oracle_projection_rate": float(np.mean([row["oracle_projection_rate"] for row in rows])),
            "predicted_projection_rate": float(np.mean([row["predicted_projection_rate"] for row in rows])),
            "oracle_vs_predicted_pixel_shift_mean": float(np.mean([row["oracle_vs_predicted_pixel_shift_mean"] for row in rows if row["oracle_vs_predicted_pixel_shift_mean"] is not None])) if any(row["oracle_vs_predicted_pixel_shift_mean"] is not None for row in rows) else None,
            "oracle_vs_predicted_pixel_shift_p95": float(np.mean([row["oracle_vs_predicted_pixel_shift_p95"] for row in rows if row["oracle_vs_predicted_pixel_shift_p95"] is not None])) if any(row["oracle_vs_predicted_pixel_shift_p95"] is not None for row in rows) else None,
        },
        "artifacts": "projection_raw.npz contains raw projected values; PNG files are percentile-normalized visualization only",
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
