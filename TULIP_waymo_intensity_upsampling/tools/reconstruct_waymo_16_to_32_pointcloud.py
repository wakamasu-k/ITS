#!/usr/bin/env python3
"""Reconstruct Waymo vehicle-frame point clouds from a TULIP prediction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tulip_adapter.waymo_16_to_32_dataset import Waymo16To32Dataset
from tulip_adapter.waymo_16_to_32_tulip import build_waymo_16_to_32_model
from tulip_adapter.waymo_pointcloud import (
    build_top_geometry,
    paired_vehicle_points,
    pointwise_metrics,
)


def parse_args() -> argparse.Namespace:
    """Parse reconstruction arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--tulip-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dataset-role", default="validation")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--intensity-transform", choices=("raw", "log1p"), default="raw")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _raw_signals(
    input_signal: torch.Tensor,
    target_signal: torch.Tensor,
    prediction: torch.Tensor,
    intensity_transform: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return range/intensity tensors in raw value units."""

    values = [item.clone() for item in (input_signal, target_signal, prediction)]
    if intensity_transform == "log1p":
        for value in values:
            value[:, 1] = torch.expm1(value[:, 1])
        values[2][:, 1] = values[2][:, 1].clamp_min(0.0)
    return values[0], values[1], values[2]


def _read_frame(tfrecord: Path, frame_index: int) -> Any:
    """Read one Waymo protobuf frame from a TFRecord."""

    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2

    if not tfrecord.is_file():
        raise FileNotFoundError(tfrecord)
    dataset = tf.data.TFRecordDataset(str(tfrecord), compression_type="")
    for index, record in enumerate(dataset):
        if index == frame_index:
            frame = dataset_pb2.Frame()
            frame.ParseFromString(record.numpy())
            return frame
    raise IndexError(f"frame {frame_index} not found in {tfrecord}")


def _top_pose(frame: Any) -> Any:
    """Parse the first-return range images and return the TOP pixel pose."""

    from waymo_open_dataset.utils import frame_utils

    _, _, _, pose = frame_utils.parse_range_image_and_camera_projection(frame)
    if pose is None:
        raise ValueError("Waymo parser returned no TOP range-image pose")
    return pose


def _source_from_pair_json(pair_json: Path) -> Tuple[Path, int]:
    """Read source TFRecord identity from the existing pair sidecar."""

    metadata = json.loads(pair_json.read_text(encoding="utf-8"))
    source = metadata.get("source_tfrecord")
    frame_index = metadata.get("frame_index")
    if not source or frame_index is None:
        raise KeyError("pair JSON must contain source_tfrecord and frame_index")
    return Path(source), int(frame_index)


def _points_for_prediction(
    range_values: np.ndarray,
    intensity_values: np.ndarray,
    geometry: Any,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create points after applying finite/positive range validity."""

    return paired_vehicle_points(range_values, intensity_values, geometry, mask)


def reconstruct_sample(
    *,
    dataset: Waymo16To32Dataset,
    sample_index: int,
    model: torch.nn.Module,
    device: torch.device,
    intensity_transform: str,
    output_root: Path,
) -> Dict[str, Any]:
    """Run TULIP and save one GT/predicted vehicle-frame point-cloud pair.

    Input shapes: Dataset input ``[2,16,2688]`` and target ``[2,32,2688]``.
    Output: a JSON-compatible summary; NPZ arrays are saved under
    ``output_root``.
    """

    sample = dataset[sample_index]
    input_signal = sample["input"].unsqueeze(0).to(device)
    target_signal = sample["target"].unsqueeze(0).to(device)
    target_mask = sample["target_mask"].unsqueeze(0).to(device)
    with torch.inference_mode():
        prediction, _ = model(input_signal, target_signal, target_mask)
        input_raw, target_raw, prediction_raw = _raw_signals(
            input_signal, target_signal, prediction, intensity_transform
        )
        nearest_raw = F.interpolate(input_raw, size=(32, 2688), mode="nearest")
        linear_raw = F.interpolate(
            input_raw, size=(32, 2688), mode="bilinear", align_corners=False
        )

    input_native = input_raw[..., :2650][0].detach().cpu().numpy()
    target_native = target_raw[..., :2650][0].detach().cpu().numpy()
    prediction_native = prediction_raw[..., :2650][0].detach().cpu().numpy()
    nearest_native = nearest_raw[..., :2650][0].detach().cpu().numpy()
    linear_native = linear_raw[..., :2650][0].detach().cpu().numpy()
    mask_native = target_mask[..., :2650][0].detach().cpu().numpy().astype(bool)
    generated_mask = np.zeros_like(mask_native)
    generated_mask[1::2] = mask_native[1::2]
    prediction_valid = (
        generated_mask
        & np.isfinite(prediction_native[0])
        & (prediction_native[0] > 0.0)
    )
    common_mask = mask_native & prediction_valid
    if not common_mask.any():
        raise ValueError("no common valid generated pixels for point-cloud comparison")

    pair_json = Path(dataset.rows[sample_index]["pair_json"])
    source_tfrecord, source_frame_index = _source_from_pair_json(pair_json)
    frame = _read_frame(source_tfrecord, source_frame_index)
    geometry = build_top_geometry(
        frame,
        _top_pose(frame),
        np.arange(0, 64, 2, dtype=np.int32),
        native_width=2650,
    )

    target_range = target_native[0]
    target_intensity = target_native[1]
    prediction_range = prediction_native[0]
    prediction_intensity = prediction_native[1]
    nearest_range = nearest_native[0]
    nearest_intensity = nearest_native[1]
    linear_range = linear_native[0]
    linear_intensity = linear_native[1]

    points_gt, intensity_gt = _points_for_prediction(
        target_range, target_intensity, geometry, common_mask
    )
    points_tulip, intensity_tulip = _points_for_prediction(
        prediction_range, prediction_intensity, geometry, common_mask
    )
    nearest_mask = common_mask & np.isfinite(nearest_range) & (nearest_range > 0.0)
    linear_mask = common_mask & np.isfinite(linear_range) & (linear_range > 0.0)
    points_nearest, intensity_nearest = _points_for_prediction(
        nearest_range, nearest_intensity, geometry, nearest_mask
    )
    points_linear, intensity_linear = _points_for_prediction(
        linear_range, linear_intensity, geometry, linear_mask
    )
    if points_gt.shape != points_tulip.shape:
        raise RuntimeError("GT and TULIP point counts differ after common masking")
    metrics = pointwise_metrics(points_tulip, points_gt, intensity_tulip, intensity_gt)
    summary: Dict[str, Any] = {
        "sample_index": int(sample_index),
        "segment_id": sample["segment_id"],
        "frame_index": int(sample["frame_index"]),
        "source_tfrecord": str(source_tfrecord),
        "source_frame_index": int(source_frame_index),
        "coordinate_frame": "Waymo vehicle frame",
        "conversion": "waymo_open_dataset.utils.range_image_utils.extract_point_cloud_from_range_image",
        "ring_ids": list(range(0, 64, 2)),
        "native_shape": [32, 2650],
        "generated_rows": list(range(1, 32, 2)),
        "mask_scope": "generated rows with GT-valid and TULIP-positive finite range",
        "gt_valid_generated_count": int(generated_mask.sum()),
        "paired_valid_count": int(common_mask.sum()),
        "tulip_valid_count": int(prediction_valid.sum()),
        "nearest_valid_count": int(nearest_mask.sum()),
        "linear_valid_count": int(linear_mask.sum()),
        "tulip_metrics": metrics,
        "nearest_metrics": pointwise_metrics(
            points_nearest, points_gt[nearest_mask[common_mask]],
            intensity_nearest, intensity_gt[nearest_mask[common_mask]],
        ) if points_nearest.shape[0] else None,
        "linear_metrics": pointwise_metrics(
            points_linear, points_gt[linear_mask[common_mask]],
            intensity_linear, intensity_gt[linear_mask[common_mask]],
        ) if points_linear.shape[0] else None,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_root / "waymo_16_to_32_pointcloud.npz",
        points_gt_vehicle=points_gt,
        intensity_gt=intensity_gt,
        points_tulip_vehicle=points_tulip,
        intensity_tulip=intensity_tulip,
        points_nearest_vehicle=points_nearest,
        intensity_nearest=intensity_nearest,
        points_linear_vehicle=points_linear,
        intensity_linear=intensity_linear,
    )
    (output_root / "metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    """Load one checkpoint, reconstruct one sample, and save NPZ/metadata."""

    args = parse_args()
    if args.sample_index < 0:
        raise ValueError("sample-index must be non-negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    dataset = Waymo16To32Dataset(
        args.index,
        dataset_role=args.dataset_role,
        intensity_transform=args.intensity_transform,
    )
    if args.sample_index >= len(dataset):
        raise IndexError(f"sample-index {args.sample_index} >= {len(dataset)}")
    model = build_waymo_16_to_32_model(args.tulip_root).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    summary = reconstruct_sample(
        dataset=dataset,
        sample_index=args.sample_index,
        model=model,
        device=device,
        intensity_transform=args.intensity_transform,
        output_root=args.output_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
