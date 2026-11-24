#!/usr/bin/env python3
"""Export separate GT, Low, and TULIP-predicted Waymo point clouds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tulip_adapter.waymo_16_to_32_dataset import Waymo16To32Dataset
from tulip_adapter.waymo_16_to_32_tulip import build_waymo_16_to_32_model
from tulip_adapter.tulip_static_map import (
    fuse_16_to_32_observed_generated,
    valid_top_pixel_pose_mask,
)
from tulip_adapter.waymo_pointcloud import (
    build_top_geometry,
    paired_vehicle_points,
    pointwise_metrics,
)
from tools.reconstruct_waymo_16_to_32_pointcloud import (
    _read_frame,
    _source_from_pair_json,
    _top_pose,
)


RING_IDS_32 = np.arange(0, 64, 2, dtype=np.int32)
GENERATED_ROWS = np.arange(1, 32, 2, dtype=np.int32)
INTENSITY_BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("0_0.5", 0.0, 0.5),
    ("0.5_1", 0.5, 1.0),
    ("1_2", 1.0, 2.0),
    ("2_5", 2.0, 5.0),
    ("5_20", 5.0, 20.0),
    ("20_100", 20.0, 100.0),
    ("100_1000", 100.0, 1000.0),
    ("1000_inf", 1000.0, float("inf")),
)


def parse_args() -> argparse.Namespace:
    """Parse export options.

    Input shapes: the index describes samples whose Dataset tensors are
    ``input [2,16,2688]`` and ``target [2,32,2688]``.  Output is one NPZ per
    selected sample containing variable-length point arrays ``[N,3]``.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--tulip-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dataset-role", default="test")
    parser.add_argument("--sample-index", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--range-scale-m", type=float, default=75.0)
    parser.add_argument("--intensity-transform", choices=("raw", "log1p"), default="log1p")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def restore_domain(
    tensor: torch.Tensor,
    *,
    range_scale_m: float,
    intensity_transform: str,
) -> torch.Tensor:
    """Restore model-domain values to raw range meters and intensity.

    Input shape: ``[B,2,H,W]``.  Output shape: ``[B,2,H,W]``.  Channel 0 is
    multiplied by ``range_scale_m``; channel 1 is inverse-transformed only
    when ``intensity_transform`` is ``log1p``.
    """

    if tensor.ndim != 4 or tensor.shape[1] != 2:
        raise ValueError(f"tensor must be [B,2,H,W], got {tuple(tensor.shape)}")
    if range_scale_m <= 0.0:
        raise ValueError("range_scale_m must be positive")
    result = tensor.clone()
    result[:, 0] *= range_scale_m
    if intensity_transform == "log1p":
        result[:, 1] = torch.expm1(result[:, 1]).clamp_min(0.0)
    elif intensity_transform != "raw":
        raise ValueError(f"unsupported intensity_transform: {intensity_transform}")
    return result


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    """Load a trusted local checkpoint across PyTorch versions."""

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _place_low_on_32_grid(
    input_raw: np.ndarray,
    input_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Place a 16-line input on the even rows of a 32-line grid.

    Input shapes: ``input_raw [2,16,2650]`` and ``input_mask [16,2650]``.
    Output shapes: sparse raw signal ``[2,32,2650]`` and bool mask
    ``[32,2650]``.  Odd rows are zero and invalid.
    """

    if input_raw.shape != (2, 16, 2650):
        raise ValueError(f"input_raw must be [2,16,2650], got {input_raw.shape}")
    if input_mask.shape != (16, 2650):
        raise ValueError(f"input_mask must be [16,2650], got {input_mask.shape}")
    sparse = np.zeros((2, 32, 2650), dtype=np.float32)
    sparse_mask = np.zeros((32, 2650), dtype=bool)
    sparse[:, 0::2] = input_raw
    sparse_mask[0::2] = input_mask
    sparse[:, ~sparse_mask] = 0.0
    return sparse, sparse_mask


def _valid_signal_mask(signal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return finite, positive-range pixels with finite intensity.

    Input shapes: ``signal [2,H,W]`` and ``mask [H,W]``.  Output shape:
    boolean ``[H,W]``.
    """

    if signal.ndim != 3 or signal.shape[0] != 2 or mask.shape != signal.shape[1:]:
        raise ValueError("signal must be [2,H,W] and mask must match [H,W]")
    return (
        mask.astype(bool)
        & np.isfinite(signal[0])
        & (signal[0] > 0.0)
        & np.isfinite(signal[1])
    )


def _band_summary(intensity: np.ndarray) -> Dict[str, Dict[str, float | int]]:
    """Summarize finite non-negative intensity values by fixed bands.

    Input shape: one-dimensional intensity array ``[N]``.  Output is a
    JSON-compatible dictionary of counts, fractions, and means.
    """

    values = np.asarray(intensity, dtype=np.float32)
    values = values[np.isfinite(values) & (values >= 0.0)]
    total = int(values.size)
    result: Dict[str, Dict[str, float | int]] = {}
    for name, lower, upper in INTENSITY_BANDS:
        selected = values[(values >= lower) & (values < upper)]
        result[name] = {
            "count": int(selected.size),
            "fraction": float(selected.size / total) if total else 0.0,
            "mean": float(selected.mean()) if selected.size else 0.0,
        }
    return result


def _points(
    signal: np.ndarray,
    mask: np.ndarray,
    geometry: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert one raw signal grid into paired vehicle-frame points.

    Input shapes: signal ``[2,32,2650]`` and mask ``[32,2650]``.  Output
    shapes: points ``[N,3]`` and intensity ``[N]``.
    """

    selected = _valid_signal_mask(signal, mask)
    return paired_vehicle_points(signal[0], signal[1], geometry, selected)


def export_sample(
    *,
    dataset: Waymo16To32Dataset,
    sample_index: int,
    model: torch.nn.Module,
    device: torch.device,
    range_scale_m: float,
    intensity_transform: str,
    output_root: Path,
) -> Dict[str, Any]:
    """Export separate GT, sparse Low, and Pred point clouds for one sample.

    Input shapes: Dataset input ``[2,16,2688]`` and target ``[2,32,2688]``.
    Output files contain variable-length vehicle-frame point arrays ``[N,3]``
    and intensity arrays ``[N]`` for GT, Low, Pred, and generated rows.
    """

    sample = dataset[sample_index]
    input_signal = sample["input"].unsqueeze(0).to(device)
    target_signal = sample["target"].unsqueeze(0).to(device)
    input_mask = sample["input_mask"].numpy().astype(bool)
    target_mask = sample["target_mask"].numpy().astype(bool)[:, :2650]
    with torch.inference_mode():
        prediction, _ = model(input_signal, target_signal, sample["target_mask"].unsqueeze(0).to(device))
        input_raw = restore_domain(
            input_signal, range_scale_m=range_scale_m, intensity_transform=intensity_transform
        )[0, :, :, :2650].detach().cpu().numpy().astype(np.float32)
        target_raw = restore_domain(
            target_signal, range_scale_m=range_scale_m, intensity_transform=intensity_transform
        )[0, :, :, :2650].detach().cpu().numpy().astype(np.float32)
        prediction_raw = restore_domain(
            prediction, range_scale_m=range_scale_m, intensity_transform=intensity_transform
        )[0, :, :, :2650].detach().cpu().numpy().astype(np.float32)

    low_raw, low_mask = _place_low_on_32_grid(input_raw, input_mask[:, :2650])
    pred_mask = _valid_signal_mask(prediction_raw, target_mask)
    gt_mask = _valid_signal_mask(target_raw, target_mask)

    pair_json = Path(dataset.rows[sample_index]["pair_json"])
    source_tfrecord, source_frame_index = _source_from_pair_json(pair_json)
    frame = _read_frame(source_tfrecord, source_frame_index)
    geometry = build_top_geometry(
        frame,
        _top_pose(frame),
        RING_IDS_32,
        native_width=2650,
    )

    points_gt, intensity_gt = _points(target_raw, gt_mask, geometry)
    points_low, intensity_low = _points(low_raw, low_mask, geometry)

    fused_raw, fused_mask, fusion_stats = fuse_16_to_32_observed_generated(
        input_raw, input_mask[:, :2650], prediction_raw
    )
    pose_valid = valid_top_pixel_pose_mask(geometry.pixel_pose[0].numpy())
    invalid_generated_pose = fused_mask[GENERATED_ROWS] & ~pose_valid[GENERATED_ROWS]
    fusion_stats["generated_invalid_pixel_pose_removed_pixels"] = int(
        invalid_generated_pose.sum()
    )
    fused_mask[GENERATED_ROWS] &= pose_valid[GENERATED_ROWS]
    fused_raw[:, ~fused_mask] = 0.0
    points_pred, intensity_pred = _points(fused_raw, fused_mask, geometry)

    generated = np.zeros((32, 2650), dtype=bool)
    generated[GENERATED_ROWS] = True
    common_generated = generated & gt_mask & pred_mask
    points_gt_generated, intensity_gt_generated = _points(target_raw, common_generated, geometry)
    points_pred_generated, intensity_pred_generated = _points(prediction_raw, common_generated, geometry)

    output_dir = output_root / f"sample_{sample_index:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "gt_low_pred_pointcloud.npz",
        points_gt_vehicle=points_gt,
        intensity_gt=intensity_gt,
        points_low_vehicle=points_low,
        intensity_low=intensity_low,
        points_pred_vehicle=points_pred,
        intensity_pred=intensity_pred,
        points_gt_generated_vehicle=points_gt_generated,
        intensity_gt_generated=intensity_gt_generated,
        points_pred_generated_vehicle=points_pred_generated,
        intensity_pred_generated=intensity_pred_generated,
    )

    summary: Dict[str, Any] = {
        "sample_index": int(sample_index),
        "segment_id": sample["segment_id"],
        "frame_index": int(sample["frame_index"]),
        "source_tfrecord": str(source_tfrecord),
        "source_frame_index": int(source_frame_index),
        "coordinate_frame": "Waymo vehicle frame",
        "range_scale_m": float(range_scale_m),
        "intensity_transform": intensity_transform,
        "native_shape": [32, 2650],
        "ring_ids": RING_IDS_32.tolist(),
        "generated_rows": GENERATED_ROWS.tolist(),
        "mask_scope": {
            "gt_low_pred": "GT own mask; Low observed even rows; Pred observed even + safe generated odd rows",
            "generated_metrics": "GT/Pred common valid pixels on odd generated rows",
        },
        "fusion_safety": fusion_stats,
        "counts": {
            "gt": int(points_gt.shape[0]),
            "low": int(points_low.shape[0]),
            "pred": int(points_pred.shape[0]),
            "gt_generated_common": int(points_gt_generated.shape[0]),
            "pred_generated_common": int(points_pred_generated.shape[0]),
        },
        "intensity_bands": {
            "gt": _band_summary(intensity_gt),
            "low": _band_summary(intensity_low),
            "pred": _band_summary(intensity_pred),
        },
        "generated_metrics": pointwise_metrics(
            points_pred_generated,
            points_gt_generated,
            intensity_pred_generated,
            intensity_gt_generated,
        ) if points_pred_generated.shape[0] else None,
        "finite": {
            "gt_points": bool(np.isfinite(points_gt).all()),
            "low_points": bool(np.isfinite(points_low).all()),
            "pred_points": bool(np.isfinite(points_pred).all()),
            "gt_intensity": bool(np.isfinite(intensity_gt).all()),
            "low_intensity": bool(np.isfinite(intensity_low).all()),
            "pred_intensity": bool(np.isfinite(intensity_pred).all()),
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _sample_indices(length: int, sample_index: Optional[int], max_frames: Optional[int]) -> List[int]:
    """Validate and select Dataset indices."""

    if sample_index is not None and max_frames is not None:
        raise ValueError("--sample-index and --max-frames cannot be used together")
    if sample_index is not None:
        if sample_index < 0 or sample_index >= length:
            raise IndexError(f"sample-index {sample_index} outside dataset length {length}")
        return [sample_index]
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max-frames must be positive")
    count = min(length, max_frames) if max_frames is not None else length
    return list(range(count))


def main() -> None:
    """Export selected samples and write a run summary."""

    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    dataset = Waymo16To32Dataset(
        args.index,
        dataset_role=args.dataset_role,
        intensity_transform=args.intensity_transform,
        range_scale_m=args.range_scale_m,
    )
    indices = _sample_indices(len(dataset), args.sample_index, args.max_frames)
    model = build_waymo_16_to_32_model(args.tulip_root).to(device)
    checkpoint = _load_checkpoint(args.checkpoint, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for index in indices:
        try:
            rows.append(
                export_sample(
                    dataset=dataset,
                    sample_index=index,
                    model=model,
                    device=device,
                    range_scale_m=args.range_scale_m,
                    intensity_transform=args.intensity_transform,
                    output_root=args.output_root,
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "sample_index": int(index),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    run_summary = {
        "checkpoint": str(args.checkpoint),
        "dataset_role": args.dataset_role,
        "range_scale_m": float(args.range_scale_m),
        "intensity_transform": args.intensity_transform,
        "requested_samples": indices,
        "exported_samples": len(rows),
        "failed_samples": len(failures),
        "output_root": str(args.output_root),
        "failures": failures,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))
    if not rows:
        raise RuntimeError("all selected samples failed")


if __name__ == "__main__":
    main()
