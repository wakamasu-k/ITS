#!/usr/bin/env python3
"""Build one static map from a 16→32 TULIP checkpoint without changing TULIP.

The output ``map_static.npz`` intentionally follows the existing Shimizu
static-map contract: it contains only world-frame ``xyz`` and raw
``intensity`` arrays.  Observed 16 rings are retained as measurements and
only the missing 16 rings are supplied by the TULIP prediction.  A
``source_is_generated`` array is stored alongside them so rendered pixels can
be traced back to measured or TULIP-generated map points.
"""
from __future__ import annotations

import argparse
import csv
import faulthandler
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_START_TIME = time.perf_counter()


def _progress(stage: str, **details: Any) -> None:
    """Expose startup/first-frame waits without changing inference settings."""
    if __name__ == '__main__':
        print(json.dumps({"stage": stage, "elapsed_s": round(time.perf_counter()-_START_TIME, 2),
                          **details}), flush=True)
        if stage == "first_frame_geometry_done":
            # Startup is complete; normal long map runs should not emit timeout stacks.
            faulthandler.cancel_dump_traceback_later()


_progress("import_numpy_torch_begin")
import numpy as np
import torch
_progress("import_numpy_torch_done", torch_version=torch.__version__)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tulip_adapter.tulip_static_map import (
    MAX_GENERATED_RANGE_M,
    NATIVE_WIDTH,
    fuse_16_to_32_observed_generated,
    static_point_mask,
    valid_top_pixel_pose_mask,
    vehicle_to_world,
)
from tulip_adapter.waymo_16_to_32_dataset import _circular_pad, transform_intensity
from tulip_adapter.waymo_16_to_32_tulip import build_waymo_16_to_32_model
from tulip_adapter.waymo_pointcloud import build_top_geometry, paired_vehicle_points
from tools.reconstruct_waymo_16_to_32_pointcloud import _top_pose
from waymo_preprocess.export_one_frame_64_32 import (
    extract_range_intensity,
    select_even_rings,
    top_first_return,
)
_progress("import_waymo_helpers_done")


RING_IDS_32 = np.arange(0, 64, 2, dtype=np.int32)
VARIANT_NAMES = {
    "A": "A_observed16",
    "B": "B_tulip32",
    "C": "C_gt32",
}


def parse_args() -> argparse.Namespace:
    """Parse one explicit TFRecord-to-static-map conversion."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecord", required=True, type=Path)
    parser.add_argument("--subset", required=True, choices=("training", "validation", "testing"))
    parser.add_argument("--tulip-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--range-scale-m", type=float, required=True)
    parser.add_argument("--intensity-transform", choices=("raw", "log1p"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--map-variants", nargs="+", choices=tuple(VARIANT_NAMES),
        default=list(VARIANT_NAMES),
        help="A=observed16, B=observed16+TULIP odd rows, C=measured GT32",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write a compressed NPZ through a same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Write JSON through a same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _model_arrays(
    range_32: np.ndarray,
    intensity_32: np.ndarray,
    mask_32: np.ndarray,
    *,
    range_scale_m: float,
    intensity_transform: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Prepare padded model tensors and retain raw measured 16-ring arrays."""
    range_16 = np.asarray(range_32[::2], dtype=np.float32)
    intensity_16 = np.asarray(intensity_32[::2], dtype=np.float32)
    mask_16 = np.asarray(mask_32[::2], dtype=bool)
    input_raw = np.stack((range_16, intensity_16), axis=0)

    input_model = input_raw.copy()
    target_model = np.stack((range_32, intensity_32), axis=0).astype(np.float32)
    input_model[0] /= np.float32(range_scale_m)
    target_model[0] /= np.float32(range_scale_m)
    input_model[1] = transform_intensity(input_model[1], intensity_transform, mask_16)
    target_model[1] = transform_intensity(target_model[1], intensity_transform, mask_32)
    input_model = _circular_pad(np.moveaxis(input_model, 0, -1), NATIVE_WIDTH)
    target_model = _circular_pad(np.moveaxis(target_model, 0, -1), NATIVE_WIDTH)
    input_mask = _circular_pad(mask_16, NATIVE_WIDTH)
    target_mask = _circular_pad(mask_32, NATIVE_WIDTH)
    return (
        torch.from_numpy(np.moveaxis(input_model, -1, 0)).unsqueeze(0),
        torch.from_numpy(np.moveaxis(target_model, -1, 0)).unsqueeze(0),
        torch.from_numpy(target_mask).unsqueeze(0),
        input_raw,
        input_mask,
    )


def _restore_prediction(
    prediction: torch.Tensor,
    *,
    range_scale_m: float,
    intensity_transform: str,
) -> np.ndarray:
    """Restore ``[1,2,32,2688]`` model output to raw ``[2,32,2650]``."""
    raw = prediction.detach().cpu().clone()
    raw[:, 0] *= float(range_scale_m)
    if intensity_transform == "log1p":
        raw[:, 1] = torch.expm1(raw[:, 1]).clamp_min(0.0)
    return raw[0, :, :, :NATIVE_WIDTH].numpy().astype(np.float32)


def _frame_variants(
    frame: Any,
    model: torch.nn.Module,
    device: torch.device,
    *,
    range_scale_m: float,
    intensity_transform: str,
) -> dict[str, tuple[np.ndarray, np.ndarray, dict[str, int]]]:
    """Infer once and construct A/B/C raw signals on the same 32-row grid."""
    native = top_first_return(frame)
    range_64, intensity_64, mask_64 = extract_range_intensity(native)
    range_32, intensity_32, mask_32, ring_ids = select_even_rings(
        range_64, intensity_64, mask_64
    )
    if not np.array_equal(ring_ids, RING_IDS_32):
        raise ValueError(f"unexpected 32-ring IDs: {ring_ids.tolist()}")
    x, target, target_mask, observed_raw, observed_mask = _model_arrays(
        range_32, intensity_32, mask_32,
        range_scale_m=range_scale_m,
        intensity_transform=intensity_transform,
    )
    with torch.inference_mode():
        prediction, _ = model(x.to(device), target.to(device), target_mask.to(device))
    prediction_raw = _restore_prediction(
        prediction,
        range_scale_m=range_scale_m,
        intensity_transform=intensity_transform,
    )

    signal_a = np.zeros((2, 32, NATIVE_WIDTH), dtype=np.float32)
    valid_a = np.zeros((32, NATIVE_WIDTH), dtype=bool)
    observed_native_mask = np.asarray(observed_mask[:, :NATIVE_WIDTH], dtype=bool)
    observed_valid = (
        observed_native_mask
        & np.isfinite(observed_raw[0]) & (observed_raw[0] > 0.0)
        & np.isfinite(observed_raw[1])
    )
    signal_a[:, 0::2] = observed_raw
    valid_a[0::2] = observed_valid
    signal_a[:, ~valid_a] = 0.0

    signal_b, valid_b, stats_b = fuse_16_to_32_observed_generated(
        observed_raw, observed_native_mask, prediction_raw
    )

    signal_c = np.stack((range_32, intensity_32), axis=0).astype(np.float32)
    valid_c = (
        np.asarray(mask_32, dtype=bool)
        & np.isfinite(signal_c[0]) & (signal_c[0] > 0.0)
        & np.isfinite(signal_c[1])
    )
    signal_c[:, ~valid_c] = 0.0

    return {
        "A": (signal_a, valid_a, {
            "observed_valid_pixels": int(observed_valid.sum()),
            "measured_valid_pixels": int(observed_valid.sum()),
        }),
        "B": (signal_b, valid_b, {**stats_b, "measured_valid_pixels": int(observed_valid.sum())}),
        "C": (signal_c, valid_c, {
            "observed_valid_pixels": int(observed_valid.sum()),
            "measured_valid_pixels": int(valid_c.sum()),
        }),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one variant's per-frame report atomically."""
    fields = (
        "frame_index", "timestamp_micros", "variant",
        "observed_valid_pixels", "measured_valid_pixels",
        "generated_valid_pixels", "generated_range_safety_removed_pixels",
        "generated_range_above_max_removed_pixels",
        "generated_invalid_pixel_pose_removed_pixels",
        "generated_negative_intensity_clamped", "points_before_static_filter",
        "points_after_static_filter", "generated_points_after_static_filter", "error",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    """Generate selected A/B/C maps from one shared frame/inference pass."""
    args = parse_args()
    if args.range_scale_m <= 0:
        raise ValueError("--range-scale-m must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be positive")
    if not args.tfrecord.is_file():
        raise FileNotFoundError(args.tfrecord)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    variants = list(dict.fromkeys(args.map_variants))
    output_dirs = {
        variant: args.output_root / VARIANT_NAMES[variant] / args.subset / args.tfrecord.stem
        for variant in variants
    }
    for variant, output_dir in output_dirs.items():
        required = (output_dir / "map_static.npz", output_dir / "render_profile.json", output_dir / "stats.json")
        if any(path.exists() for path in required) and not args.overwrite:
            raise FileExistsError(
                f"{variant} output already exists under {output_dir}; pass --overwrite explicitly"
            )
    if args.dry_run:
        print(json.dumps({
            "tfrecord": str(args.tfrecord),
            "output_dirs": {k: str(v) for k, v in output_dirs.items()},
            "dry_run": True,
        }, ensure_ascii=False, indent=2))
        return

    device = torch.device(args.device)
    _progress("cuda_check_begin", requested_device=str(device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    _progress("cuda_check_done", device=str(device),
              gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None)
    _progress("model_build_begin")
    model = build_waymo_16_to_32_model(args.tulip_root)
    _progress("model_build_done")
    model = model.to(device)
    _progress("model_to_device_done")
    _progress("checkpoint_load_begin", checkpoint=str(args.checkpoint),
              size_mb=round(args.checkpoint.stat().st_size/1024**2, 1))
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("checkpoint must contain a model state_dict")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    _progress("checkpoint_load_done")

    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2

    xyz_parts = {variant: [] for variant in variants}
    intensity_parts = {variant: [] for variant in variants}
    source_generated_parts = {variant: [] for variant in variants}
    per_frame = {variant: [] for variant in variants}
    failures = {variant: [] for variant in variants}
    _progress("tfrecord_open_begin", tfrecord=str(args.tfrecord))
    dataset = tf.data.TFRecordDataset(str(args.tfrecord), compression_type="")
    _progress("tfrecord_wait_first_record")
    for frame_index, record in enumerate(dataset):
        if args.max_frames is not None and frame_index >= args.max_frames:
            break
        frame = dataset_pb2.Frame()
        frame.ParseFromString(record.numpy())
        timestamp = int(frame.timestamp_micros)
        try:
            if frame_index == 0: _progress("first_frame_prediction_begin")
            frame_variants = _frame_variants(
                frame, model, device,
                range_scale_m=args.range_scale_m,
                intensity_transform=args.intensity_transform,
            )
            if frame_index == 0: _progress("first_frame_prediction_done_geometry_begin")
            geometry = build_top_geometry(
                frame, _top_pose(frame), RING_IDS_32, native_width=NATIVE_WIDTH
            )
            if frame_index == 0: _progress("first_frame_geometry_done")
            pose_valid = valid_top_pixel_pose_mask(geometry.pixel_pose[0].numpy())
            for variant in variants:
                signal, valid, variant_stats = frame_variants[variant]
                signal = signal.copy()
                valid = valid.copy()
                row: dict[str, Any] = {
                    "frame_index": frame_index, "timestamp_micros": timestamp,
                    "variant": variant, "error": "", **variant_stats,
                }
                if variant == "B":
                    invalid_generated_pose = valid[1::2] & ~pose_valid[1::2]
                    row["generated_invalid_pixel_pose_removed_pixels"] = int(
                        invalid_generated_pose.sum()
                    )
                    valid[1::2] &= pose_valid[1::2]
                    signal[:, ~valid] = 0.0
                source_generated_grid = np.zeros_like(valid, dtype=bool)
                if variant == "B":
                    source_generated_grid[1::2] = valid[1::2]
                points_vehicle, intensity = paired_vehicle_points(
                    signal[0], signal[1], geometry, valid
                )
                source_is_generated = source_generated_grid[valid]
                if source_is_generated.shape[0] != points_vehicle.shape[0]:
                    raise RuntimeError("point provenance does not align with projected points")
                keep = static_point_mask(points_vehicle, frame.laser_labels)
                points_world = vehicle_to_world(
                    points_vehicle[keep], frame.pose.transform
                )
                xyz_parts[variant].append(points_world)
                intensity_parts[variant].append(intensity[keep].astype(np.float32))
                source_generated_parts[variant].append(source_is_generated[keep])
                row["points_before_static_filter"] = int(points_vehicle.shape[0])
                row["points_after_static_filter"] = int(points_world.shape[0])
                row["generated_points_after_static_filter"] = int(source_is_generated[keep].sum())
                per_frame[variant].append(row)
            print(json.dumps({
                "processed": frame_index + 1, "frame_index": frame_index,
                "points": {v: int(xyz_parts[v][-1].shape[0]) for v in variants},
            }), flush=True)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            for variant in variants:
                row = {
                    "frame_index": frame_index, "timestamp_micros": timestamp,
                    "variant": variant, "error": error,
                }
                per_frame[variant].append(row)
                failures[variant].append(row.copy())
            print(json.dumps({"failed_frame": frame_index, "error": error}), flush=True)

    summaries: dict[str, Any] = {}
    variant_descriptions = {
        "A": "measured native rings 0,4,...,60 on even 32-grid rows",
        "B": "measured even rows plus safe TULIP-generated odd rows",
        "C": "measured GT native rings 0,2,...,62",
    }
    for variant in variants:
        output_dir = output_dirs[variant]
        if not xyz_parts[variant]:
            _write_csv(output_dir / "per_frame.csv", per_frame[variant])
            _atomic_json(output_dir / "failures.json", {"failures": failures[variant]})
            raise RuntimeError(f"no static-map points were produced for variant {variant}")
        xyz = np.concatenate(xyz_parts[variant], axis=0).astype(np.float32)
        intensity = np.concatenate(intensity_parts[variant], axis=0).astype(np.float32)
        source_is_generated = np.concatenate(source_generated_parts[variant], axis=0).astype(bool)
        _atomic_npz(
            output_dir / "map_static.npz",
            xyz=xyz,
            intensity=intensity,
            source_is_generated=source_is_generated,
        )
        _write_csv(output_dir / "per_frame.csv", per_frame[variant])
        _atomic_json(output_dir / "failures.json", {"failures": failures[variant]})
        profile = {
            "task": "Waymo TOP LiDAR 16-to-32 A/B/C static map",
            "variant": variant,
            "variant_description": variant_descriptions[variant],
            "source_tfrecord": str(args.tfrecord),
            "subset": args.subset,
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "range_scale_m": float(args.range_scale_m),
            "intensity_transform": args.intensity_transform,
            "ring_ids_16_observed": list(range(0, 64, 4)),
            "ring_ids_32_output": RING_IDS_32.tolist(),
            "generated_range_safety_filter": {
                "applied": variant == "B",
                "condition_m": f"0 < range <= {MAX_GENERATED_RANGE_M:g}",
                "action": "remove (do not clip)",
                "observed_rows_affected": False,
            },
            "dynamic_filter": "vehicle/pedestrian/cyclist boxes removed; signs retained; existing roof box applied",
            "coordinate_frame": "Waymo world frame",
        }
        rows = per_frame[variant]
        stats = {
            "variant": variant,
            "processed_frames": len(rows),
            "failed_frames": len(failures[variant]),
            "num_points": int(xyz.shape[0]),
            "generated_points": int(source_is_generated.sum()),
            "generated_range_safety_removed_pixels": int(sum(
                row.get("generated_range_safety_removed_pixels", 0) for row in rows
            )),
            "generated_range_above_max_removed_pixels": int(sum(
                row.get("generated_range_above_max_removed_pixels", 0) for row in rows
            )),
            "generated_invalid_pixel_pose_removed_pixels": int(sum(
                row.get("generated_invalid_pixel_pose_removed_pixels", 0) for row in rows
            )),
            "intensity_min": float(np.min(intensity)),
            "intensity_max": float(np.max(intensity)),
            "xyz_min": xyz.min(axis=0).astype(float).tolist(),
            "xyz_max": xyz.max(axis=0).astype(float).tolist(),
        }
        _atomic_json(output_dir / "render_profile.json", profile)
        _atomic_json(output_dir / "stats.json", stats)
        summaries[variant] = {"output_dir": str(output_dir), **stats}
    print(json.dumps({"variants": summaries}, ensure_ascii=False, indent=2))
    if any(failures[variant] for variant in variants):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
