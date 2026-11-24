#!/usr/bin/env python3
"""Convert one shifted-RI pose into a Shimizu-compatible virtual submap.

The shifted-RI metadata stores camera<-world.  Shimizu's fixed-SE(3) anchor
renderer expects vehicle->world, so this adapter reconstructs that pose using
the Waymo FRONT camera calibration and writes a regular ``submaps.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecord", type=Path, required=True)
    parser.add_argument("--shifted-ri-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--submap-id", type=int, default=1)
    parser.add_argument("--submap-radius-m", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def matrix(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(4, 4)


def load_frame(path: Path, frame_index: int):
    for index, record in enumerate(tf.data.TFRecordDataset(str(path), compression_type="")):
        if index == frame_index:
            frame = dataset_pb2.Frame()
            frame.ParseFromString(record.numpy())
            return frame
    raise IndexError(f"frame {frame_index} not found")


def yaw_deg(transform: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(transform[1, 0], transform[0, 0])))


def main() -> None:
    args = parse_args()
    shifted = json.loads(args.shifted_ri_json.read_text(encoding="utf-8"))
    anchor_frame_index = int(shifted["anchor_frame_index"])
    frame = load_frame(args.tfrecord, anchor_frame_index)
    calibration = next(
        c for c in frame.context.camera_calibrations
        if c.name == dataset_pb2.CameraName.FRONT
    )

    # T_wc = camera<-world, T_vc = vehicle<-camera.
    # T_wc = inv(T_vc) @ inv(T_wv), therefore
    # T_wv = inv(T_wc) @ inv(T_vc).
    world_to_camera = matrix(shifted["world_to_cam"])
    vehicle_from_camera = matrix(calibration.extrinsic.transform)
    vehicle_to_world = np.linalg.inv(world_to_camera) @ np.linalg.inv(vehicle_from_camera)

    segment = args.tfrecord.stem
    output_dir = args.output_root / args.subset / segment
    output_path = output_dir / "submaps.json"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    delta = shifted.get("delta", {})
    submap = {
        "submap_id": int(args.submap_id),
        "anchor_frame_index": anchor_frame_index,
        "anchor_pose_xyyaw": {
            "x": float(vehicle_to_world[0, 3]),
            "y": float(vehicle_to_world[1, 3]),
            "yaw_deg": yaw_deg(vehicle_to_world),
        },
        "anchor_pose_T_wv": vehicle_to_world.reshape(-1).tolist(),
        "member_frame_indices": [anchor_frame_index],
        "neighbor_submap_ids": [],
        "virtual_view": True,
        "source_shifted_ri_json": str(args.shifted_ri_json),
        "delta": delta,
    }
    document = {
        "meta": {
            "subset": args.subset,
            "segment_id": segment,
            "num_frames": 1,
            "num_submaps": 1,
            "spacing_m": None,
            "submap_radius_m": float(args.submap_radius_m),
            "anchor_mode": "virtual_shifted_ri_pose",
        },
        "submaps": [submap],
    }
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output_path), "submap": submap}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
