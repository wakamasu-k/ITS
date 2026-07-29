#!/usr/bin/env python3
"""Read-only audit of the source Waymo and Shimizu 32-line datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def segment_id_from_tfrecord(path: Path) -> str:
    suffix = ".tfrecord"
    return path.name[: -len(suffix)] if path.name.endswith(suffix) else path.stem


def child_directory_names(path: Path) -> set[str]:
    if not path.is_dir():
        return set()
    return {item.name for item in path.iterdir() if item.is_dir()}


def inspect_map(map_path: Path) -> dict[str, Any]:
    if not map_path.is_file():
        return {"exists": False, "path": str(map_path)}
    with np.load(map_path, allow_pickle=False, mmap_mode="r") as data:
        arrays = {
            key: {
                "shape": [int(value) for value in data[key].shape],
                "dtype": str(data[key].dtype),
            }
            for key in data.files
        }
    return {"exists": True, "path": str(map_path), "arrays": arrays}


def load_json_if_present(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def audit(waymo_root: Path, line32_root: Path) -> dict[str, Any]:
    split_counts: dict[str, int] = {}
    split_segments: dict[str, set[str]] = {}
    for split in ("training", "validation", "testing"):
        paths = sorted((waymo_root / split).glob("*.tfrecord"))
        split_counts[split] = len(paths)
        split_segments[split] = {segment_id_from_tfrecord(path) for path in paths}

    maps_training = child_directory_names(
        line32_root / "datasets" / "maps" / "training"
    )
    cameras_training = child_directory_names(
        line32_root / "datasets" / "cam_gray" / "training"
    )
    common_training = sorted(
        split_segments["training"] & maps_training & cameras_training
    )

    representative = common_training[0] if common_training else None
    representative_report: dict[str, Any] | None = None
    if representative is not None:
        map_dir = line32_root / "datasets" / "maps" / "training" / representative
        camera_dir = (
            line32_root
            / "datasets"
            / "cam_gray"
            / "training"
            / representative
            / "FRONT"
        )
        representative_report = {
            "segment_id": representative,
            "map": inspect_map(map_dir / "map_static.npz"),
            "stats": load_json_if_present(map_dir / "stats.json"),
            "render_profile": load_json_if_present(
                map_dir / "render_profile.json"
            ),
            "front_camera_png_count": len(list(camera_dir.glob("*_cam.png"))),
            "front_camera_json_count": len(list(camera_dir.glob("*_cam.json"))),
        }

    return {
        "waymo_root": str(waymo_root.resolve()),
        "line32_root": str(line32_root.resolve()),
        "tfrecord_counts": split_counts,
        "line32_scene_counts": {
            "maps_training": len(maps_training),
            "cam_gray_training": len(cameras_training),
            "common_with_source_training": len(common_training),
        },
        "common_training_examples": common_training[:10],
        "representative": representative_report,
        "interpretation": {
            "line32_root_is_tulip_tensor_dataset": False,
            "line32_root_role": (
                "Shimizu localization pipeline outputs: static maps, camera "
                "images, submaps, retrieval/LoFTR pairs and evaluation GT."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--waymo-root", required=True, type=Path)
    parser.add_argument("--line32-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(args.waymo_root, args.line32_root)
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
        print(f"saved_report={args.output}")


if __name__ == "__main__":
    main()
