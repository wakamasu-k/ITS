"""Role-preserving 32-to-64 pair planning for extracted training frames."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

ROLES = ("train", "validation", "test")
REQUIRED_COLUMNS = (
    "dataset_role", "q_subset", "q_seg", "q_frame_index",
    "frame_timestamp_micros", "source_tfrecord", "frame_npz",
)
INDEX_COLUMNS = (
    "dataset_role", "q_subset", "q_seg", "q_frame_index",
    "timestamp_micros", "pair_npz", "pair_json", "input_height",
    "target_height", "width", "channels", "observed_target_rows",
    "generated_target_rows",
)


@dataclass(frozen=True)
class TrainingFrame:
    dataset_role: str
    source_subset: str
    segment_id: str
    frame_index: int
    timestamp_micros: int
    source_tfrecord: Path
    frame_npz: Path

    @property
    def key(self) -> tuple[str, str, int]:
        return self.dataset_role, self.segment_id, self.frame_index


def load_training_frames(manifest: Path,
                         max_frames: int | None = None) -> list[TrainingFrame]:
    if max_frames is not None and max_frames < 0:
        raise ValueError("max_frames must be non-negative")
    frames = []
    seen = set()
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in REQUIRED_COLUMNS
                   if name not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"frame index is missing columns: {missing}")
        for line, row in enumerate(reader, 2):
            if max_frames is not None and len(frames) >= max_frames:
                break
            role = row["dataset_role"].strip()
            if role not in ROLES:
                raise ValueError(f"line {line}: invalid dataset_role {role!r}")
            try:
                frame = TrainingFrame(
                    dataset_role=role,
                    source_subset=row["q_subset"].strip(),
                    segment_id=row["q_seg"].strip(),
                    frame_index=int(row["q_frame_index"]),
                    timestamp_micros=int(row["frame_timestamp_micros"]),
                    source_tfrecord=Path(row["source_tfrecord"]),
                    frame_npz=Path(row["frame_npz"]),
                )
            except ValueError as exc:
                raise ValueError(f"line {line}: invalid numeric identity") from exc
            if not frame.source_subset or not frame.segment_id:
                raise ValueError(f"line {line}: empty frame identity")
            if frame.frame_index < 0 or frame.timestamp_micros < 0:
                raise ValueError(f"line {line}: negative frame identity")
            if frame.key in seen:
                raise ValueError(f"line {line}: duplicate frame key {frame.key}")
            if not frame.source_tfrecord.is_file():
                raise FileNotFoundError(frame.source_tfrecord)
            if not frame.frame_npz.is_file():
                raise FileNotFoundError(frame.frame_npz)
            seen.add(frame.key)
            frames.append(frame)
    return frames


def pair_paths(frame: TrainingFrame) -> tuple[Path, Path]:
    return (
        frame.frame_npz.with_name("tulip_pair_32_to_64.npz"),
        frame.frame_npz.with_name("tulip_pair_32_to_64.json"),
    )


def completed_pair(frame: TrainingFrame) -> bool:
    npz_path, json_path = pair_paths(frame)
    if not npz_path.is_file() or not json_path.is_file():
        return False
    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        return (
            metadata["dataset_role"] == frame.dataset_role
            and metadata["subset"] == frame.source_subset
            and metadata["segment_id"] == frame.segment_id
            and int(metadata["frame_index"]) == frame.frame_index
            and int(metadata["timestamp_micros"]) == frame.timestamp_micros
            and metadata["task"]
            == "Waymo TOP LiDAR 32-to-64 range and intensity upsampling"
            and bool(metadata["exact_subset_verified"])
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def index_row(frame: TrainingFrame, metadata: Mapping[str, object]
              ) -> dict[str, str]:
    npz_path, json_path = pair_paths(frame)
    input_shape = metadata["input_shape"]
    target_shape = metadata["target_shape"]
    if not isinstance(input_shape, list) or not isinstance(target_shape, list):
        raise ValueError("pair metadata shapes must be lists")
    return {
        "dataset_role": frame.dataset_role,
        "q_subset": frame.source_subset,
        "q_seg": frame.segment_id,
        "q_frame_index": str(frame.frame_index),
        "timestamp_micros": str(frame.timestamp_micros),
        "pair_npz": str(npz_path),
        "pair_json": str(json_path),
        "input_height": str(input_shape[0]),
        "target_height": str(target_shape[0]),
        "width": str(input_shape[1]),
        "channels": str(input_shape[2]),
        "observed_target_rows": "0:64:2",
        "generated_target_rows": "1:64:2",
    }


def write_rows(rows: Sequence[Mapping[str, str]], output: Path,
               fields: Sequence[str] = INDEX_COLUMNS) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
