"""Planning helpers for leakage-safe training frame extraction."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

ROLES = ("train", "validation", "test")
SEGMENT_COLUMNS = ("dataset_role", "source_split", "segment_id", "source_tfrecord")
FRAME_COLUMNS = (
    "dataset_role", "q_subset", "q_seg", "q_frame_index",
    "frame_timestamp_micros", "source_tfrecord", "tfrecord_exists",
)


@dataclass(frozen=True)
class TrainingSegment:
    dataset_role: str
    source_split: str
    segment_id: str
    tfrecord: Path


def load_training_segments(
    manifest: Path,
    limits: Mapping[str, int],
    excluded_segments: set[str],
) -> list[TrainingSegment]:
    for role in ROLES:
        if limits.get(role, 0) < 0:
            raise ValueError(f"{role} segment limit must be non-negative")
    selected: list[TrainingSegment] = []
    counts = {role: 0 for role in ROLES}
    seen: dict[str, str] = {}
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in SEGMENT_COLUMNS
                   if name not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"segment manifest is missing columns: {missing}")
        for line, row in enumerate(reader, 2):
            role = row["dataset_role"].strip()
            if role not in ROLES:
                raise ValueError(f"line {line}: invalid dataset_role {role!r}")
            segment = row["segment_id"].strip()
            source_split = row["source_split"].strip()
            tfrecord = Path(row["source_tfrecord"])
            if not segment or not source_split:
                raise ValueError(f"line {line}: empty segment identity")
            if segment in excluded_segments:
                raise ValueError(f"line {line}: evaluation leakage: {segment}")
            old_role = seen.setdefault(segment, role)
            if old_role != role:
                raise ValueError(
                    f"line {line}: segment appears in multiple roles: {segment}")
            if not tfrecord.is_file():
                raise FileNotFoundError(tfrecord)
            if counts[role] < limits.get(role, 0):
                selected.append(TrainingSegment(
                    role, source_split, segment, tfrecord))
                counts[role] += 1
    return selected


def selected_frame_indices(frames_per_segment: int, frame_stride: int,
                           start_frame: int = 0) -> tuple[int, ...]:
    if frames_per_segment < 1:
        raise ValueError("frames_per_segment must be positive")
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    return tuple(start_frame + i * frame_stride for i in range(frames_per_segment))


def training_output_paths(output_root: Path, segment: TrainingSegment,
                          frame_index: int) -> tuple[Path, Path]:
    directory = (
        output_root / segment.dataset_role / segment.segment_id
        / f"{frame_index:06d}"
    )
    return directory / "frame_64_32.npz", directory / "metadata.json"


def completed_training_output(output_root: Path, segment: TrainingSegment,
                              frame_index: int) -> bool:
    npz_path, metadata_path = training_output_paths(
        output_root, segment, frame_index)
    if not npz_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return (
            metadata["segment_id"] == segment.segment_id
            and metadata["dataset_role"] == segment.dataset_role
            and int(metadata["source_frame_index"]) == frame_index
            and metadata["source_subset"] == segment.source_split
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def pending_frame_indices(
    output_root: Path,
    segment: TrainingSegment,
    indices: Sequence[int],
    resume: bool,
) -> tuple[list[int], int]:
    if not resume:
        return list(indices), 0
    pending = [
        index for index in indices
        if not completed_training_output(output_root, segment, index)
    ]
    return pending, len(indices) - len(pending)


def frame_manifest_row(segment: TrainingSegment, frame_index: int,
                       timestamp_micros: int, npz_path: Path,
                       metadata_path: Path) -> dict[str, str]:
    return {
        "dataset_role": segment.dataset_role,
        "q_subset": segment.source_split,
        "q_seg": segment.segment_id,
        "q_frame_index": str(frame_index),
        "frame_timestamp_micros": str(timestamp_micros),
        "source_tfrecord": str(segment.tfrecord),
        "tfrecord_exists": str(segment.tfrecord.is_file()).lower(),
        "frame_npz": str(npz_path),
        "frame_metadata": str(metadata_path),
    }


def write_frame_manifest(rows: Sequence[Mapping[str, str]], output: Path) -> None:
    fields = FRAME_COLUMNS + ("frame_npz", "frame_metadata")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
