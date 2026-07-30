"""Planning and validation helpers for manifest-driven Waymo export."""
from __future__ import annotations
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REQUIRED_COLUMNS = (
    "q_subset", "q_seg", "q_frame_index", "frame_timestamp_micros",
    "source_tfrecord", "tfrecord_exists",
)

@dataclass(frozen=True)
class ExportItem:
    subset: str
    segment_id: str
    frame_index: int
    timestamp_micros: int
    tfrecord: Path

    @property
    def key(self) -> tuple[str, str, int]:
        return self.subset, self.segment_id, self.frame_index

def normalize_context_name(value: str) -> str:
    result = value.strip()
    if result.startswith("segment-"):
        result = result[len("segment-"):]
    suffix = "_with_camera_labels"
    if result.endswith(suffix):
        result = result[:-len(suffix)]
    return result

def context_matches(segment_id: str, context_name: str) -> bool:
    return normalize_context_name(segment_id) == normalize_context_name(context_name)

def load_export_items(manifest: Path, max_frames: int | None = None
                      ) -> list[ExportItem]:
    if max_frames is not None and max_frames < 0:
        raise ValueError("max_frames must be non-negative")
    items = []
    seen = set()
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in REQUIRED_COLUMNS
                   if name not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"manifest is missing columns: {missing}")
        for line, row in enumerate(reader, 2):
            if max_frames is not None and len(items) >= max_frames:
                break
            try:
                item = ExportItem(
                    subset=row["q_subset"].strip(),
                    segment_id=row["q_seg"].strip(),
                    frame_index=int(row["q_frame_index"]),
                    timestamp_micros=int(row["frame_timestamp_micros"]),
                    tfrecord=Path(row["source_tfrecord"]),
                )
            except ValueError as exc:
                raise ValueError(f"line {line}: invalid numeric identity") from exc
            if not item.subset or not item.segment_id:
                raise ValueError(f"line {line}: empty subset or segment")
            if item.frame_index < 0 or item.timestamp_micros < 0:
                raise ValueError(f"line {line}: negative frame identity")
            if row["tfrecord_exists"].strip().lower() != "true":
                raise FileNotFoundError(f"line {line}: TFRecord marked missing")
            if not item.tfrecord.is_file():
                raise FileNotFoundError(item.tfrecord)
            if item.key in seen:
                raise ValueError(f"line {line}: duplicate frame key {item.key}")
            seen.add(item.key)
            items.append(item)
    return items

def frame_output_dir(root: Path, item: ExportItem) -> Path:
    return root / item.subset / item.segment_id / f"{item.frame_index:06d}"

def output_paths(root: Path, item: ExportItem) -> tuple[Path, Path]:
    directory = frame_output_dir(root, item)
    return directory / "frame_64_32.npz", directory / "metadata.json"

def completed_output(root: Path, item: ExportItem) -> bool:
    npz_path, metadata_path = output_paths(root, item)
    if not npz_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return (
            metadata["segment_id"] == item.segment_id
            and int(metadata["source_frame_index"]) == item.frame_index
            and int(metadata["timestamp_micros"]) == item.timestamp_micros
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False

def group_by_tfrecord(items: Sequence[ExportItem]
                      ) -> dict[Path, list[ExportItem]]:
    result: dict[Path, list[ExportItem]] = {}
    for item in items:
        result.setdefault(item.tfrecord, []).append(item)
    for group in result.values():
        group.sort(key=lambda value: value.frame_index)
    return result

def validate_frame_identity(item: ExportItem, context_name: str,
                            timestamp_micros: int) -> None:
    if not context_matches(item.segment_id, context_name):
        raise ValueError(
            f"context mismatch: {item.segment_id} != {context_name}")
    if item.timestamp_micros != timestamp_micros:
        raise ValueError(
            f"timestamp mismatch for {item.key}: "
            f"{item.timestamp_micros} != {timestamp_micros}")

def select_pending(items: Sequence[ExportItem], output_root: Path,
                   resume: bool) -> tuple[list[ExportItem], int]:
    if not resume:
        return list(items), 0
    pending = []
    skipped = 0
    for item in items:
        if completed_output(output_root, item):
            skipped += 1
        else:
            pending.append(item)
    return pending, skipped
