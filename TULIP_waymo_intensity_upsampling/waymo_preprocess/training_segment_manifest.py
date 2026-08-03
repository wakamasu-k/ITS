"""Build deterministic, leakage-safe Waymo segment splits for TULIP training."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Iterable, Mapping, Sequence

OUTPUT_COLUMNS = ("dataset_role", "source_split", "segment_id", "source_tfrecord")
ROLES = ("train", "validation", "test")


def segment_id_from_tfrecord(path: Path) -> str:
    suffix = ".tfrecord"
    if not path.name.endswith(suffix):
        raise ValueError(f"not a TFRecord path: {path}")
    return path.name[:-len(suffix)]


def read_excluded_segments(frame_manifest: Path) -> set[str]:
    with frame_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "q_seg" not in (reader.fieldnames or ()):
            raise ValueError("evaluation manifest is missing q_seg")
        segments = {row["q_seg"].strip() for row in reader if row["q_seg"].strip()}
    if not segments:
        raise ValueError("evaluation manifest contains no segments")
    return segments


def discover_tfrecords(waymo_root: Path,
                       source_splits: Sequence[str] = ("training",)
                       ) -> list[tuple[str, str, Path]]:
    found: list[tuple[str, str, Path]] = []
    seen: dict[str, Path] = {}
    for source_split in source_splits:
        for path in sorted((waymo_root / source_split).glob("*.tfrecord")):
            segment = segment_id_from_tfrecord(path)
            if segment in seen:
                raise ValueError(
                    f"duplicate segment {segment}: {seen[segment]} and {path}")
            seen[segment] = path
            found.append((source_split, segment, path))
    return found


def _score(seed: int, segment: str) -> int:
    digest = hashlib.sha256(f"{seed}:{segment}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def split_segments(records: Iterable[tuple[str, str, Path]], *,
                   excluded_segments: set[str], seed: int = 20260730,
                   validation_fraction: float = 0.1,
                   test_fraction: float = 0.1,
                   max_segments: int | None = None
                   ) -> tuple[list[dict[str, str]], dict[str, int]]:
    if not 0 <= validation_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("split fractions must be in [0, 1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be < 1")
    if max_segments is not None and max_segments < 1:
        raise ValueError("max_segments must be positive")

    records = list(records)
    eligible = [record for record in records if record[1] not in excluded_segments]
    excluded_count = len(records) - len(eligible)
    eligible.sort(key=lambda record: (_score(seed, record[1]), record[1]))
    if max_segments is not None:
        eligible = eligible[:max_segments]

    total = len(eligible)
    validation_count = int(total * validation_fraction)
    test_count = int(total * test_fraction)
    if total >= 3 and validation_fraction > 0:
        validation_count = max(1, validation_count)
    if total >= 3 and test_fraction > 0:
        test_count = max(1, test_count)
    while validation_count + test_count >= total and total > 0:
        if test_count >= validation_count and test_count > 0:
            test_count -= 1
        elif validation_count > 0:
            validation_count -= 1

    roles = (["validation"] * validation_count
             + ["test"] * test_count
             + ["train"] * (total - validation_count - test_count))
    rows = [{
        "dataset_role": role,
        "source_split": source_split,
        "segment_id": segment,
        "source_tfrecord": str(path),
    } for role, (source_split, segment, path) in zip(roles, eligible)]
    rows.sort(key=lambda row: (ROLES.index(row["dataset_role"]), row["segment_id"]))
    summary = {
        "discovered_segments": len(records),
        "excluded_evaluation_segments": excluded_count,
        "selected_segments": len(rows),
        "train_segments": sum(row["dataset_role"] == "train" for row in rows),
        "validation_segments": sum(
            row["dataset_role"] == "validation" for row in rows),
        "test_segments": sum(row["dataset_role"] == "test" for row in rows),
    }
    validate_no_overlap(rows, excluded_segments)
    return rows, summary


def validate_no_overlap(rows: Sequence[Mapping[str, str]],
                        excluded_segments: set[str]) -> None:
    role_by_segment: dict[str, str] = {}
    for row in rows:
        segment = row["segment_id"]
        role = row["dataset_role"]
        if segment in excluded_segments:
            raise ValueError(f"evaluation leakage detected: {segment}")
        old_role = role_by_segment.setdefault(segment, role)
        if old_role != role:
            raise ValueError(f"segment appears in multiple roles: {segment}")


def write_segment_rows(rows: Sequence[Mapping[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
