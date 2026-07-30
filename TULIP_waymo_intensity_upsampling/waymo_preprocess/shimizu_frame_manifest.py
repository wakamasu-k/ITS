"""Create a frame-level correspondence table from Shimizu's GT manifest."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

REQUIRED_COLUMNS = ("q_subset", "q_seg", "q_frame_index", "q_meta_json")
OUTPUT_COLUMNS = (
    "q_subset", "q_seg", "q_frame_index", "frame_timestamp_micros",
    "q_meta_json", "source_tfrecord", "tfrecord_exists", "source_manifest_rows",
)

def resolve_camera_json(raw_path: str, line32_root: Path,
                        path_maps: Sequence[tuple[Path, Path]] = ()) -> Path:
    candidate = Path(raw_path)
    if candidate.is_file():
        return candidate
    raw = candidate.as_posix()
    for old, new in path_maps:
        prefix = old.as_posix().rstrip("/")
        if raw == prefix or raw.startswith(prefix + "/"):
            mapped = new / raw[len(prefix):].lstrip("/")
            if mapped.is_file():
                return mapped
    marker = "/cam_gray/"
    if marker in raw:
        mapped = line32_root / "datasets" / "cam_gray" / raw.split(marker, 1)[1]
        if mapped.is_file():
            return mapped
    raise FileNotFoundError(f"cannot resolve q_meta_json: {raw_path}")

def parse_path_maps(values: Sequence[str]) -> list[tuple[Path, Path]]:
    result = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"path map must be OLD=NEW: {value!r}")
        old, new = value.split("=", 1)
        if not old or not new:
            raise ValueError(f"path map must be OLD=NEW: {value!r}")
        result.append((Path(old), Path(new)))
    return result

def read_camera_identity(path: Path) -> tuple[int, int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    try:
        frame_index = int(value["frame_index"])
        timestamp = int(value["frame_timestamp_micros"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid camera identity in {path}: {exc}") from exc
    if frame_index < 0 or timestamp < 0:
        raise ValueError(f"invalid camera identity in {path}")
    return frame_index, timestamp

def build_frame_rows(manifest: Path, line32_root: Path, waymo_root: Path, *,
                     path_maps: Sequence[tuple[Path, Path]] = ()) -> list[dict[str, str]]:
    frames: dict[tuple[str, str, int], dict[str, str]] = {}
    counts: dict[tuple[str, str, int], int] = {}
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"manifest is missing columns: {missing}")
        for line, source in enumerate(reader, 2):
            subset = source["q_subset"].strip()
            segment = source["q_seg"].strip()
            try:
                frame_index = int(source["q_frame_index"])
            except ValueError as exc:
                raise ValueError(f"line {line}: invalid q_frame_index") from exc
            if not subset or not segment or frame_index < 0:
                raise ValueError(f"line {line}: invalid query identity")
            metadata = resolve_camera_json(source["q_meta_json"], line32_root, path_maps)
            json_index, timestamp = read_camera_identity(metadata)
            if json_index != frame_index:
                raise ValueError(f"line {line}: frame index mismatch: "
                                 f"{frame_index} != {json_index}")
            key = (subset, segment, frame_index)
            tfrecord = waymo_root / subset / f"{segment}.tfrecord"
            row = {"q_subset": subset, "q_seg": segment,
                   "q_frame_index": str(frame_index),
                   "frame_timestamp_micros": str(timestamp),
                   "q_meta_json": str(metadata),
                   "source_tfrecord": str(tfrecord),
                   "tfrecord_exists": str(tfrecord.is_file()).lower(),
                   "source_manifest_rows": "1"}
            if key in frames:
                old = frames[key]
                if old["frame_timestamp_micros"] != row["frame_timestamp_micros"]:
                    raise ValueError(f"timestamp conflict for {key}")
                counts[key] += 1
            else:
                frames[key] = row
                counts[key] = 1
    rows = []
    for key in sorted(frames):
        row = frames[key]
        row["source_manifest_rows"] = str(counts[key])
        rows.append(row)
    return rows

def write_frame_rows(rows: Sequence[Mapping[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
