"""Convert pair-level handover manifests to one row per Waymo frame."""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence

FIELDS = ("segment_id", "frame_index", "timestamp_us", "source_split",
          "source_tfrecord", "source_manifest_rows", "metadata_json", "status")

@dataclass(frozen=True)
class FrameIdentity:
    segment_id: str
    frame_index: int
    timestamp_us: int

def segment_id_from_tfrecord(path: Path) -> str:
    return path.name[:-9] if path.name.endswith(".tfrecord") else path.stem

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

def resolve_metadata_path(raw_path: str, path_maps: Sequence[tuple[Path, Path]],
                          line32_root: Path | None) -> Path:
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
    marker = "/datasets/"
    if line32_root is not None and marker in raw:
        mapped = line32_root / "datasets" / raw.split(marker, 1)[1]
        if mapped.is_file():
            return mapped
    raise FileNotFoundError(f"cannot resolve metadata JSON: {raw_path}")

def load_identity(path: Path) -> FrameIdentity:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    try:
        value = FrameIdentity(str(metadata["segment_id"]),
                              int(metadata["frame_index"]),
                              int(metadata["timestamp_us"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid frame identity in {path}: {exc}") from exc
    if not value.segment_id or value.frame_index < 0 or value.timestamp_us < 0:
        raise ValueError(f"invalid frame identity in {path}")
    return value

def index_tfrecords(root: Path) -> dict[str, tuple[str, Path]]:
    result = {}
    for split in ("training", "validation", "testing"):
        for path in sorted((root / split).glob("*.tfrecord")):
            segment = segment_id_from_tfrecord(path)
            if segment in result:
                raise ValueError(f"duplicate source TFRecord for {segment}")
            result[segment] = (split, path)
    return result

def build_rows(manifest: Path, waymo_root: Path, *, metadata_column="cam_json",
               path_maps=(), line32_root: Path | None = None) -> list[dict[str, str]]:
    sources = index_tfrecords(waymo_root)
    frames: dict[tuple[str, int], tuple[FrameIdentity, Path, int]] = {}
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or metadata_column not in reader.fieldnames:
            raise ValueError(f"manifest needs {metadata_column!r}; found {reader.fieldnames}")
        for line, source_row in enumerate(reader, 2):
            path = resolve_metadata_path(source_row[metadata_column], path_maps,
                                         line32_root)
            identity = load_identity(path)
            listed = source_row.get("segment_id", "").strip()
            if listed and listed != identity.segment_id:
                raise ValueError(f"line {line}: segment mismatch: "
                                 f"{listed} != {identity.segment_id}")
            key = (identity.segment_id, identity.frame_index)
            if key not in frames:
                frames[key] = (identity, path, 1)
            else:
                old, old_path, count = frames[key]
                if old.timestamp_us != identity.timestamp_us:
                    raise ValueError(f"timestamp conflict for {key}: "
                                     f"{old.timestamp_us} != {identity.timestamp_us}")
                frames[key] = (old, old_path, count + 1)
    rows = []
    for identity, path, count in sorted(
            frames.values(), key=lambda item: (item[0].segment_id, item[0].frame_index)):
        source = sources.get(identity.segment_id)
        if source:
            split, tfrecord = source
            tfrecord_text = str(tfrecord)
            status = "metadata_matched"
        else:
            split, tfrecord_text, status = "", "", "missing_tfrecord"
        rows.append({"segment_id": identity.segment_id,
                     "frame_index": str(identity.frame_index),
                     "timestamp_us": str(identity.timestamp_us),
                     "source_split": split, "source_tfrecord": tfrecord_text,
                     "source_manifest_rows": str(count), "metadata_json": str(path),
                     "status": status})
    return rows

def verify_rows(rows: list[dict[str, str]],
                reader: Callable[[Path], Iterable[tuple[int, str, int]]]) -> None:
    grouped = defaultdict(list)
    for row in rows:
        if row["source_tfrecord"]:
            grouped[Path(row["source_tfrecord"])].append(row)
    for tfrecord, group in grouped.items():
        expected = {int(row["frame_index"]): row for row in group}
        remaining = set(expected)
        last = max(remaining)
        for index, context, timestamp in reader(tfrecord):
            if index in expected:
                row = expected[index]
                if context != row["segment_id"]:
                    raise ValueError(f"{tfrecord} frame {index}: context mismatch "
                                     f"{context} != {row['segment_id']}")
                if timestamp != int(row["timestamp_us"]):
                    raise ValueError(f"{tfrecord} frame {index}: timestamp mismatch "
                                     f"{timestamp} != {row['timestamp_us']}")
                row["status"] = "tfrecord_verified"
                remaining.remove(index)
            if index >= last:
                break
        if remaining:
            raise ValueError(f"{tfrecord}: missing frame indices {sorted(remaining)}")

def iter_waymo_frame_identities(path: Path) -> Iterator[tuple[int, str, int]]:
    """Import the heavy runtime only when strict verification is requested."""
    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2 as open_dataset
    for index, record in enumerate(tf.data.TFRecordDataset(str(path), compression_type="")):
        frame = open_dataset.Frame()
        frame.ParseFromString(record.numpy())
        yield index, str(frame.context.name), int(frame.timestamp_micros)

def write_csv(rows: Sequence[Mapping[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
