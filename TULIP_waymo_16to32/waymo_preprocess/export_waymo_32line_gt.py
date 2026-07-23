"""Export raw Waymo TOP LiDAR 32-line range/intensity GT arrays.

This Phase 3 exporter reuses the Phase 1 TFRecord and native Range Image
helpers.  It intentionally performs no normalization, clipping, padding, or
model/Dataset processing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
from waymo_open_dataset import dataset_pb2 as open_dataset

if __package__ in (None, ""):
    from waymo_range_image import extract_range_intensity, make_32line_gt
    from waymo_tfrecord_io import get_top_first_return, iter_frames
else:
    from waymo_preprocess.waymo_range_image import extract_range_intensity, make_32line_gt
    from waymo_preprocess.waymo_tfrecord_io import get_top_first_return, iter_frames


SCRIPT_VERSION = "phase3-raw-32line-export-v1"
LIDAR_NAME = "TOP"
RETURN_INDEX = 0
NATIVE_SHAPE = (64, 2650, 4)
RING_IDS_32 = np.arange(0, 64, 2, dtype=np.int32)
FORBIDDEN_OUTPUT_ROOT = Path("/mnt/w/32line")


class AllFramesFailedError(RuntimeError):
    """Raised when no frame was exported or skipped successfully."""


def _safe_component(value: str, field_name: str) -> str:
    """Validate a path component supplied by metadata or CLI.

    Input shape: one string.
    Output shape: one path-safe string.
    """
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"{field_name} must be a single path component: {value!r}")
    return value


def _is_within(path: Path, parent: Path) -> bool:
    """Return whether resolved path is parent or one of its descendants."""
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def validate_output_root(output_root: str | Path) -> Path:
    """Validate and resolve the output root.

    Input shape: one filesystem path.
    Output shape: one absolute ``Path``.

    Raises:
        ValueError: If output is under the read-only ``/mnt/w/32line`` tree.
    """
    resolved = Path(output_root).expanduser().resolve()
    if _is_within(resolved, FORBIDDEN_OUTPUT_ROOT):
        raise ValueError(f"Output under {FORBIDDEN_OUTPUT_ROOT} is forbidden: {resolved}")
    return resolved


def _validate_tfrecord_path(path: Path) -> Path:
    """Validate one existing ``.tfrecord`` path."""
    path = path.expanduser()
    if path.suffix != ".tfrecord":
        raise ValueError(f"Input must have .tfrecord suffix: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"TFRecord does not exist: {path}")
    return path.resolve()


def discover_tfrecords(
    *,
    input_root: str | Path | None = None,
    tfrecord: str | Path | None = None,
    tfrecord_list: str | Path | None = None,
    max_tfrecords: int | None = None,
) -> list[Path]:
    """Discover one exclusive TFRecord source in reproducible sorted order.

    Input shape: exactly one directory, file, or list-file argument.
    Output shape: sorted list of existing ``Path`` objects.
    """
    sources = [input_root is not None, tfrecord is not None, tfrecord_list is not None]
    if sum(sources) != 1:
        raise ValueError("Specify exactly one of --input-root, --tfrecord, or --tfrecord-list.")
    if max_tfrecords is not None and max_tfrecords <= 0:
        raise ValueError("max_tfrecords must be positive.")

    if tfrecord is not None:
        paths = [_validate_tfrecord_path(Path(tfrecord))]
    elif input_root is not None:
        root = Path(input_root).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"TFRecord directory does not exist: {root}")
        paths = sorted(
            path.resolve()
            for path in root.iterdir()
            if path.is_file() and path.suffix == ".tfrecord"
        )
    else:
        list_path = Path(tfrecord_list).expanduser()
        if not list_path.is_file():
            raise FileNotFoundError(f"TFRecord list does not exist: {list_path}")
        paths = []
        for line_number, raw_line in enumerate(list_path.read_text().splitlines(), 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            candidate = Path(line).expanduser()
            if not candidate.is_absolute():
                candidate = list_path.parent / candidate
            try:
                paths.append(_validate_tfrecord_path(candidate))
            except (FileNotFoundError, ValueError) as error:
                raise ValueError(f"Invalid TFRecord list entry at line {line_number}: {error}") from error
        paths = sorted(paths)

    if not paths:
        raise ValueError("No .tfrecord files were selected.")
    return paths if max_tfrecords is None else paths[:max_tfrecords]


def build_raw_32line(native_range_image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build raw packed 32-line GT from native 64-line data.

    Input shape: native float32 Range Image ``[64,2650,4]``.
    Output shapes: packed float32 ``[32,2650,2]``, bool mask ``[32,2650]``,
    and int32 ring IDs ``[32]``.

    No normalization, clipping, log transform, gamma correction, histogram
    equalization, uint8 conversion, or width padding is performed.
    """
    native = np.asarray(native_range_image, dtype=np.float32)
    if tuple(native.shape) != NATIVE_SHAPE:
        raise ValueError(f"Expected native shape {list(NATIVE_SHAPE)}, got {list(native.shape)}")
    range_64, intensity_64, mask_64 = extract_range_intensity(native)
    range_32, intensity_32, mask_32, ring_ids_32 = make_32line_gt(
        range_64, intensity_64, mask_64
    )
    packed = np.stack((range_32, intensity_32), axis=-1).astype(np.float32, copy=False)
    if packed.shape != (32, 2650, 2):
        raise ValueError(f"Unexpected packed output shape: {packed.shape}")
    if packed.dtype != np.float32 or not np.isfinite(packed).all():
        raise ValueError("Packed output must be finite float32.")
    return packed, mask_32.astype(bool, copy=False), ring_ids_32


def _finite_repeated(value: Any, expected_length: int) -> list[float] | None:
    """Convert a repeated protobuf scalar field only when exact and finite."""
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if len(values) != expected_length or not np.isfinite(values).all():
        return None
    return values


def _top_calibration(frame: Any) -> Any | None:
    """Return the TOP calibration without synthesizing missing metadata."""
    context = getattr(frame, "context", None)
    calibrations = getattr(context, "laser_calibrations", ())
    top_value = open_dataset.LaserName.TOP
    for calibration in calibrations:
        name = getattr(calibration, "name", None)
        if name == top_value or name == LIDAR_NAME:
            return calibration
        try:
            if int(name) == int(top_value):
                return calibration
        except (TypeError, ValueError):
            continue
    return None


def _frame_geometry_metadata(frame: Any) -> tuple[dict[str, Any], dict[str, str]]:
    """Extract optional TOP calibration and pose metadata with reasons."""
    reasons: dict[str, str] = {}
    calibration = _top_calibration(frame)
    extrinsic: list[float] | None = None
    beam_64: list[float] | None = None
    if calibration is None:
        reasons["top_lidar_extrinsic"] = "TOP laser calibration is unavailable."
        reasons["beam_inclinations_64"] = "TOP laser calibration is unavailable."
    else:
        extrinsic_message = getattr(calibration, "extrinsic", None)
        extrinsic = _finite_repeated(getattr(extrinsic_message, "transform", ()), 16)
        if extrinsic is None:
            reasons["top_lidar_extrinsic"] = "TOP extrinsic is unavailable or not a finite 4x4 transform."
        beam_64 = _finite_repeated(getattr(calibration, "beam_inclinations", ()), 64)
        if beam_64 is None:
            reasons["beam_inclinations_64"] = "TOP beam_inclinations is unavailable or does not contain 64 finite values."

    pose_message = getattr(frame, "pose", None)
    vehicle_pose = _finite_repeated(getattr(pose_message, "transform", ()), 16)
    if vehicle_pose is None:
        reasons["vehicle_pose"] = "Frame vehicle pose is unavailable or not a finite 4x4 transform."

    return {
        "top_lidar_extrinsic": extrinsic,
        "beam_inclinations_64": beam_64,
        "beam_inclinations_32": beam_64[::2] if beam_64 is not None else None,
        "vehicle_pose": vehicle_pose,
    }, reasons


def build_sidecar(
    *,
    frame: Any,
    frame_index: int,
    split: str,
    source_tfrecord: Path,
    packed: np.ndarray,
    mask_32: np.ndarray,
    ring_ids_32: np.ndarray,
    relative_npy_path: str,
    relative_json_path: str,
) -> dict[str, Any]:
    """Build one JSON-serializable sidecar dictionary.

    Input shapes: packed ``[32,2650,2]`` and mask ``[32,2650]``.
    Output shape: one JSON object.
    """
    context = getattr(frame, "context", None)
    context_name = str(getattr(context, "name", ""))
    _safe_component(context_name, "context_name")
    timestamp = getattr(frame, "timestamp_micros", None)
    if timestamp is None:
        raise ValueError("Frame timestamp_micros is unavailable.")
    valid_values = packed[mask_32]
    geometry, geometry_reasons = _frame_geometry_metadata(frame)
    if valid_values.size:
        range_values = valid_values[:, 0]
        intensity_values = valid_values[:, 1]
        range_min = float(np.min(range_values))
        range_max = float(np.max(range_values))
        intensity_min = float(np.min(intensity_values))
        intensity_max = float(np.max(intensity_values))
    else:
        range_min = range_max = intensity_min = intensity_max = None
    return {
        "source_tfrecord": str(source_tfrecord),
        "source_frame_index": int(frame_index),
        "context_name": context_name,
        "timestamp_micros": int(timestamp),
        "split": split,
        "native_shape": list(NATIVE_SHAPE),
        "output_shape": list(packed.shape),
        "return_index": RETURN_INDEX,
        "lidar_name": LIDAR_NAME,
        "channel_semantics": {"0": "range", "1": "intensity"},
        "ring_ids_32": ring_ids_32.tolist(),
        "valid_count": int(mask_32.sum()),
        "valid_rate": float(mask_32.mean()),
        "range_min_valid": range_min,
        "range_max_valid": range_max,
        "intensity_min_valid": intensity_min,
        "intensity_max_valid": intensity_max,
        "normalization": "none",
        "dtype": str(packed.dtype),
        "relative_npy_path": relative_npy_path,
        "relative_json_path": relative_json_path,
        **geometry,
        "metadata_unavailable_reasons": geometry_reasons,
    }


def _atomic_write_json(path: Path, value: Any, *, overwrite: bool = True) -> None:
    """Atomically write JSON beside its final destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_write_npy(path: Path, array: np.ndarray, *, overwrite: bool = False) -> None:
    """Atomically write a non-pickle NumPy array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    """Read an existing JSONL manifest or return an empty list."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid manifest JSON at line {line_number}: {path}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Manifest line {line_number} is not an object: {path}")
        records.append(record)
    return records


def _write_manifest(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Atomically rewrite a JSONL manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _failure(path: Path, frame_index: int, error: Exception) -> dict[str, Any]:
    """Build one JSON-safe failure record."""
    return {
        "source_tfrecord": str(path),
        "frame_index": int(frame_index),
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def _frame_limit(
    *,
    source_kind: str,
    max_frames: int | None,
    max_frames_per_tfrecord: int | None,
) -> int | None:
    """Validate and select the frame limit for the selected source kind."""
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive.")
    if max_frames_per_tfrecord is not None and max_frames_per_tfrecord <= 0:
        raise ValueError("max_frames_per_tfrecord must be positive.")
    if source_kind == "single":
        if max_frames_per_tfrecord is not None:
            raise ValueError("Use --max-frames with --tfrecord.")
        return max_frames
    if max_frames is not None:
        raise ValueError("Use --max-frames-per-tfrecord with --input-root or --tfrecord-list.")
    return max_frames_per_tfrecord


def export_tfrecords(
    paths: Sequence[Path],
    *,
    output_root: str | Path,
    split: str,
    frame_limit: int | None,
    overwrite: bool = False,
    skip_existing: bool = False,
) -> dict[str, Any]:
    """Export selected TFRecords into raw per-frame 32-line files.

    Input shape: sequence of TFRecord paths.
    Output shape: summary dictionary; files are ``[32,2650,2]`` float32 NPYs.

    Raises:
        AllFramesFailedError: If no frame was saved or skipped successfully.
    """
    if not paths:
        raise ValueError("No TFRecords were selected.")
    if overwrite and skip_existing:
        raise ValueError("--overwrite and --skip-existing are mutually exclusive.")
    output = validate_output_root(output_root)
    split = _safe_component(split, "split")
    split_root = output / split
    manifest_path = split_root / "manifest.jsonl"
    failures_path = split_root / "failures.json"
    existing_records = _read_manifest(manifest_path)
    records_by_sample = {
        str(record.get("sample_id")): record for record in existing_records if record.get("sample_id")
    }
    failures: list[dict[str, Any]] = []
    saved_count = 0
    skipped_count = 0
    processed_frames = 0
    failed_frames = 0
    processed_tfrecords = 0

    for tfrecord_path in paths:
        path = Path(tfrecord_path).expanduser().resolve()
        handled_in_file = False
        frame_iterator: Iterator[Any] = iter_frames(path)
        frame_index = 0
        while frame_limit is None or frame_index < frame_limit:
            try:
                frame = next(frame_iterator)
            except StopIteration:
                break
            except Exception as error:
                failures.append(_failure(path, frame_index, error))
                failed_frames += 1
                break

            try:
                context_name = str(getattr(getattr(frame, "context", None), "name", ""))
                _safe_component(context_name, "context_name")
                relative_base = Path(split) / context_name / f"{frame_index:06d}"
                relative_npy = relative_base.with_suffix(".npy").as_posix()
                relative_json = relative_base.with_suffix(".json").as_posix()
                npy_path = output / relative_npy
                json_path = output / relative_json
                both_exist = npy_path.exists() and json_path.exists()
                any_exist = npy_path.exists() or json_path.exists()
                if both_exist and skip_existing:
                    skipped_count += 1
                    processed_frames += 1
                    handled_in_file = True
                    frame_index += 1
                    continue
                if any_exist and not overwrite:
                    raise FileExistsError(
                        f"Existing output requires --overwrite or --skip-existing: {npy_path}, {json_path}"
                    )

                native = get_top_first_return(frame)
                packed, mask_32, ring_ids_32 = build_raw_32line(native)
                sidecar = build_sidecar(
                    frame=frame,
                    frame_index=frame_index,
                    split=split,
                    source_tfrecord=path,
                    packed=packed,
                    mask_32=mask_32,
                    ring_ids_32=ring_ids_32,
                    relative_npy_path=relative_npy,
                    relative_json_path=relative_json,
                )
                sample_id = f"{context_name}/{frame_index:06d}"
                manifest_record = {
                    "sample_id": sample_id,
                    "relative_npy_path": relative_npy,
                    "relative_json_path": relative_json,
                    "context_name": context_name,
                    "source_tfrecord": str(path),
                    "source_frame_index": frame_index,
                    "timestamp_micros": sidecar["timestamp_micros"],
                    "shape": list(packed.shape),
                    "valid_rate": sidecar["valid_rate"],
                }
                # Write JSON and NPY independently, each through an atomic rename.
                _atomic_write_json(json_path, sidecar, overwrite=overwrite)
                _atomic_write_npy(npy_path, packed, overwrite=overwrite)
                records_by_sample[sample_id] = manifest_record
                saved_count += 1
                processed_frames += 1
                handled_in_file = True
            except Exception as error:
                failures.append(_failure(path, frame_index, error))
                failed_frames += 1
            frame_index += 1
        if handled_in_file:
            processed_tfrecords += 1

    _write_manifest(manifest_path, list(records_by_sample.values()))
    _atomic_write_json(failures_path, failures, overwrite=True)
    summary = {
        "processed_tfrecords": processed_tfrecords,
        "processed_frames": processed_frames,
        "saved_samples": saved_count,
        "skipped_samples": skipped_count,
        "failed_frames": failed_frames,
        "failure_count": len(failures),
        "output_root": str(output),
        "manifest": str(manifest_path),
        "failures": str(failures_path),
    }
    if processed_frames == 0:
        raise AllFramesFailedError(
            f"All selected frames failed; see {failures_path}"
        )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    """Build the Phase 3 command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-root", type=Path)
    source.add_argument("--tfrecord", type=Path)
    source.add_argument("--tfrecord-list", type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-tfrecords", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-frames-per-tfrecord", type=int)
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument("--overwrite", action="store_true")
    controls.add_argument("--skip-existing", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Run the Phase 3 exporter and return a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.tfrecord is not None:
            source_kind = "single"
        else:
            source_kind = "multiple"
        frame_limit = _frame_limit(
            source_kind=source_kind,
            max_frames=args.max_frames,
            max_frames_per_tfrecord=args.max_frames_per_tfrecord,
        )
        paths = discover_tfrecords(
            input_root=args.input_root,
            tfrecord=args.tfrecord,
            tfrecord_list=args.tfrecord_list,
            max_tfrecords=args.max_tfrecords,
        )
        summary = export_tfrecords(
            paths,
            output_root=args.output_root,
            split=args.split,
            frame_limit=frame_limit,
            overwrite=args.overwrite,
            skip_existing=args.skip_existing,
        )
    except (AllFramesFailedError, FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"processed TFRecords: {summary['processed_tfrecords']}")
    print(f"processed frames: {summary['processed_frames']}")
    print(f"saved samples: {summary['saved_samples']}")
    print(f"skipped samples: {summary['skipped_samples']}")
    print(f"failure count: {summary['failure_count']}")
    print(f"manifest: {summary['manifest']}")
    print(f"failures: {summary['failures']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
