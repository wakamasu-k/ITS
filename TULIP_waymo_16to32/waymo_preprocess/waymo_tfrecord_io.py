"""Read native Waymo TOP LiDAR range images from TFRecord files.

This module is isolated from the existing repositories and must run in the
Waymo/TensorFlow environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils


def _matrix_float_to_numpy(range_image: object) -> np.ndarray:
    """Convert a Waymo MatrixFloat to float32 HWC.

    Args:
        range_image: Waymo MatrixFloat protobuf with shape.dims and data.

    Returns:
        Native array with shape [H, W, C] and dtype float32.

    Raises:
        TypeError: If MatrixFloat fields are missing.
        ValueError: If dimensions or data length are invalid.
    """
    if not hasattr(range_image, "shape") or not hasattr(range_image, "data"):
        raise TypeError("range_image must provide MatrixFloat shape and data.")

    dimensions = []
    for dimension in range_image.shape.dims:
        size = int(dimension.size if hasattr(dimension, "size") else dimension)
        if size <= 0:
            raise ValueError(f"Range Image dimension must be positive, got {size}.")
        dimensions.append(size)
    if len(dimensions) != 3:
        raise ValueError(f"Expected a 3D Range Image, got dims={dimensions}.")

    values = np.asarray(range_image.data, dtype=np.float32)
    expected_size = int(np.prod(dimensions))
    if values.size != expected_size:
        raise ValueError(
            f"Range Image data size {values.size} does not match "
            f"dims={dimensions} ({expected_size})."
        )
    return values.reshape(tuple(dimensions)).astype(np.float32, copy=False)


def iter_frames(tfrecord_path: str | Path) -> Iterator[open_dataset.Frame]:
    """Yield parsed Waymo frames in TFRecord order.

    Args:
        tfrecord_path: Uncompressed Waymo TFRecord path.

    Yields:
        Waymo Frame messages. Each frame comes from one TFRecord example.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the path is not a file.
        protobuf parsing errors: Propagated for malformed records.
    """
    path = Path(tfrecord_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"TFRecord does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"TFRecord path is not a file: {path}")

    dataset = tf.data.TFRecordDataset(str(path), compression_type="")
    for record in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(record.numpy())
        yield frame


def get_frame(tfrecord_path: str | Path, frame_index: int) -> open_dataset.Frame:
    """Read one zero-based frame from a Waymo TFRecord.

    Args:
        tfrecord_path: Uncompressed Waymo TFRecord path.
        frame_index: Non-negative zero-based frame index.

    Returns:
        Parsed Waymo Frame at frame_index.

    Raises:
        ValueError: If frame_index is negative.
        IndexError: If the frame does not exist.
        FileNotFoundError: If the TFRecord does not exist.
    """
    if frame_index < 0:
        raise ValueError(f"frame_index must be non-negative, got {frame_index}.")
    for current_index, frame in enumerate(iter_frames(tfrecord_path)):
        if current_index == frame_index:
            return frame
    raise IndexError(f"frame_index {frame_index} is outside {tfrecord_path}.")


def get_top_first_return(frame: open_dataset.Frame) -> np.ndarray:
    """Extract TOP LiDAR first-return native Range Image.

    Args:
        frame: Parsed Waymo Frame message.

    Returns:
        TOP first-return Range Image with shape [H, W, C] and dtype float32.
        No normalization or camera projection is applied.

    Raises:
        KeyError: If TOP LiDAR is absent.
        ValueError: If TOP has no first return or invalid dimensions.
    """
    range_images, _, _, _ = frame_utils.parse_range_image_and_camera_projection(frame)
    top_lidar = open_dataset.LaserName.TOP
    if top_lidar not in range_images:
        raise KeyError("TOP LiDAR is missing from parsed range images.")
    top_returns = range_images[top_lidar]
    if not top_returns:
        raise ValueError("TOP LiDAR has no first-return Range Image.")
    return _matrix_float_to_numpy(top_returns[0])
