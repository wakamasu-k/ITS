"""Waymo range-image to point-cloud helpers for the TULIP reproduction.

The actual Waymo coordinate conversion is delegated to the installed Waymo
Open Dataset utility.  This is deliberate: beam inclination ordering,
TOP-lidar pixel poses, and vehicle-frame conversion must not be reconstructed
from row numbers or guessed angles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np


@dataclass
class WaymoTopGeometry:
    """Verified TOP geometry tensors used by the Waymo SDK converter.

    ``inclinations`` has shape ``[H]`` and ``pixel_pose`` has shape
    ``[H,W,4,4]`` for the selected native rings.  ``extrinsic`` and
    ``frame_pose`` both have shape ``[4,4]``.  The resulting points are in
    the Waymo vehicle frame when ``pixel_pose`` and ``frame_pose`` are used.
    """

    inclinations: Any
    extrinsic: Any
    pixel_pose: Any
    frame_pose: Any
    ring_ids: np.ndarray
    native_height: int
    native_width: int


def valid_range_mask(
    range_image: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return finite, positive, optionally externally masked range pixels.

    Input shape: ``range_image`` is ``[H,W]``; ``valid_mask`` is optional
    ``[H,W]``.  Output shape: boolean ``[H,W]``.
    """

    values = np.asarray(range_image, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"range_image must be [H,W], got {values.shape}")
    result = np.isfinite(values) & (values > 0.0)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=bool)
        if supplied.shape != values.shape:
            raise ValueError(
                f"valid_mask must have shape {values.shape}, got {supplied.shape}"
            )
        result &= supplied
    return result


def pointwise_metrics(
    predicted_points: np.ndarray,
    target_points: np.ndarray,
    predicted_intensity: np.ndarray,
    target_intensity: np.ndarray,
) -> dict[str, float | int]:
    """Calculate paired point and intensity errors at identical pixels.

    Input shapes: point arrays are ``[N,3]`` and intensity arrays are ``[N]``.
    Output shape: a scalar-statistics dictionary.  This is a paired metric,
    not a nearest-neighbour Chamfer distance.
    """

    pred = np.asarray(predicted_points, dtype=np.float32)
    target = np.asarray(target_points, dtype=np.float32)
    pred_i = np.asarray(predicted_intensity, dtype=np.float32)
    target_i = np.asarray(target_intensity, dtype=np.float32)
    if pred.ndim != 2 or pred.shape[1] != 3 or target.shape != pred.shape:
        raise ValueError("predicted_points and target_points must both be [N,3]")
    if pred_i.shape != (pred.shape[0],) or target_i.shape != (pred.shape[0],):
        raise ValueError("intensity arrays must both be [N]")
    if pred.shape[0] == 0:
        raise ValueError("paired point arrays are empty")
    if not all(np.isfinite(value).all() for value in (pred, target, pred_i, target_i)):
        raise ValueError("point and intensity arrays must be finite")
    delta = pred - target
    return {
        "count": int(pred.shape[0]),
        "coordinate_mae_m": float(np.abs(delta).mean()),
        "euclidean_mae_m": float(np.linalg.norm(delta, axis=1).mean()),
        "intensity_mae": float(np.abs(pred_i - target_i).mean()),
    }


def _matrix_float_to_numpy(message: Any) -> np.ndarray:
    """Decode a Waymo MatrixFloat protobuf message to a NumPy array."""

    dimensions = [
        int(value.size if hasattr(value, "size") else value)
        for value in message.shape.dims
    ]
    values = np.asarray(message.data, dtype=np.float32)
    expected = int(np.prod(dimensions))
    if values.size != expected:
        raise ValueError(f"MatrixFloat size mismatch: {values.size} != {expected}")
    return values.reshape(dimensions)


def build_top_geometry(
    frame: Any,
    range_image_top_pose: Any,
    ring_ids: np.ndarray,
    *,
    native_width: int,
) -> WaymoTopGeometry:
    """Build selected TOP geometry using the official Waymo conventions.

    Input: a parsed Waymo ``Frame``, its TOP ``range_image_top_pose`` protobuf,
    and native ring IDs such as ``[0,2,...,62]``.  Output: a
    :class:`WaymoTopGeometry` object whose selected tensors match
    ``[len(ring_ids), native_width]``.  No angle or coordinate convention is
    inferred locally.
    """

    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2
    from waymo_open_dataset.utils import range_image_utils, transform_utils

    selected = np.asarray(ring_ids, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("ring_ids must be a non-empty 1-D array")
    if np.any(selected < 0) or np.any(selected >= 64):
        raise ValueError("ring_ids must be within native TOP height 64")

    top = dataset_pb2.LaserName.TOP
    calibrations = [item for item in frame.context.laser_calibrations if item.name == top]
    if len(calibrations) != 1:
        raise ValueError(f"expected one TOP calibration, found {len(calibrations)}")
    calibration = calibrations[0]
    native_height = 64
    if calibration.beam_inclinations:
        beam = tf.constant(calibration.beam_inclinations, dtype=tf.float32)
    else:
        beam = range_image_utils.compute_inclination(
            tf.constant(
                [calibration.beam_inclination_min, calibration.beam_inclination_max],
                dtype=tf.float32,
            ),
            height=native_height,
        )
    beam = tf.reverse(beam, axis=[-1])
    beam = tf.gather(beam, tf.constant(selected, dtype=tf.int32))

    pose_values = _matrix_float_to_numpy(range_image_top_pose)
    if pose_values.shape[:2] != (native_height, native_width) or pose_values.shape[2] < 6:
        raise ValueError(
            "TOP range-image pose must be [64,W,6], "
            f"got {pose_values.shape}"
        )
    pose_tensor = tf.convert_to_tensor(pose_values, dtype=tf.float32)
    rotation = transform_utils.get_rotation_matrix(
        pose_tensor[..., 0], pose_tensor[..., 1], pose_tensor[..., 2]
    )
    translation = pose_tensor[..., 3:6]
    pixel_pose = transform_utils.get_transform(rotation, translation)
    pixel_pose = tf.gather(pixel_pose, tf.constant(selected, dtype=tf.int32), axis=0)

    extrinsic = np.asarray(calibration.extrinsic.transform, dtype=np.float32).reshape(4, 4)
    frame_pose = np.asarray(frame.pose.transform, dtype=np.float32).reshape(4, 4)
    return WaymoTopGeometry(
        inclinations=beam,
        extrinsic=tf.convert_to_tensor(extrinsic[None, ...], dtype=tf.float32),
        pixel_pose=tf.convert_to_tensor(pixel_pose[None, ...], dtype=tf.float32),
        frame_pose=tf.convert_to_tensor(frame_pose[None, ...], dtype=tf.float32),
        ring_ids=selected.astype(np.int32),
        native_height=native_height,
        native_width=native_width,
    )


def range_image_to_vehicle_points(
    range_image: np.ndarray,
    geometry: WaymoTopGeometry,
    mask: np.ndarray,
) -> np.ndarray:
    """Convert a selected TOP range image to vehicle-frame points.

    Input shapes: range image and mask are ``[H,W]`` where ``H`` equals the
    number of selected rings; geometry tensors use the same ``H,W``.  Output
    shape: ``[N,3]`` in Waymo vehicle coordinates, where ``N`` is the number
    of true valid pixels in ``mask``.
    """

    import tensorflow as tf
    from waymo_open_dataset.utils import range_image_utils

    values = np.asarray(range_image, dtype=np.float32)
    supplied_mask = valid_range_mask(values, mask)
    if values.shape != (geometry.ring_ids.size, geometry.native_width):
        raise ValueError(
            f"range image must be {(geometry.ring_ids.size, geometry.native_width)}, "
            f"got {values.shape}"
        )
    if not supplied_mask.any():
        return np.empty((0, 3), dtype=np.float32)
    tensor = tf.convert_to_tensor(values[None, ...], dtype=tf.float32)
    cartesian = range_image_utils.extract_point_cloud_from_range_image(
        tensor,
        geometry.extrinsic,
        geometry.inclinations[None, ...],
        pixel_pose=geometry.pixel_pose,
        frame_pose=geometry.frame_pose,
    )
    points = np.asarray(cartesian[0].numpy(), dtype=np.float32)
    return points[supplied_mask]


def paired_vehicle_points(
    range_image: np.ndarray,
    intensity_image: np.ndarray,
    geometry: WaymoTopGeometry,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert range/intensity pixels into paired vehicle-frame points.

    Input shapes: range, intensity, and mask are all ``[H,W]``.  Output
    shapes: points ``[N,3]`` and intensity ``[N]`` using the same row-major
    mask order.
    """

    range_values = np.asarray(range_image, dtype=np.float32)
    intensity_values = np.asarray(intensity_image, dtype=np.float32)
    if intensity_values.shape != range_values.shape:
        raise ValueError("range and intensity shapes differ")
    selected_mask = valid_range_mask(range_values, mask)
    selected_mask &= np.isfinite(intensity_values)
    points = range_image_to_vehicle_points(range_values, geometry, selected_mask)
    return points, intensity_values[selected_mask].astype(np.float32)
