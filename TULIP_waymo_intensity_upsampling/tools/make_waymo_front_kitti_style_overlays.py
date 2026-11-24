#!/usr/bin/env python3
"""Create KITTI-style Waymo camera overlays for GT, Low, and Pred.

The tool consumes the already exported vehicle-frame point clouds and one
Waymo TFRecord frame.  It does not run TULIP inference.  For each of GT, the
16-line Low signal, and Pred it writes black-background pseudo-color images,
RGB overlays, legends, masks, and raw projected values.

Waymo's camera convention in this repository is ``X_c`` forward with
``u = fx * (-Y_c) / X_c + cx`` and ``v = fy * (-Z_c) / X_c + cy``.  Therefore
the depth images are labelled as camera-forward ``X_c`` rather than KITTI's
camera ``Z`` depth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODE_NAMES = ("gt", "low", "pred")
CAMERA_NAMES = ("FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT")
INTENSITY_COLORMAPS = ("plasma", "viridis", "turbo", "inferno", "magma", "cividis")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointcloud-npz", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--camera", default="FRONT", choices=CAMERA_NAMES)
    parser.add_argument("--modes", nargs="+", choices=MODE_NAMES, default=list(MODE_NAMES))
    parser.add_argument("--summary-name", default="summary.json")
    parser.add_argument("--depth-min", type=float, default=4.0)
    parser.add_argument("--depth-max", type=float, default=40.0)
    parser.add_argument("--depth-gamma", type=float, default=0.75)
    parser.add_argument(
        "--depth-mapping",
        choices=("log1p", "piecewise"),
        default="log1p",
        help="distance color mapping; piecewise reserves a separate far-distance color range",
    )
    parser.add_argument(
        "--depth-far-start",
        type=float,
        default=20.0,
        help="piecewise mapping boundary in meters",
    )
    parser.add_argument(
        "--depth-near-fraction",
        type=float,
        default=0.65,
        help="fraction of Turbo colors reserved for the near-distance segment",
    )
    parser.add_argument("--intensity-min", type=float, default=0.0)
    parser.add_argument("--intensity-max", type=float, default=20.0)
    parser.add_argument("--intensity-log-max", type=float, default=20000.0)
    parser.add_argument(
        "--intensity-colormap",
        choices=INTENSITY_COLORMAPS,
        default="turbo",
        help="colormap for intensity images; depth remains Turbo",
    )
    parser.add_argument("--point-size", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.65)
    return parser.parse_args()


def _read_frame(tfrecord: Path, frame_index: int) -> Any:
    """Read one Waymo Frame protobuf from a TFRecord.

    Input: a TFRecord path and a zero-based frame index.  Output: a
    ``waymo_open_dataset.dataset_pb2.Frame`` object.
    """
    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2

    if not tfrecord.is_file():
        raise FileNotFoundError(tfrecord)
    records = tf.data.TFRecordDataset(str(tfrecord), compression_type="")
    for index, record in enumerate(records):
        if index == frame_index:
            frame = dataset_pb2.Frame()
            frame.ParseFromString(record.numpy())
            return frame
    raise IndexError(f"frame {frame_index} not found in {tfrecord}")


def _camera_id(camera: str) -> int:
    """Resolve a Waymo camera enum name to its integer ID."""
    from waymo_open_dataset import dataset_pb2

    if not hasattr(dataset_pb2.CameraName, camera):
        raise ValueError(f"unknown camera name: {camera}")
    return int(getattr(dataset_pb2.CameraName, camera))


def _decode_camera_image(frame: Any, camera_id: int) -> np.ndarray:
    """Decode one camera JPEG as RGB uint8 ``[H,W,3]``."""
    record = next((item for item in frame.images if item.name == camera_id), None)
    if record is None:
        raise ValueError(f"camera image {camera_id} is missing from frame")
    encoded = np.frombuffer(record.image, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("failed to decode Waymo camera JPEG")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _camera_calibration(frame: Any, camera_id: int) -> Any:
    """Return the unique calibration record for a camera."""
    matches = [item for item in frame.context.camera_calibrations if item.name == camera_id]
    if len(matches) != 1:
        raise ValueError(f"expected one camera calibration, found {len(matches)}")
    return matches[0]


def _matrix4(values: Iterable[float]) -> np.ndarray:
    """Convert 16 row-major extrinsic values to a ``[4,4]`` matrix."""
    values_array = np.asarray(list(values), dtype=np.float64)
    if values_array.size != 16:
        raise ValueError(f"expected 16 extrinsic values, got {values_array.size}")
    return values_array.reshape(4, 4)


def project_vehicle_to_camera(
    points_vehicle: np.ndarray,
    calibration: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project vehicle points to Waymo camera coordinates.

    Input shape: ``points_vehicle [N,3]``.  Outputs ``u [N]``, ``v [N]``,
    camera-forward depth ``X_c [N]``, and a front-facing boolean mask ``[N]``.
    """
    points = np.asarray(points_vehicle, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_vehicle must be [N,3], got {points.shape}")
    intrinsic = list(calibration.intrinsic)
    if len(intrinsic) < 4:
        raise ValueError("camera intrinsic must contain fx, fy, cx, cy")
    fx, fy, cx, cy = (float(value) for value in intrinsic[:4])
    vehicle_from_camera = _matrix4(calibration.extrinsic.transform)
    camera_from_vehicle = np.linalg.inv(vehicle_from_camera)
    homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1
    )
    camera = (homogeneous @ camera_from_vehicle.T)[:, :3]
    depth = camera[:, 0]
    front = np.isfinite(depth) & (depth > 0.1)
    safe_depth = np.where(front, depth, 1.0)
    u = fx * (-camera[:, 1]) / safe_depth + cx
    v = fy * (-camera[:, 2]) / safe_depth + cy
    return u.astype(np.float32), v.astype(np.float32), depth.astype(np.float32), front


def _inside(u: np.ndarray, v: np.ndarray, width: int, height: int) -> np.ndarray:
    """Return the finite points inside image bounds."""
    return (
        np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0.0)
        & (u < float(width))
        & (v >= 0.0)
        & (v < float(height))
    )


def _nearest_map(
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    values: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterize the nearest point per rounded pixel.

    Input shapes: all point arrays ``[N]``.  Output shapes: value map and mask
    are ``[height,width]``.
    """
    result = np.zeros((height, width), dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)
    nearest = np.full((height, width), np.inf, dtype=np.float32)
    px = np.rint(u).astype(np.int64)
    py = np.rint(v).astype(np.int64)
    valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    for x, y, z, value in zip(px[valid], py[valid], depth[valid], values[valid]):
        if z < nearest[y, x]:
            nearest[y, x] = z
            result[y, x] = value
            mask[y, x] = True
    return result, mask


def _normalize(values: np.ndarray, lower: float, upper: float, *, log1p: bool, gamma: float = 1.0) -> np.ndarray:
    """Normalize raw values to ``[0,1]`` for PNG display only."""
    if not upper > lower:
        raise ValueError(f"upper must be greater than lower, got {lower}, {upper}")
    data = np.asarray(values, dtype=np.float32)
    if log1p:
        if lower < 0:
            raise ValueError("log1p display lower bound must be non-negative")
        data = np.log1p(np.clip(data, 0.0, None))
        lower = float(np.log1p(lower))
        upper = float(np.log1p(upper))
    normalized = np.clip((data - lower) / (upper - lower), 0.0, 1.0)
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    return np.power(normalized, gamma)


def _normalize_depth(
    values: np.ndarray,
    *,
    lower: float,
    upper: float,
    gamma: float,
    mapping: str,
    far_start: float,
    near_fraction: float,
) -> np.ndarray:
    """Normalize camera-forward depth with optional far-distance expansion.

    Input shape: arbitrary depth array. Output shape: the same array with
    values in ``[0,1]``. ``log1p`` retains the original mapping. ``piecewise``
    uses log1p+gamma for ``lower..far_start`` and a separate linear color
    segment for ``far_start..upper`` so distant points do not all saturate.
    """
    if mapping == "log1p":
        return _normalize(values, lower, upper, log1p=True, gamma=gamma)
    if not lower < far_start < upper:
        raise ValueError(
            f"piecewise requires lower < far_start < upper, got {lower}, {far_start}, {upper}"
        )
    if not 0.0 < near_fraction < 1.0:
        raise ValueError("depth-near-fraction must be in (0,1)")
    data = np.asarray(values, dtype=np.float32)
    near = _normalize(data, lower, far_start, log1p=True, gamma=gamma)
    far = np.clip((data - far_start) / (upper - far_start), 0.0, 1.0)
    return np.where(data <= far_start, near * near_fraction, near_fraction + far * (1.0 - near_fraction))


def _colors(normalized: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Convert normalized values and mask to RGB uint8 colors on black."""
    rgb = np.rint(matplotlib.colormaps["turbo"](normalized)[..., :3] * 255.0).astype(np.uint8)
    output = np.zeros((*normalized.shape, 3), dtype=np.uint8)
    output[np.asarray(mask, dtype=bool)] = rgb[np.asarray(mask, dtype=bool)]
    return output


def _draw_points(
    camera_rgb: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    colors_rgb: np.ndarray,
    *,
    point_size: int,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw filled colored circles on black and on the RGB camera.

    Input point arrays are ``[N]`` and colors are ``[N,3]`` RGB.  Outputs are
    RGB uint8 color and overlay images ``[H,W,3]``, plus the drawn mask.
    """
    if point_size < 1:
        raise ValueError("point-size must be at least 1")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0,1]")
    height, width = camera_rgb.shape[:2]
    color_bgr = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    order = np.argsort(np.nan_to_num(v, nan=np.inf))
    order = order[::-1]
    for index in order:
        center = (int(np.rint(u[index])), int(np.rint(v[index])))
        color = tuple(int(value) for value in colors_rgb[index][::-1])
        cv2.circle(color_bgr, center, point_size, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(mask, center, point_size, 255, thickness=-1, lineType=cv2.LINE_AA)
    camera_bgr = cv2.cvtColor(camera_rgb, cv2.COLOR_RGB2BGR)
    overlay_bgr = camera_bgr.copy()
    active = mask > 0
    overlay_bgr[active] = (
        (1.0 - alpha) * camera_bgr[active].astype(np.float32)
        + alpha * color_bgr[active].astype(np.float32)
    ).astype(np.uint8)
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    return (
        cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB),
        mask,
    )


def _save_legend(
    path: Path,
    label: str,
    lower: float,
    upper: float,
    *,
    log1p: bool,
    gamma: float,
    normalized_samples: np.ndarray | None = None,
    colormap: str = "turbo",
) -> None:
    """Save a standalone Turbo color legend PNG."""
    if normalized_samples is None:
        gradient = np.linspace(0.0, 1.0, 512, dtype=np.float32)[None, :]
    else:
        gradient = np.asarray(normalized_samples, dtype=np.float32)[None, :]
    figure, axis = plt.subplots(figsize=(9, 1.5), constrained_layout=True)
    axis.imshow(gradient, cmap=colormap, aspect="auto")
    ticks = np.linspace(0.0, 1.0, 6)
    labels = []
    for tick in ticks:
        value = lower + tick * (upper - lower)
        labels.append(f"{value:g}")
    axis.set_xticks(ticks * 511.0, labels)
    axis.set_yticks([])
    mode = "log1p display" if log1p else "linear display"
    axis.set_xlabel(f"{label} ({mode}, gamma={gamma:g})")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _save_mode(
    mode_dir: Path,
    camera_rgb: np.ndarray,
    points: np.ndarray,
    intensity: np.ndarray,
    calibration: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Project and save all images for one mode."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{mode_dir.name} points must be [N,3], got {points.shape}")
    if intensity.ndim != 1 or intensity.shape[0] != points.shape[0]:
        raise ValueError(f"{mode_dir.name} intensity must be [N], got {intensity.shape}")
    height, width = camera_rgb.shape[:2]
    u, v, depth, front = project_vehicle_to_camera(points, calibration)
    valid = front & _inside(u, v, width, height) & np.isfinite(intensity) & (intensity >= 0.0)
    u, v, depth, intensity = u[valid], v[valid], depth[valid], intensity[valid]
    depth_norm = _normalize_depth(
        depth,
        lower=args.depth_min,
        upper=args.depth_max,
        gamma=args.depth_gamma,
        mapping=args.depth_mapping,
        far_start=args.depth_far_start,
        near_fraction=args.depth_near_fraction,
    )
    depth_colors = np.asarray(matplotlib.colormaps["turbo"](depth_norm)[..., :3] * 255.0, dtype=np.uint8)
    intensity_norm = _normalize(intensity, args.intensity_min, args.intensity_max, log1p=False)
    intensity_colors = np.asarray(
        matplotlib.colormaps[args.intensity_colormap](intensity_norm)[..., :3] * 255.0,
        dtype=np.uint8,
    )
    intensity_log_norm = _normalize(intensity, args.intensity_min, args.intensity_log_max, log1p=True)
    intensity_log_colors = np.asarray(
        matplotlib.colormaps[args.intensity_colormap](intensity_log_norm)[..., :3] * 255.0,
        dtype=np.uint8,
    )

    depth_color, depth_overlay, depth_drawn = _draw_points(
        camera_rgb, u, v, depth_colors, point_size=args.point_size, alpha=args.alpha
    )
    intensity_color, intensity_overlay, intensity_drawn = _draw_points(
        camera_rgb, u, v, intensity_colors, point_size=args.point_size, alpha=args.alpha
    )
    intensity_log_color, intensity_log_overlay, intensity_log_drawn = _draw_points(
        camera_rgb, u, v, intensity_log_colors, point_size=args.point_size, alpha=args.alpha
    )
    mode_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mode_dir / "depth_color.png"), cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mode_dir / "depth_overlay.png"), cv2.cvtColor(depth_overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mode_dir / "intensity_color.png"), cv2.cvtColor(intensity_color, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mode_dir / "intensity_overlay.png"), cv2.cvtColor(intensity_overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mode_dir / "intensity_log1p_color.png"), cv2.cvtColor(intensity_log_color, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mode_dir / "intensity_log1p_overlay.png"), cv2.cvtColor(intensity_log_overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mode_dir / "mask.png"), depth_drawn)
    _save_legend(
        mode_dir / "depth_legend.png",
        "camera-forward depth Xc (m)",
        args.depth_min,
        args.depth_max,
        log1p=args.depth_mapping == "log1p",
        gamma=args.depth_gamma,
        normalized_samples=_normalize_depth(
            np.linspace(args.depth_min, args.depth_max, 512, dtype=np.float32),
            lower=args.depth_min,
            upper=args.depth_max,
            gamma=args.depth_gamma,
            mapping=args.depth_mapping,
            far_start=args.depth_far_start,
            near_fraction=args.depth_near_fraction,
        ),
    )
    _save_legend(
        mode_dir / "intensity_legend.png",
        "raw intensity",
        args.intensity_min,
        args.intensity_max,
        log1p=False,
        gamma=1.0,
        colormap=args.intensity_colormap,
    )
    _save_legend(
        mode_dir / "intensity_log1p_legend.png",
        "raw intensity",
        args.intensity_min,
        args.intensity_log_max,
        log1p=True,
        gamma=1.0,
        colormap=args.intensity_colormap,
    )
    map_intensity, map_mask = _nearest_map(u, v, depth, intensity, width, height)
    np.savez_compressed(
        mode_dir / "projection_raw.npz",
        u=u,
        v=v,
        depth_xc=depth,
        intensity=intensity,
        intensity_map=map_intensity,
        intensity_mask=map_mask,
    )
    return {
        "projected_points": int(u.size),
        "drawn_pixels": int((depth_drawn > 0).sum()),
        "depth_min_projected": float(depth.min()) if depth.size else None,
        "depth_max_projected": float(depth.max()) if depth.size else None,
        "intensity_min_projected": float(intensity.min()) if intensity.size else None,
        "intensity_max_projected": float(intensity.max()) if intensity.size else None,
        "raw_npz": str(mode_dir / "projection_raw.npz"),
    }


def main() -> None:
    """Generate separate GT, Low, and Pred KITTI-style camera artifacts."""
    args = parse_args()
    if args.depth_min < 0 or args.depth_max <= args.depth_min:
        raise ValueError("depth bounds are invalid")
    if args.intensity_min < 0 or args.intensity_max <= args.intensity_min:
        raise ValueError("intensity bounds are invalid")
    if args.intensity_log_max <= args.intensity_min:
        raise ValueError("intensity-log-max must exceed intensity-min")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    source_tfrecord = Path(metadata["source_tfrecord"])
    source_frame_index = int(metadata["source_frame_index"])
    data = np.load(args.pointcloud_npz, allow_pickle=False)
    required = {
        "points_gt_vehicle", "intensity_gt",
        "points_low_vehicle", "intensity_low",
        "points_pred_vehicle", "intensity_pred",
    }
    missing = sorted(required.difference(data.files))
    if missing:
        raise KeyError(f"pointcloud NPZ is missing keys: {missing}")
    frame = _read_frame(source_tfrecord, source_frame_index)
    calibration = _camera_calibration(frame, _camera_id(args.camera))
    camera_rgb = _decode_camera_image(frame, _camera_id(args.camera))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_dir / "camera_rgb.png"), cv2.cvtColor(camera_rgb, cv2.COLOR_RGB2BGR))
    summary: Dict[str, Any] = {
        "camera": args.camera,
        "source_tfrecord": str(source_tfrecord),
        "source_frame_index": source_frame_index,
        "camera_shape": list(camera_rgb.shape),
        "coordinate_convention": "vehicle<-camera extrinsic inverse; Xc forward; u=-Yc/Xc; v=-Zc/Xc",
        "depth_semantics": "camera-forward Xc, not KITTI Z_cam and not radial LiDAR range",
        "display_only": {
            "depth_min": args.depth_min,
            "depth_max": args.depth_max,
            "depth_mapping": args.depth_mapping,
            "depth_far_start": args.depth_far_start,
            "depth_near_fraction": args.depth_near_fraction,
            "depth_gamma": args.depth_gamma,
            "intensity_min": args.intensity_min,
            "intensity_max": args.intensity_max,
            "intensity_log_max": args.intensity_log_max,
            "intensity_colormap": args.intensity_colormap,
            "point_size": args.point_size,
            "alpha": args.alpha,
        },
        "modes": {},
    }
    for mode in args.modes:
        summary["modes"][mode] = _save_mode(
            args.output_dir / mode,
            camera_rgb,
            np.asarray(data[f"points_{mode}_vehicle"], dtype=np.float32),
            np.asarray(data[f"intensity_{mode}"], dtype=np.float32),
            calibration,
            args,
        )
    (args.output_dir / args.summary_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
