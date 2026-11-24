#!/usr/bin/env python3
"""Create pure RGB-camera GT, Low, and TULIP Pred intensity comparisons.

GT uses the 32-line target range and intensity. Low uses the 16 observed
lines placed on the 32-row grid. Pred uses both TULIP-predicted range and
intensity. The existing oracle condition is not used by this tool.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tulip_adapter.waymo_16_to_32_dataset import Waymo16To32Dataset
from tulip_adapter.waymo_16_to_32_tulip import build_waymo_16_to_32_model
from tulip_adapter.waymo_pointcloud import build_top_geometry, paired_vehicle_points
from tools.evaluate_waymo_16_to_32_camera_projection import (
    _camera_calibration,
    _camera_name,
    _decode_camera_image,
    _in_image,
    _zbuffer,
    project_vehicle_to_camera,
)
from tools.reconstruct_waymo_16_to_32_pointcloud import (
    _raw_signals,
    _read_frame,
    _source_from_pair_json,
    _top_pose,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--tulip-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-role", default="test")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--camera",
        default="FRONT",
        choices=("FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"),
    )
    parser.add_argument("--range-scale-m", type=float, default=75.0)
    parser.add_argument("--intensity-transform", choices=("raw", "log1p"), default="log1p")
    parser.add_argument("--intensity-vmax", type=float, default=20.0)
    parser.add_argument("--intensity-log-vmax", type=float, default=20000.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a trusted local checkpoint across PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def inverse_transform(
    tensor: torch.Tensor,
    *,
    range_scale_m: float,
    intensity_transform: str,
) -> np.ndarray:
    """Convert [B,2,H,W] model values to raw [B,2,H,W] float32 values."""
    values = tensor.detach().cpu().numpy().astype(np.float32, copy=True)
    if values.ndim != 4 or values.shape[1] != 2:
        raise ValueError(f"expected [B,2,H,W], got {values.shape}")
    values[:, 0] *= np.float32(range_scale_m)
    if intensity_transform == "log1p":
        values[:, 1] = np.expm1(values[:, 1])
    return values


def project_mode(
    range_values: np.ndarray,
    intensity_values: np.ndarray,
    mask: np.ndarray,
    geometry: Any,
    calibration: Any,
    image_shape: tuple[int, int, int],
) -> dict[str, np.ndarray]:
    """Project one mode into the camera image.

    Input shapes: range/intensity/mask are [32,2650], [32,2650], [32,2650].
    Output point arrays are [N] and maps are [height,width].
    """
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(range_values)
        & (range_values > 0.0)
        & np.isfinite(intensity_values)
    )
    points, values = paired_vehicle_points(range_values, intensity_values, geometry, valid)
    height, width = image_shape[:2]
    u, v, depth, front = project_vehicle_to_camera(points, calibration)
    inside = front & _in_image(u, v, width, height)
    map_values, map_mask = _zbuffer(
        u[inside], v[inside], depth[inside], values[inside], width, height
    )
    return {
        "u": u[inside],
        "v": v[inside],
        "depth": depth[inside],
        "intensity": values[inside],
        "map": map_values,
        "mask": map_mask,
        "valid_count": np.asarray(int(valid.sum()), dtype=np.int64),
    }


def normalized_map(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    vmax: float,
    log1p_display: bool,
) -> np.ndarray:
    """Convert a raw map to RGB display colors using a shared fixed scale."""
    if vmax <= 0:
        raise ValueError("vmax must be positive")
    data = np.clip(np.asarray(values, dtype=np.float32), 0.0, None)
    if log1p_display:
        data = np.log1p(data)
        upper = float(np.log1p(vmax))
    else:
        upper = float(vmax)
    normalized = np.clip(data / max(upper, 1e-6), 0.0, 1.0)
    color = plt.get_cmap("magma")(normalized)[..., :3]
    valid = np.asarray(mask, dtype=bool) & np.isfinite(values) & (values > 0.0)
    output = np.zeros((*values.shape, 3), dtype=np.uint8)
    output[valid] = np.rint(color[valid] * 255.0).astype(np.uint8)
    return output


def save_projection_panels(
    modes: list[dict[str, np.ndarray]],
    titles: list[str],
    *,
    output: Path,
    title: str,
    vmax: float,
    log1p_display: bool,
) -> None:
    """Save three common-scale projected intensity panels."""
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    image = None
    for axis, mode, panel_title in zip(axes, modes, titles):
        display = np.clip(mode["map"], 0.0, None)
        if log1p_display:
            display = np.log1p(display)
            display_vmax = np.log1p(vmax)
        else:
            display_vmax = vmax
        image = axis.imshow(
            np.ma.array(display, mask=~mode["mask"]),
            cmap="magma",
            vmin=0.0,
            vmax=float(display_vmax),
            interpolation="nearest",
        )
        axis.set_title(panel_title)
        axis.set_xlabel("camera pixel u")
        axis.set_ylabel("camera pixel v")
    if image is not None:
        figure.colorbar(image, ax=axes, pad=0.01)
    figure.suptitle(title, fontsize=15)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def save_overlay_panels(
    camera_rgb: np.ndarray,
    modes: list[dict[str, np.ndarray]],
    titles: list[str],
    *,
    output: Path,
    title: str,
    vmax: float,
    log1p_display: bool,
) -> None:
    """Save three common-scale intensity overlays on the RGB image."""
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for axis, mode, panel_title in zip(axes, modes, titles):
        color = normalized_map(
            mode["map"],
            mode["mask"],
            vmax=vmax,
            log1p_display=log1p_display,
        )
        overlay = camera_rgb.astype(np.float32).copy()
        valid = mode["mask"] & (color.sum(axis=2) > 0)
        overlay[valid] = 0.55 * overlay[valid] + 0.45 * color[valid]
        axis.imshow(np.clip(overlay, 0, 255).astype(np.uint8))
        axis.set_title(panel_title)
        axis.axis("off")
    figure.suptitle(title, fontsize=15)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    """Generate pure GT/Low/Pred camera projection artifacts."""
    args = parse_args()
    if args.sample_index < 0:
        raise ValueError("sample-index must be non-negative")
    if args.range_scale_m <= 0 or args.intensity_vmax <= 0:
        raise ValueError("scale and display maxima must be positive")
    if args.intensity_log_vmax <= 0:
        raise ValueError("intensity-log-vmax must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    dataset = Waymo16To32Dataset(
        args.index,
        dataset_role=args.dataset_role,
        intensity_transform=args.intensity_transform,
        range_scale_m=args.range_scale_m,
    )
    if args.sample_index >= len(dataset):
        raise IndexError(f"sample-index {args.sample_index} >= dataset length {len(dataset)}")
    sample = dataset[args.sample_index]
    model = build_waymo_16_to_32_model(args.tulip_root).to(device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with torch.inference_mode():
        input_signal = sample["input"].unsqueeze(0).to(device)
        target_signal = sample["target"].unsqueeze(0).to(device)
        target_mask = sample["target_mask"].unsqueeze(0).to(device)
        prediction, _ = model(input_signal, target_signal, target_mask)
    input_raw = inverse_transform(
        input_signal,
        range_scale_m=args.range_scale_m,
        intensity_transform=args.intensity_transform,
    )[0, ..., :2650]
    target_raw = inverse_transform(
        target_signal,
        range_scale_m=args.range_scale_m,
        intensity_transform=args.intensity_transform,
    )[0, ..., :2650]
    pred_raw = inverse_transform(
        prediction,
        range_scale_m=args.range_scale_m,
        intensity_transform=args.intensity_transform,
    )[0, ..., :2650]

    target_mask_native = sample["target_mask"].numpy()[..., :2650].astype(bool)
    input_mask_native = sample["input_mask"].numpy()[..., :2650].astype(bool)
    low_raw = np.zeros_like(target_raw, dtype=np.float32)
    low_mask = np.zeros_like(target_mask_native, dtype=bool)
    low_raw[:, 0::2, :] = input_raw
    low_mask[0::2, :] = input_mask_native

    source_tfrecord, source_frame_index = _source_from_pair_json(
        ROOT / dataset.rows[args.sample_index]["pair_json"]
    )
    frame = _read_frame(source_tfrecord, source_frame_index)
    camera_id = _camera_name(args.camera)
    calibration = _camera_calibration(frame, camera_id)
    camera_rgb = _decode_camera_image(frame, camera_id)
    geometry = build_top_geometry(
        frame,
        _top_pose(frame),
        np.arange(0, 64, 2, dtype=np.int32),
        native_width=2650,
    )
    modes = [
        project_mode(
            target_raw[0],
            target_raw[1],
            target_mask_native,
            geometry,
            calibration,
            camera_rgb.shape,
        ),
        project_mode(
            low_raw[0],
            low_raw[1],
            low_mask,
            geometry,
            calibration,
            camera_rgb.shape,
        ),
        project_mode(
            pred_raw[0],
            pred_raw[1],
            target_mask_native,
            geometry,
            calibration,
            camera_rgb.shape,
        ),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    titles = [
        "GT intensity",
        "Low intensity (16-line)",
        "TULIP Pred intensity",
    ]
    plt.imsave(args.output_dir / "camera_rgb.png", camera_rgb)
    save_projection_panels(
        modes,
        titles,
        output=args.output_dir / "intensity_projection_gt_low_pred_linear.png",
        title=f"Waymo {args.camera} pure GT / Low / Pred intensity",
        vmax=args.intensity_vmax,
        log1p_display=False,
    )
    save_projection_panels(
        modes,
        titles,
        output=args.output_dir / "intensity_projection_gt_low_pred_log1p.png",
        title=f"Waymo {args.camera} pure GT / Low / Pred intensity log1p",
        vmax=args.intensity_log_vmax,
        log1p_display=True,
    )
    save_overlay_panels(
        camera_rgb,
        modes,
        titles,
        output=args.output_dir / "intensity_overlay_gt_low_pred_linear.png",
        title=f"Waymo {args.camera} RGB overlay GT / Low / Pred",
        vmax=args.intensity_vmax,
        log1p_display=False,
    )
    save_overlay_panels(
        camera_rgb,
        modes,
        titles,
        output=args.output_dir / "intensity_overlay_gt_low_pred_log1p.png",
        title=f"Waymo {args.camera} RGB overlay GT / Low / Pred log1p",
        vmax=args.intensity_log_vmax,
        log1p_display=True,
    )
    np.savez_compressed(
        args.output_dir / "camera_comparison_raw.npz",
        gt_u=modes[0]["u"],
        gt_v=modes[0]["v"],
        gt_depth=modes[0]["depth"],
        gt_intensity=modes[0]["intensity"],
        gt_map=modes[0]["map"],
        gt_mask=modes[0]["mask"],
        low_u=modes[1]["u"],
        low_v=modes[1]["v"],
        low_depth=modes[1]["depth"],
        low_intensity=modes[1]["intensity"],
        low_map=modes[1]["map"],
        low_mask=modes[1]["mask"],
        pred_u=modes[2]["u"],
        pred_v=modes[2]["v"],
        pred_depth=modes[2]["depth"],
        pred_intensity=modes[2]["intensity"],
        pred_map=modes[2]["map"],
        pred_mask=modes[2]["mask"],
    )
    metadata = {
        "checkpoint": str(args.checkpoint),
        "dataset_role": args.dataset_role,
        "sample_index": args.sample_index,
        "camera": args.camera,
        "source_tfrecord": str(source_tfrecord),
        "source_frame_index": int(source_frame_index),
        "range_scale_m": args.range_scale_m,
        "intensity_transform": args.intensity_transform,
        "coordinate_convention": "vehicle<-camera extrinsic inverse; Xc forward; u=-Yc/Xc; v=-Zc/Xc",
        "comparison_scope": "pure GT range+intensity, Low 16-line range+intensity, Pred range+intensity",
        "native_shape": [32, 2650],
        "camera_shape": list(camera_rgb.shape),
        "png_is_visualization_only": True,
        "display_scales": {
            "linear_intensity_vmax": args.intensity_vmax,
            "log1p_intensity_vmax_raw": args.intensity_log_vmax,
        },
        "modes": {
            name: {
                "valid_lidar_pixels": int(mode["valid_count"]),
                "projected_points_in_image": int(mode["u"].size),
                "projection_rate_over_valid_lidar_pixels": float(
                    mode["u"].size / max(int(mode["valid_count"]), 1)
                ),
            }
            for name, mode in zip(("gt", "low", "pred"), modes)
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "camera": args.camera,
        "camera_shape": list(camera_rgb.shape),
        "gt_projected_points": int(modes[0]["u"].size),
        "low_projected_points": int(modes[1]["u"].size),
        "pred_projected_points": int(modes[2]["u"].size),
        "raw_npz": str(args.output_dir / "camera_comparison_raw.npz"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
