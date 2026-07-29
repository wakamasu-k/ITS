#!/usr/bin/env python3
"""Run the trained two-channel TULIP model on one traceable KITTI frame.

Inputs are the outputs of ``one_frame_diagnostic.py``.  Both the direct network
prediction and the evaluation-style fused prediction are saved:

* ``pred_raw``: direct network output after inverse log/scale conversion.
* ``pred_fused``: invalid ranges removed and measured low-resolution rows
  restored exactly, matching the intent of the existing evaluation pipeline.

Keeping both prevents post-processing from being mistaken for model quality.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TULIP_PACKAGE_ROOT = REPOSITORY_ROOT / "tulip"
for path in (REPOSITORY_ROOT, TULIP_PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model import tulip as tulip_model  # noqa: E402
from util.kitti_range_image import (  # noqa: E402
    DEFAULT_LOW_ROW_INDICES,
    KITTI_COLS,
    KITTI_ROWS,
    unproject_range_image_to_xyzi,
    validate_range_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two-channel TULIP inference for one KITTI frame."
    )
    parser.add_argument("--low", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--frame", default="000000")
    parser.add_argument("--max_range_m", type=float, default=80.0)
    parser.add_argument("--min_range_m", type=float, default=2.0)
    parser.add_argument(
        "--low_rows",
        type=int,
        nargs="+",
        default=list(DEFAULT_LOW_ROW_INDICES),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", type=Path, required=True)
    return parser.parse_args()


def _checkpoint_value(saved_args: object, name: str):
    if isinstance(saved_args, dict):
        return saved_args.get(name)
    return getattr(saved_args, name)


def build_model_from_checkpoint(
    checkpoint: dict,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    """Rebuild the exact architecture recorded in the training checkpoint."""

    saved_args = checkpoint.get("args")
    if saved_args is None:
        raise ValueError("Checkpoint does not contain saved training arguments")

    required = {
        name: _checkpoint_value(saved_args, name)
        for name in (
            "model_select",
            "img_size_low_res",
            "img_size_high_res",
            "patch_size",
            "in_chans",
            "window_size",
            "swin_v2",
            "pixel_shuffle",
            "circular_padding",
            "log_transform",
            "patch_unmerging",
        )
    }
    if required["in_chans"] != 2:
        raise ValueError(
            f"Intensity inference requires a 2-channel checkpoint, "
            f"got in_chans={required['in_chans']}"
        )

    model_factory = getattr(tulip_model, required["model_select"])
    model = model_factory(
        img_size=tuple(required["img_size_low_res"]),
        target_img_size=tuple(required["img_size_high_res"]),
        patch_size=tuple(required["patch_size"]),
        in_chans=required["in_chans"],
        window_size=required["window_size"],
        swin_v2=required["swin_v2"],
        pixel_shuffle=required["pixel_shuffle"],
        circular_padding=required["circular_padding"],
        log_transform=required["log_transform"],
        patch_unmerging=required["patch_unmerging"],
    )
    incompatible = model.load_state_dict(checkpoint["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError(f"Unexpected checkpoint mismatch: {incompatible}")
    model.to(device)
    model.eval()
    return model, required


def meter_hwc_to_training_tensor(
    image: np.ndarray,
    *,
    max_range_m: float,
    log_transform: bool,
    device: torch.device,
) -> torch.Tensor:
    """Apply the exact KITTI preprocessing used by the two-channel training."""

    array = np.asarray(image, dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).clone()
    tensor[:, 0] /= max_range_m
    if log_transform:
        # The historical training pipeline applies log1p to both channels.
        # We preserve that behavior here to diagnose the existing checkpoint.
        tensor = torch.log1p(tensor)
    return tensor.to(device)


def training_tensor_to_meter_hwc(
    tensor: torch.Tensor,
    *,
    max_range_m: float,
    log_transform: bool,
) -> np.ndarray:
    """Invert training preprocessing while keeping model outputs unclipped."""

    output = tensor.detach().float().cpu()
    if log_transform:
        output = torch.expm1(output)
    array = output[0].permute(1, 2, 0).numpy()
    array[..., 0] *= max_range_m
    return array.astype(np.float32)


def write_ascii_xyzi_ply(path: Path, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float32)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {points.shape[0]}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property float intensity\nend_header\n")
        np.savetxt(stream, points, fmt="%.7f %.7f %.7f %.7f")


def error_metrics(
    prediction: np.ndarray,
    gt: np.ndarray,
    *,
    low_rows: tuple[int, ...],
) -> dict[str, float | int]:
    """Report intensity errors separately for valid and newly generated rows."""

    gt_valid = gt[..., 0] > 0.0
    generated_rows = np.ones(KITTI_ROWS, dtype=bool)
    generated_rows[list(low_rows)] = False
    generated_valid = gt_valid & generated_rows[:, None]

    intensity_error = prediction[..., 1] - gt[..., 1]
    range_error = prediction[..., 0] - gt[..., 0]

    def masked(values: np.ndarray, mask: np.ndarray, prefix: str) -> dict:
        selected = values[mask]
        if selected.size == 0:
            return {
                f"{prefix}_count": 0,
                f"{prefix}_mae": float("nan"),
                f"{prefix}_rmse": float("nan"),
                f"{prefix}_bias": float("nan"),
            }
        return {
            f"{prefix}_count": int(selected.size),
            f"{prefix}_mae": float(np.mean(np.abs(selected))),
            f"{prefix}_rmse": float(np.sqrt(np.mean(selected**2))),
            f"{prefix}_bias": float(np.mean(selected)),
        }

    metrics: dict[str, float | int] = {}
    metrics.update(masked(intensity_error, gt_valid, "intensity_gt_valid"))
    metrics.update(
        masked(intensity_error, generated_valid, "intensity_generated_gt_valid")
    )
    metrics.update(masked(range_error, gt_valid, "range_gt_valid_m"))
    metrics.update(
        masked(range_error, generated_valid, "range_generated_gt_valid_m")
    )
    return metrics


def main() -> None:
    args = parse_args()
    for path in (args.low, args.gt, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    low = np.load(args.low).astype(np.float32)
    gt = np.load(args.gt).astype(np.float32)
    low_rows = tuple(args.low_rows)
    if low.shape != (len(low_rows), KITTI_COLS, 2):
        raise ValueError(f"Unexpected low shape: {low.shape}")
    if gt.shape != (KITTI_ROWS, KITTI_COLS, 2):
        raise ValueError(f"Unexpected GT shape: {gt.shape}")
    if not np.array_equal(low, gt[list(low_rows)]):
        raise ValueError("Low input is not the requested exact subset of GT")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model, architecture = build_model_from_checkpoint(checkpoint, device=device)
    log_transform = bool(architecture["log_transform"])

    low_tensor = meter_hwc_to_training_tensor(
        low,
        max_range_m=args.max_range_m,
        log_transform=log_transform,
        device=device,
    )
    gt_tensor = meter_hwc_to_training_tensor(
        gt,
        max_range_m=args.max_range_m,
        log_transform=log_transform,
        device=device,
    )

    with torch.no_grad(), torch.cuda.amp.autocast(
        enabled=device.type == "cuda"
    ):
        prediction_tensor, _, _ = model(low_tensor, gt_tensor, eval=True)

    pred_raw = training_tensor_to_meter_hwc(
        prediction_tensor,
        max_range_m=args.max_range_m,
        log_transform=log_transform,
    )

    # Reproduce the physically valid post-processing separately.  Direct model
    # output remains saved so clipping or row replacement cannot hide errors.
    pred_fused = pred_raw.copy()
    valid_range = (
        np.isfinite(pred_fused[..., 0])
        & (pred_fused[..., 0] >= args.min_range_m)
        & (pred_fused[..., 0] <= args.max_range_m)
    )
    pred_fused[~valid_range] = 0.0
    pred_fused[list(low_rows)] = low

    output_dir = args.out_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_id = f"seq{args.sequence}_frame{args.frame}"
    np.save(output_dir / f"{scene_id}_pred_raw_64x1024x2.npy", pred_raw)
    np.save(output_dir / f"{scene_id}_pred_fused_64x1024x2.npy", pred_fused)

    pred_points = unproject_range_image_to_xyzi(
        pred_fused,
        full_rows=KITTI_ROWS,
        cols=KITTI_COLS,
        range_unit="meter",
        max_range_m=args.max_range_m,
    )
    write_ascii_xyzi_ply(
        output_dir / f"{scene_id}_pred_fused_xyzi.ply", pred_points
    )

    metrics = {
        "scene_id": scene_id,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "architecture": architecture,
        "pred_raw_range_min_m": float(np.nanmin(pred_raw[..., 0])),
        "pred_raw_range_max_m": float(np.nanmax(pred_raw[..., 0])),
        "pred_raw_intensity_min": float(np.nanmin(pred_raw[..., 1])),
        "pred_raw_intensity_max": float(np.nanmax(pred_raw[..., 1])),
        "pred_raw": error_metrics(pred_raw, gt, low_rows=low_rows),
        "pred_fused": error_metrics(pred_fused, gt, low_rows=low_rows),
        "pred_fused_validation": validate_range_image(
            pred_fused, max_range_m=args.max_range_m
        ),
        "known_rows_restored_exactly": bool(
            np.array_equal(pred_fused[list(low_rows)], low)
        ),
    }
    (output_dir / f"{scene_id}_inference_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved TULIP inference outputs to: {output_dir}")


if __name__ == "__main__":
    main()
