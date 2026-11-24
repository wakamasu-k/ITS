#!/usr/bin/env python3
"""Render A/B/C static maps from one common Waymo FRONT camera anchor.

The numerical comparison uses one-pixel z-buffer projections.  Display images
use the same depth-aware radius-2 splat as the Shimizu renderer, with one
shared log-intensity normalization across A/B/C.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2


OFFSETS_R2 = [
    (-2, 0), (-1, -1), (-1, 0), (-1, 1), (0, -2), (0, -1),
    (0, 1), (0, 2), (1, -1), (1, 0), (1, 1), (2, 0),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tfrecord", type=Path, required=True)
    p.add_argument("--maps-root", type=Path, required=True)
    p.add_argument("--subset", default="training")
    p.add_argument("--frame-index", type=int, default=2)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--int-percentile", type=float, default=99.5)
    p.add_argument("--display-gamma", type=float, default=0.5)
    return p.parse_args()


def matrix(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(4, 4)


def load_frame(path: Path, index: int):
    for i, record in enumerate(tf.data.TFRecordDataset(str(path), compression_type="")):
        if i == index:
            frame = dataset_pb2.Frame()
            frame.ParseFromString(record.numpy())
            return frame
    raise IndexError(f"frame {index} not found in {path}")


def camera_model(frame):
    name = dataset_pb2.CameraName.FRONT
    calibration = next(c for c in frame.context.camera_calibrations if c.name == name)
    fx, fy, cx, cy = map(float, calibration.intrinsic[:4])
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    # Waymo convention used in Shimizu's fixed-SE(3) renderer.
    world_to_camera = np.linalg.inv(matrix(calibration.extrinsic.transform)) @ np.linalg.inv(
        matrix(frame.pose.transform)
    )
    return calibration, k, world_to_camera


def save_front_rgb(frame, path: Path) -> np.ndarray:
    """Save the unmodified Waymo FRONT JPEG for visual reference."""
    image = next(im for im in frame.images if im.name == dataset_pb2.CameraName.FRONT)
    bgr = cv2.imdecode(np.frombuffer(image.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("failed to decode FRONT camera image")
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"failed to write {path}")
    return bgr


def project(xyz: np.ndarray, intensity: np.ndarray, k: np.ndarray, twc: np.ndarray, width: int, height: int):
    homogeneous = np.column_stack((xyz, np.ones(xyz.shape[0], dtype=xyz.dtype)))
    camera = homogeneous @ twc.T
    depth = camera[:, 0]
    front = np.isfinite(camera).all(axis=1) & (depth > 0.1)
    u = np.zeros(depth.shape, dtype=np.int32)
    v = np.zeros(depth.shape, dtype=np.int32)
    u[front] = (k[0, 0] * (-camera[front, 1] / depth[front]) + k[0, 2]).astype(np.int32)
    v[front] = (k[1, 1] * (-camera[front, 2] / depth[front]) + k[1, 2]).astype(np.int32)
    inside = front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u[inside], v[inside], depth[inside].astype(np.float32), intensity[inside].astype(np.float32), {
        "points_total": int(xyz.shape[0]),
        "points_in_front": int(front.sum()),
        "points_in_image": int(inside.sum()),
    }


def zbuffer(u, v, depth, intensity, width: int, height: int, splat: bool):
    depth_map = np.full((height, width), np.inf, dtype=np.float32)
    intensity_map = np.zeros((height, width), dtype=np.float32)
    order = np.argsort(depth)[::-1]
    uo, vo, do, io = u[order], v[order], depth[order], intensity[order]
    depth_map[vo, uo] = do
    intensity_map[vo, uo] = io
    if splat:
        for du, dv in OFFSETS_R2:
            uu, vv = uo + du, vo + dv
            valid = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
            if not np.any(valid):
                continue
            uu, vv, dd, ii = uu[valid], vv[valid], do[valid], io[valid]
            nearer = dd < depth_map[vv, uu]
            depth_map[vv[nearer], uu[nearer]] = dd[nearer]
            intensity_map[vv[nearer], uu[nearer]] = ii[nearer]
    return depth_map, intensity_map


def pair_metrics(candidate, reference) -> dict[str, float | int]:
    dc, ic = candidate
    dr, ir = reference
    mc, mr = np.isfinite(dc), np.isfinite(dr)
    overlap = mc & mr
    union = mc | mr
    result: dict[str, float | int] = {
        "covered_pixels": int(mc.sum()),
        "reference_covered_pixels": int(mr.sum()),
        "overlap_pixels": int(overlap.sum()),
        "additional_pixels_vs_reference": int((mc & ~mr).sum()),
        "missing_pixels_vs_reference": int((mr & ~mc).sum()),
        "coverage_recall_vs_reference": float(overlap.sum() / max(1, mr.sum())),
        "coverage_iou_vs_reference": float(overlap.sum() / max(1, union.sum())),
    }
    if np.any(overlap):
        depth_error = dc[overlap] - dr[overlap]
        log_intensity_error = np.log1p(np.maximum(ic[overlap], 0.0)) - np.log1p(
            np.maximum(ir[overlap], 0.0)
        )
        result.update({
            "depth_mae_m": float(np.mean(np.abs(depth_error))),
            "depth_rmse_m": float(np.sqrt(np.mean(depth_error ** 2))),
            "log_intensity_mae": float(np.mean(np.abs(log_intensity_error))),
            "log_intensity_rmse": float(np.sqrt(np.mean(log_intensity_error ** 2))),
        })
    return result


def nearest_reference_within_radius(
    candidate_depth: np.ndarray,
    reference_depth: np.ndarray,
    radius_px: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the nearest reference pixel/depth for every candidate pixel."""
    h, w = candidate_depth.shape
    reference_mask = np.isfinite(reference_depth)
    best_distance2 = np.full((h, w), np.inf, dtype=np.float32)
    nearest_depth = np.full((h, w), np.inf, dtype=np.float32)
    for dy in range(-radius_px, radius_px + 1):
        for dx in range(-radius_px, radius_px + 1):
            distance2 = dx * dx + dy * dy
            if distance2 > radius_px * radius_px:
                continue
            shifted_mask = np.roll(reference_mask, shift=(-dy, -dx), axis=(0, 1))
            shifted_depth = np.roll(reference_depth, shift=(-dy, -dx), axis=(0, 1))
            if dy > 0:
                shifted_mask[-dy:, :] = False
            elif dy < 0:
                shifted_mask[:-dy, :] = False
            if dx > 0:
                shifted_mask[:, -dx:] = False
            elif dx < 0:
                shifted_mask[:, :-dx] = False
            update = shifted_mask & (distance2 < best_distance2)
            best_distance2[update] = float(distance2)
            nearest_depth[update] = shifted_depth[update]
    return np.sqrt(best_distance2), nearest_depth


def tolerant_classification(
    candidate_depth: np.ndarray,
    reference_depth: np.ndarray,
    radius_px: int = 3,
) -> tuple[np.ndarray, dict[str, int | float], dict[str, np.ndarray]]:
    """Classify B pixels using a 3px GT neighbourhood and depth error."""
    candidate = np.isfinite(candidate_depth)
    reference = np.isfinite(reference_depth)
    pixel_distance, nearest_depth = nearest_reference_within_radius(
        candidate_depth, reference_depth, radius_px
    )
    matched = candidate & (pixel_distance <= radius_px)
    depth_error = np.full(candidate_depth.shape, np.inf, dtype=np.float32)
    depth_error[matched] = np.abs(candidate_depth[matched] - nearest_depth[matched])
    exact = candidate & reference
    good = matched & (depth_error <= 0.5)
    caution = matched & (depth_error > 0.5) & (depth_error <= 1.0)
    poor = matched & (depth_error > 1.0) & (depth_error <= 2.0)
    severe = matched & (depth_error > 2.0)
    no_nearby_gt = candidate & ~matched
    reference_only = reference & ~candidate

    # BGR colours: exact white, good green, caution yellow, poor orange,
    # severe magenta, no-nearby-GT red, and exact C-only blue.
    image = np.zeros((*candidate_depth.shape, 3), dtype=np.uint8)
    image[exact] = (255, 255, 255)
    image[good & ~exact] = (0, 255, 0)
    image[caution] = (0, 255, 255)
    image[poor] = (0, 128, 255)
    image[severe] = (255, 0, 255)
    image[no_nearby_gt] = (0, 0, 255)
    image[reference_only] = (255, 0, 0)

    counts = {
        "candidate_pixels": int(candidate.sum()),
        "exact_same_pixel": int(exact.sum()),
        "matched_within_radius": int(matched.sum()),
        "matched_within_radius_fraction": float(matched.sum() / max(1, candidate.sum())),
        "good_depth_le_0_5m": int(good.sum()),
        "caution_depth_0_5_to_1m": int(caution.sum()),
        "poor_depth_1_to_2m": int(poor.sum()),
        "severe_depth_gt_2m": int(severe.sum()),
        "no_reference_within_radius": int(no_nearby_gt.sum()),
        "reference_only_exact_pixel": int(reference_only.sum()),
        "radius_px": int(radius_px),
    }
    masks = {
        "severe": severe,
        "no_nearby_gt": no_nearby_gt,
    }
    return image, counts, masks


def main() -> None:
    args = parse_args()
    frame = load_frame(args.tfrecord, args.frame_index)
    calibration, k, twc = camera_model(frame)
    width, height = int(calibration.width), int(calibration.height)
    segment = args.tfrecord.stem
    roots = {"A": "A_observed16", "B": "B_tulip32", "C": "C_gt32"}
    projections, numeric, display = {}, {}, {}
    pooled_log_intensity = []
    for key, directory in roots.items():
        path = args.maps_root / directory / args.subset / segment / "map_static.npz"
        with np.load(path) as data:
            xyz = data["xyz"]
            intensity = data["intensity"]
        u, v, depth, values, counts = project(xyz, intensity, k, twc, width, height)
        projections[key] = (u, v, depth, values)
        numeric[key] = (*zbuffer(u, v, depth, values, width, height, False), counts)
        display[key] = zbuffer(u, v, depth, values, width, height, True)
        pooled_log_intensity.append(np.log1p(np.maximum(values, 0.0)))

    pooled = np.concatenate(pooled_log_intensity)
    shared_upper = float(np.percentile(pooled, args.int_percentile))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_bgr = save_front_rgb(frame, args.output_dir / "source_front_rgb.jpg")
    rendered = []
    variant_report = {}
    for key in ("A", "B", "C"):
        depth_map, intensity_map = display[key]
        mask = np.isfinite(depth_map)
        gray = np.zeros((height, width), dtype=np.uint8)
        normalized = np.clip(
            np.log1p(np.maximum(intensity_map[mask], 0.0)) / max(shared_upper, 1e-6),
            0.0, 1.0,
        )
        gray[mask] = (np.power(normalized, args.display_gamma) * 255.0).astype(np.uint8)
        cv2.imwrite(str(args.output_dir / f"{key}_front.png"), gray)
        rendered.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        dn, inn, counts = numeric[key]
        variant_report[key] = {
            **counts,
            "unique_pixels": int(np.isfinite(dn).sum()),
            "image_coverage": float(np.isfinite(dn).mean()),
        }

    panel = np.concatenate(rendered, axis=1)
    for i, text in enumerate(("A observed 16", "B TULIP 32", "C GT 32")):
        cv2.putText(panel, text, (i * width + 30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
    cv2.imwrite(str(args.output_dir / "ABC_front_comparison.png"), panel)

    # Diagnostic pixel classes: white=common B/C, red=B only, blue=C only.
    mask_b = np.isfinite(numeric["B"][0])
    mask_c = np.isfinite(numeric["C"][0])
    common = mask_b & mask_c
    b_only = mask_b & ~mask_c
    c_only = mask_c & ~mask_b
    classes = np.zeros((height, width, 3), dtype=np.uint8)
    classes[common] = (255, 255, 255)
    classes[b_only] = (0, 0, 255)
    classes[c_only] = (255, 0, 0)
    cv2.rectangle(classes, (10, 10), (620, 120), (0, 0, 0), -1)
    cv2.putText(classes, "WHITE: common B/C", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(classes, "RED: B-only additional", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(classes, "BLUE: C-only missing in B", (25, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.imwrite(str(args.output_dir / "B_vs_C_pixel_classes.png"), classes)

    # Make sparse B-only pixels visible on the normal RGB image without
    # changing the numerical mask used by the report.
    visible_b_only = cv2.dilate(b_only.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    overlay = source_bgr.copy()
    red = np.zeros_like(overlay)
    red[:, :] = (0, 0, 255)
    overlay[visible_b_only] = cv2.addWeighted(
        overlay[visible_b_only], 0.35, red[visible_b_only], 0.65, 0.0
    )
    cv2.putText(overlay, "RED: B-only additional pixels (5px display dilation)",
                (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imwrite(str(args.output_dir / "B_additional_on_source_rgb.jpg"), overlay)

    tolerant_image, tolerant_counts, tolerant_masks = tolerant_classification(
        numeric["B"][0], numeric["C"][0], radius_px=3
    )
    cv2.rectangle(tolerant_image, (10, 10), (790, 225), (0, 0, 0), -1)
    legend = (
        ("WHITE exact B/C", (255, 255, 255)),
        ("GREEN <=3px and depth error <=0.5m", (0, 255, 0)),
        ("YELLOW depth error 0.5-1m", (0, 255, 255)),
        ("ORANGE depth error 1-2m", (0, 128, 255)),
        ("MAGENTA depth error >2m", (255, 0, 255)),
        ("RED no GT within 3px", (0, 0, 255)),
        ("BLUE C-only at exact pixel", (255, 0, 0)),
    )
    for row, (label, colour) in enumerate(legend):
        cv2.putText(tolerant_image, label, (25, 38 + row * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)
    cv2.imwrite(str(args.output_dir / "B_vs_C_tolerant_depth_classes.png"), tolerant_image)

    suspicious = tolerant_masks["severe"] | tolerant_masks["no_nearby_gt"]
    visible_suspicious = cv2.dilate(suspicious.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    suspicious_overlay = source_bgr.copy()
    magenta = np.zeros_like(suspicious_overlay)
    magenta[:, :] = (255, 0, 255)
    suspicious_overlay[visible_suspicious] = cv2.addWeighted(
        suspicious_overlay[visible_suspicious], 0.3,
        magenta[visible_suspicious], 0.7, 0.0
    )
    cv2.putText(suspicious_overlay, "MAGENTA: no GT <=3px or depth error >2m",
                (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
    cv2.imwrite(str(args.output_dir / "B_suspicious_on_source_rgb.jpg"), suspicious_overlay)

    report = {
        "source_tfrecord": str(args.tfrecord),
        "frame_index": args.frame_index,
        "camera": "FRONT",
        "image_size": [width, height],
        "numeric_projection": "one-pixel nearest-depth z-buffer",
        "display_projection": "shared log1p intensity normalization plus depth-aware radius-2 splat",
        "shared_log_intensity_percentile": args.int_percentile,
        "shared_log_intensity_upper": shared_upper,
        "shared_display_gamma": args.display_gamma,
        "variants": variant_report,
        "A_vs_C": pair_metrics(numeric["A"][:2], numeric["C"][:2]),
        "B_vs_C": pair_metrics(numeric["B"][:2], numeric["C"][:2]),
        "B_vs_A": pair_metrics(numeric["B"][:2], numeric["A"][:2]),
        "B_vs_C_tolerant_3px": tolerant_counts,
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
