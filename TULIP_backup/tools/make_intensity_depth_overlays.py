#!/usr/bin/env python3
"""
KITTIのカメラ画像へLiDAR点群を投影し、同じ可視点を使って次を作成する。

1. 反射強度の疑似カラー画像
2. 反射強度のカメラオーバーレイ
3. Camera Z距離の疑似カラー画像
4. Camera Z距離のカメラオーバーレイ

入力点群:
- KITTI .bin: float32 [x, y, z, reflectance]
- ASCII .ply: x, y, z と intensity / reflectance / i を含むもの

細かい色分け対応版。

必要モジュール:
    pip install numpy opencv-python
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


def read_calib(path: Path) -> Dict[str, np.ndarray]:
    """KITTI calib.txtからP0-P3とTrを読む。"""
    calib: Dict[str, np.ndarray] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue

            key, values = line.strip().split(":", 1)
            vals = np.fromstring(values, sep=" ", dtype=np.float64)

            if key in {"P0", "P1", "P2", "P3"} and vals.size == 12:
                calib[key] = vals.reshape(3, 4)
            elif key in {"Tr", "Tr_velo_to_cam"} and vals.size == 12:
                calib["Tr"] = vals.reshape(3, 4)

    if "Tr" not in calib:
        raise ValueError(f"Tr / Tr_velo_to_cam が見つかりません: {path}")

    return calib


def load_ascii_ply(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """ASCII PLYからxyzと反射強度を読む。"""
    properties: list[str] = []
    vertex_count = None
    header_lines = 0
    ply_format = None

    with path.open("r", encoding="utf-8", errors="strict") as f:
        for line in f:
            header_lines += 1
            stripped = line.strip()

            if stripped.startswith("format "):
                ply_format = stripped.split()[1]
            elif stripped.startswith("element vertex "):
                vertex_count = int(stripped.split()[-1])
            elif stripped.startswith("property "):
                properties.append(stripped.split()[-1])
            elif stripped == "end_header":
                break

    if ply_format != "ascii":
        raise ValueError(
            f"このスクリプトはASCII PLYのみ対応です: format={ply_format}, file={path}"
        )
    if vertex_count is None:
        raise ValueError(f"PLYのelement vertexが見つかりません: {path}")

    data = np.loadtxt(
        path,
        dtype=np.float32,
        skiprows=header_lines,
        max_rows=vertex_count,
    )
    if data.ndim == 1:
        data = data[None, :]

    index = {name: i for i, name in enumerate(properties)}
    missing = [name for name in ("x", "y", "z") if name not in index]
    if missing:
        raise ValueError(f"PLYに必要なpropertyがありません: {missing}")

    intensity_name = next(
        (name for name in ("intensity", "reflectance", "i") if name in index),
        None,
    )
    if intensity_name is None:
        raise ValueError(
            "PLYに intensity / reflectance / i のいずれも見つかりません。"
        )

    xyz = np.column_stack(
        (
            data[:, index["x"]],
            data[:, index["y"]],
            data[:, index["z"]],
        )
    ).astype(np.float32)
    intensity = data[:, index[intensity_name]].astype(np.float32)
    return xyz, intensity


def load_points(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """KITTI BINまたはASCII PLYを読む。"""
    suffix = path.suffix.lower()

    if suffix == ".bin":
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 4 != 0:
            raise ValueError(f"KITTI BINの要素数が4の倍数ではありません: {path}")
        points = raw.reshape(-1, 4)
        return points[:, :3], points[:, 3]

    if suffix == ".ply":
        return load_ascii_ply(path)

    raise ValueError(f"未対応の点群形式です: {suffix}")


def project_visible_points(
    xyz_lidar: np.ndarray,
    intensity: np.ndarray,
    projection: np.ndarray,
    tr_lidar_to_cam: np.ndarray,
    image_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    LiDAR点をカメラ平面へ投影する。

    同一画素へ複数点が投影された場合はCamera Zが最小の点を採用する。
    戻り値は u, v, Camera Z, intensity。
    """
    height, width = image_shape

    xyz1 = np.concatenate(
        (
            xyz_lidar.astype(np.float64),
            np.ones((xyz_lidar.shape[0], 1), dtype=np.float64),
        ),
        axis=1,
    )

    cam_xyz = (tr_lidar_to_cam @ xyz1.T).T
    camera_z = cam_xyz[:, 2]

    cam_xyz1 = np.concatenate(
        (cam_xyz, np.ones((cam_xyz.shape[0], 1), dtype=np.float64)),
        axis=1,
    )
    uvw = (projection @ cam_xyz1.T).T

    valid = (
        np.isfinite(camera_z)
        & (camera_z > 0.0)
        & np.isfinite(uvw).all(axis=1)
        & np.isfinite(intensity)
        & (np.abs(uvw[:, 2]) > 1e-12)
    )

    uvw = uvw[valid]
    depth = camera_z[valid]
    reflectance = intensity[valid]

    u = np.rint(uvw[:, 0] / uvw[:, 2]).astype(np.int32)
    v = np.rint(uvw[:, 1] / uvw[:, 2]).astype(np.int32)

    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[inside]
    v = v[inside]
    depth = depth[inside]
    reflectance = reflectance[inside]

    if u.size == 0:
        return u, v, depth, reflectance

    # 画素番号ごとに並べ、同じ画素ではdepthが小さい点を先頭にする。
    linear = v.astype(np.int64) * width + u.astype(np.int64)
    order = np.lexsort((depth, linear))
    linear_sorted = linear[order]

    keep = np.ones(linear_sorted.size, dtype=bool)
    keep[1:] = linear_sorted[1:] != linear_sorted[:-1]
    selected = order[keep]

    return u[selected], v[selected], depth[selected], reflectance[selected]


def apply_color_bins(normalized: np.ndarray, bins: int) -> np.ndarray:
    """
    0-1値を指定段階数に量子化する。
    bins=0 または bins>=256 の場合は連続色（最大256階調）のままにする。
    """
    if bins <= 1 or bins >= 256:
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    return (
        np.rint(np.clip(normalized, 0.0, 1.0) * (bins - 1)) / (bins - 1)
    ).astype(np.float32)


def normalize_intensity(
    values: np.ndarray,
    mode: str,
    intensity_min: float,
    intensity_max: float,
    upper_percentile: float,
) -> np.ndarray:
    """
    反射強度を0-1へ正規化する。

    fixed:
        intensity_min-intensity_maxを固定して比較可能にする。
    percentile:
        各画像のpercentileで外れ値を除き、見やすくする。
    equalized:
        percentile正規化後にヒストグラム均等化し、色を広く使う。
        見やすさ優先で、画像間の絶対値比較には向かない。
    """
    if values.size == 0:
        return np.empty(0, dtype=np.float32)

    if mode == "fixed":
        lo = float(intensity_min)
        hi = float(intensity_max)
    else:
        lo = max(float(intensity_min), float(np.percentile(values, 0.5)))
        hi = float(np.percentile(values, upper_percentile))

    if hi <= lo:
        hi = lo + 1e-6

    normalized = np.clip((values - lo) / (hi - lo), 0.0, 1.0)

    if mode == "equalized":
        values_u8 = np.rint(normalized * 255.0).astype(np.uint8)
        normalized = (
            cv2.equalizeHist(values_u8.reshape(-1, 1)).reshape(-1).astype(np.float32)
            / 255.0
        )

    return normalized.astype(np.float32)


def normalize_depth(
    depth: np.ndarray,
    depth_min: float,
    depth_max: float,
    gamma: float,
    scale: str,
) -> np.ndarray:
    """
    Camera Zを0-1へ正規化する。

    linear:
        距離に比例する通常の線形表示。
    gamma:
        gamma<1で近距離の色差を広げる。
    log:
        近距離に集中した点を最も分かりやすく広げる。
    """
    span = depth_max - depth_min
    clipped = np.clip(depth - depth_min, 0.0, span)

    if scale == "log":
        normalized = np.log1p(clipped) / np.log1p(span)
    else:
        normalized = clipped / span

    if scale in {"gamma", "log"}:
        normalized = np.power(normalized, gamma)

    return normalized.astype(np.float32)


def turbo_colors(normalized: np.ndarray, reverse: bool) -> np.ndarray:
    """0-1値をOpenCV TURBOのBGR色へ変換する。"""
    mapped = 1.0 - normalized if reverse else normalized
    values_u8 = np.rint(np.clip(mapped, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(values_u8.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)


def render_points(
    image_shape: Tuple[int, int],
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    colors: np.ndarray,
    point_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    点を黒背景へ描画する。

    遠い点から描き、近い点を後から上書きするため、点を太くしても
    intensity画像とdepth画像で同じ可視点関係を保ちやすい。
    """
    height, width = image_shape
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)

    if u.size == 0:
        return canvas, mask

    # 遠方 -> 近傍。近い点が最終的に前面に残る。
    draw_order = np.argsort(depth)[::-1]
    radius = max(0, int(point_size))

    for idx in draw_order:
        center = (int(u[idx]), int(v[idx]))
        color = tuple(int(c) for c in colors[idx])

        if radius == 0:
            canvas[center[1], center[0]] = color
            mask[center[1], center[0]] = 255
        else:
            cv2.circle(canvas, center, radius, color, thickness=-1, lineType=cv2.LINE_8)
            cv2.circle(mask, center, radius, 255, thickness=-1, lineType=cv2.LINE_8)

    return canvas, mask


def blend_overlay(
    camera_bgr: np.ndarray,
    point_color: np.ndarray,
    mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """点がある場所だけカメラ画像と疑似カラーを合成する。"""
    covered = mask > 0
    overlay = camera_bgr.copy()
    blended = cv2.addWeighted(camera_bgr, 1.0 - alpha, point_color, alpha, 0.0)
    overlay[covered] = blended[covered]
    return overlay


def make_colorbar(
    height: int,
    width: int,
    reverse: bool,
    top_label: str,
    bottom_label: str,
) -> np.ndarray:
    """簡易的な縦カラーバー画像を作る。"""
    height = max(height, 120)
    width = max(width, 180)
    bar_width = 36

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    y_values = np.linspace(1.0, 0.0, height - 20, dtype=np.float32)
    colors = turbo_colors(y_values, reverse=reverse)

    x0 = 10
    for y, color in enumerate(colors, start=10):
        canvas[y, x0 : x0 + bar_width] = color

    cv2.putText(canvas, top_label, (58, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, bottom_label, (58, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", required=True, type=Path)
    parser.add_argument("--calib", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--camera", default="P0", choices=("P0", "P1", "P2", "P3"))
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--prefix", default="pred")
    parser.add_argument("--depth_min", type=float, default=0.0)
    parser.add_argument("--depth_max", type=float, default=80.0)
    parser.add_argument(
        "--depth_scale",
        choices=("linear", "gamma", "log"),
        default="log",
        help="距離の色展開。近距離が赤に集中する場合はlog推奨。",
    )
    parser.add_argument(
        "--depth_gamma",
        type=float,
        default=0.75,
        help="gamma/log表示の追加補正。小さいほど近距離の色差が広がる。",
    )
    parser.add_argument(
        "--depth_bins",
        type=int,
        default=40,
        help="距離の色段階数。40なら0-80mを約2m刻み。256で連続色。",
    )
    parser.add_argument(
        "--intensity_mode",
        choices=("fixed", "percentile", "equalized"),
        default="equalized",
        help="反射強度の色展開。見やすさはequalized、画像間比較はfixed。",
    )
    parser.add_argument("--intensity_min", type=float, default=0.0)
    parser.add_argument("--intensity_max", type=float, default=1.0)
    parser.add_argument("--intensity_percentile", type=float, default=99.5)
    parser.add_argument(
        "--intensity_bins",
        type=int,
        default=64,
        help="反射強度の色段階数。256で連続色。",
    )
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument(
        "--point_size",
        type=int,
        default=2,
        help="表示上の点半径。正確な1画素投影は0。",
    )
    args = parser.parse_args()

    if not args.points.exists():
        raise FileNotFoundError(f"点群がありません: {args.points}")
    if not args.calib.exists():
        raise FileNotFoundError(f"calibがありません: {args.calib}")
    if not args.image.exists():
        raise FileNotFoundError(f"カメラ画像がありません: {args.image}")
    if not (0.0 < args.alpha <= 1.0):
        raise ValueError("--alphaは0より大きく1以下にしてください。")
    if args.depth_max <= args.depth_min:
        raise ValueError("--depth_maxは--depth_minより大きくしてください。")
    if args.depth_gamma <= 0.0:
        raise ValueError("--depth_gammaは0より大きくしてください。")
    if args.intensity_max <= args.intensity_min:
        raise ValueError("--intensity_maxは--intensity_minより大きくしてください。")
    if args.depth_bins < 0 or args.intensity_bins < 0:
        raise ValueError("色段階数は0以上にしてください。")

    camera_gray = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if camera_gray is None:
        raise RuntimeError(f"カメラ画像を読み込めません: {args.image}")
    camera_bgr = cv2.cvtColor(camera_gray, cv2.COLOR_GRAY2BGR)

    calib = read_calib(args.calib)
    if args.camera not in calib:
        raise ValueError(f"{args.camera}がcalibにありません: {args.calib}")

    xyz, intensity = load_points(args.points)
    u, v, camera_z, intensity_visible = project_visible_points(
        xyz_lidar=xyz,
        intensity=intensity,
        projection=calib[args.camera],
        tr_lidar_to_cam=calib["Tr"],
        image_shape=camera_gray.shape,
    )

    intensity_normalized = normalize_intensity(
        intensity_visible,
        mode=args.intensity_mode,
        intensity_min=args.intensity_min,
        intensity_max=args.intensity_max,
        upper_percentile=args.intensity_percentile,
    )
    depth_normalized = normalize_depth(
        camera_z,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        gamma=args.depth_gamma,
        scale=args.depth_scale,
    )

    intensity_normalized = apply_color_bins(
        intensity_normalized, args.intensity_bins
    )
    depth_normalized = apply_color_bins(
        depth_normalized, args.depth_bins
    )

    # 反射強度: 低い=青紫、高い=赤
    intensity_colors = turbo_colors(intensity_normalized, reverse=False)
    # Camera Z: 近い=赤、遠い=青紫
    depth_colors = turbo_colors(depth_normalized, reverse=True)

    intensity_color, intensity_mask = render_points(
        camera_gray.shape,
        u,
        v,
        camera_z,
        intensity_colors,
        point_size=args.point_size,
    )
    depth_color, depth_mask = render_points(
        camera_gray.shape,
        u,
        v,
        camera_z,
        depth_colors,
        point_size=args.point_size,
    )

    intensity_overlay = blend_overlay(
        camera_bgr,
        intensity_color,
        intensity_mask,
        alpha=args.alpha,
    )
    depth_overlay = blend_overlay(
        camera_bgr,
        depth_color,
        depth_mask,
        alpha=args.alpha,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix

    outputs = {
        "intensity_color": args.out_dir / f"{prefix}_intensity_color.png",
        "intensity_overlay": args.out_dir / f"{prefix}_intensity_overlay.png",
        "depth_color": args.out_dir / f"{prefix}_depth_color.png",
        "depth_overlay": args.out_dir / f"{prefix}_depth_overlay.png",
        "mask": args.out_dir / f"{prefix}_mask.png",
        "intensity_legend": args.out_dir / f"{prefix}_intensity_legend.png",
        "depth_legend": args.out_dir / f"{prefix}_depth_legend.png",
    }

    cv2.imwrite(str(outputs["intensity_color"]), intensity_color)
    cv2.imwrite(str(outputs["intensity_overlay"]), intensity_overlay)
    cv2.imwrite(str(outputs["depth_color"]), depth_color)
    cv2.imwrite(str(outputs["depth_overlay"]), depth_overlay)
    cv2.imwrite(str(outputs["mask"]), intensity_mask)

    intensity_legend = make_colorbar(
        height=260,
        width=220,
        reverse=False,
        top_label="High intensity",
        bottom_label="Low intensity",
    )
    depth_legend = make_colorbar(
        height=260,
        width=220,
        reverse=True,
        top_label=f"Near ({args.depth_min:g} m)",
        bottom_label=f"Far ({args.depth_max:g} m)",
    )
    cv2.imwrite(str(outputs["intensity_legend"]), intensity_legend)
    cv2.imwrite(str(outputs["depth_legend"]), depth_legend)

    print(f"loaded points      : {xyz.shape[0]}")
    print(f"visible image pixels: {u.size}")
    if intensity_visible.size:
        print(
            "intensity min/max  : "
            f"{float(np.min(intensity_visible)):.6f} / "
            f"{float(np.max(intensity_visible)):.6f}"
        )
    if camera_z.size:
        print(
            "Camera Z min/max   : "
            f"{float(np.min(camera_z)):.3f} / "
            f"{float(np.max(camera_z)):.3f} m"
        )

    print(
        f"intensity display  : mode={args.intensity_mode}, "
        f"bins={args.intensity_bins}"
    )
    print(
        f"depth display      : scale={args.depth_scale}, "
        f"gamma={args.depth_gamma}, bins={args.depth_bins}, "
        f"range={args.depth_min:g}-{args.depth_max:g} m"
    )

    for key, path in outputs.items():
        print(f"saved {key:18s}: {path}")


if __name__ == "__main__":
    main()