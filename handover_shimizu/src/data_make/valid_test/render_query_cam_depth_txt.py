#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 使い方:
#   python render_query_cam_depth_txt.py --help
#   必要な入力パスと出力先を引数で指定して実行する。


"""
render_query_cam_depth_txt.py

クエリ側（cam_gray/cross/.../q に書き出した png/json）について、
クエリ系列で作った静的点群地図（map_static.npz）を「クエリカメラの姿勢・intrinsic」で投影して

- ri_000.png（目視確認用）
- ri_000_index_depth.npz（point_index, depth）
- ri_000.json（メタ）
- overlay_cam_ri.png（任意、実カメラ画像との重ね合わせ）

を出力する。

レンダリング方式は make_loftr_shifted_ri_txt_final.py / render_anchor_intensity_txt.py に合わせる：
- 投影: x-forward (depth=Xc, U=-Yc, V=-Zc), u,v は roundせず astype(int32)
- z-buffer: far->near の順で上書き（近い点が勝つ）
- point_size=2 の場合は半径2pxのスプラット
- 後処理: (mask付き) ヒストグラム平坦化 -> CLAHE


"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# make_loftr_shifted_ri_txt_final.py に合わせた point_size=2 のオフセット
OFFSETS_R2 = [
    (0, 1), (1, 0), (0, -1), (-1, 0),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
    (0, 2), (2, 0), (0, -2), (-2, 0),
    (1, 2), (2, 1), (-1, 2), (-2, 1),
    (1, -2), (2, -1), (-1, -2), (-2, -1),
    (2, 2), (2, -2), (-2, 2), (-2, -2),
]


def parse_bool(x: str) -> bool:
    """CLI などで受け取った真偽値表現を bool に正規化する。"""
    s = str(x).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise ValueError(f"Invalid bool: {x}")


def read_text_lines(p: Path) -> List[str]:
    """空行やコメントを除いてテキスト行を読み込む。"""
    lines: List[str] = []
    with open(p, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
    return lines


def ensure_dir(p: Path) -> None:
    """出力先ディレクトリを作成して存在を保証する。"""
    p.mkdir(parents=True, exist_ok=True)


def load_json(p: Path) -> dict:
    """JSON ファイルを読み込んで Python オブジェクトとして返す。"""
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(p: Path, obj: dict) -> None:
    """Python オブジェクトを JSON ファイルとして保存する。"""
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def compute_up_value(intens: np.ndarray, pct: float) -> float:
    """姿勢や向きの判定に使う上方向の指標値を計算する。"""
    if intens.size == 0:
        return 1.0
    up = float(np.percentile(intens, pct))
    return max(up, 1e-6)


def masked_hist_equalize_uint8(img_u8: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """mask=True の画素だけヒストグラム平坦化。mask外は 0 のまま。"""
    if img_u8.dtype != np.uint8:
        raise ValueError("img_u8 must be uint8")
    if mask is None:
        return cv2.equalizeHist(img_u8)

    mask = mask.astype(bool)
    out = img_u8.copy()

    vals = img_u8[mask]
    if vals.size < 16:
        out[~mask] = 0
        return out

    hist = np.bincount(vals, minlength=256).astype(np.float64)
    cdf = hist.cumsum()
    nz = np.nonzero(hist)[0]
    if nz.size == 0:
        out[~mask] = 0
        return out

    cdf_min = cdf[nz[0]]
    cdf_max = cdf[-1]
    if cdf_max - cdf_min < 1e-12:
        out[~mask] = 0
        return out

    lut = np.floor((cdf - cdf_min) / (cdf_max - cdf_min) * 255.0 + 0.5)
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    out[mask] = lut[vals]
    out[~mask] = 0
    return out


def apply_clahe_masked_uint8(img_u8: np.ndarray, mask: Optional[np.ndarray], clip_limit: float, tile_grid: int) -> np.ndarray:
    """CLAHE を全体に適用してから mask外を 0 にする（簡単で安定）。"""
    if img_u8.dtype != np.uint8:
        raise ValueError("img_u8 must be uint8")
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid), int(tile_grid)))
    out = clahe.apply(img_u8)
    if mask is not None:
        mask = mask.astype(bool)
        out[~mask] = 0
    return out


def project_points_world_to_image(
    xyz_world: np.ndarray,
    T_wc: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    world->camera (T_wc) で点群をカメラ座標へ変換し、Waymoの軸定義(x-forward)に合わせて投影する。

    Waymoカメラ座標（想定）:
      X forward, Y left, Z up
    画像座標:
      u right, v down

    変換:
      depth = Xc
      U = -Yc
      V = -Zc

    ※ render_anchor_intensity_txt.py に合わせて roundは使わず astype(int32)
    """
    R = T_wc[:3, :3].astype(np.float32)
    t = T_wc[:3, 3].astype(np.float32)

    Xc = (xyz_world.astype(np.float32) @ R.T) + t[None, :]
    Z = Xc[:, 0]      # forward
    U = -Xc[:, 1]     # right
    V = -Xc[:, 2]     # down

    front = (Z > 0.1) & np.isfinite(Z)
    Z_safe = np.where(front, Z, 1.0)

    u = fx * (U / Z_safe) + cx
    v = fy * (V / Z_safe) + cy

    u = u.astype(np.int32)
    v = v.astype(np.int32)

    depth = Z.astype(np.float32)
    return u, v, depth, front


def rasterize_zbuffer(
    W: int,
    H: int,
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    base_norm_pts: np.ndarray,
    pt_idx: np.ndarray,
    point_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    点群を画像へ z-buffer で描画し、point_size=2 の場合は半径2pxスプラット（近い点優先）。

    戻り:
      base_map: (H,W) float32 [0,1]
      depth_map:(H,W) float32
      idx_map:  (H,W) int32  (pixel->元点群index, 無ければ -1)
    """
    base_map = np.zeros((H, W), dtype=np.float32)
    depth_map = np.full((H, W), np.inf, dtype=np.float32)
    idx_map = -np.ones((H, W), dtype=np.int32)

    order = np.argsort(depth)[::-1]  # far -> near（near が後に来て上書きされる）
    uo = u[order]
    vo = v[order]
    do = depth[order].astype(np.float32)
    bo = base_norm_pts[order].astype(np.float32)
    io = pt_idx[order].astype(np.int32)

    # 中心画素
    base_map[vo, uo] = bo
    depth_map[vo, uo] = do
    idx_map[vo, uo] = io

    if int(point_size) == 2:
        for du, dv in OFFSETS_R2:
            uu = uo + int(du)
            vv = vo + int(dv)
            m = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            if not np.any(m):
                continue
            uu2 = uu[m]
            vv2 = vv[m]
            base_map[vv2, uu2] = bo[m]
            depth_map[vv2, uu2] = do[m]
            idx_map[vv2, uu2] = io[m]

    return base_map, depth_map, idx_map


@dataclass
class CamParams:
    """レンダリングに使うカメラ内部・外部パラメータをまとめる。"""
    W: int
    H: int
    fx: float
    fy: float
    cx: float
    cy: float
    Twc: np.ndarray


def load_cam_params_from_export_json(p: Path) -> CamParams:
    """
    cross/q の 00007.json 形式（cam_intrinsic, camera_image_shape, world_to_cam）も、
    training の *_cam.json 形式（intrinsic, width, height, world_to_cam）も吸収する。
    """
    d = load_json(p)

    # サイズ情報
    if "camera_image_shape" in d and isinstance(d["camera_image_shape"], list) and len(d["camera_image_shape"]) >= 2:
        H = int(d["camera_image_shape"][0])
        W = int(d["camera_image_shape"][1])
    else:
        W = int(d.get("width", 1920))
        H = int(d.get("height", 1280))

    # 内部パラメータ
    if "intrinsic" in d and isinstance(d["intrinsic"], list) and len(d["intrinsic"]) >= 4:
        fx, fy, cx, cy = map(float, d["intrinsic"][:4])
    elif "cam_intrinsic" in d and isinstance(d["cam_intrinsic"], list) and len(d["cam_intrinsic"]) >= 4:
        fx, fy, cx, cy = map(float, d["cam_intrinsic"][:4])
    else:
        raise KeyError(f"Cannot find intrinsic in {p} (expected 'intrinsic' or 'cam_intrinsic').")

    # world_to_cam 行列
    if "world_to_cam" not in d:
        raise KeyError(f"Cannot find 'world_to_cam' in {p}.")
    Twc = np.array(d["world_to_cam"], dtype=np.float32)
    if Twc.shape != (4, 4):
        raise ValueError(f"world_to_cam shape must be (4,4), got {Twc.shape} in {p}")

    return CamParams(W=W, H=H, fx=fx, fy=fy, cx=cx, cy=cy, Twc=Twc)


def resolve_map_static_npz(maps_root: Path, segment: str, subset_hint: str) -> Optional[Path]:
    """
    maps_root/{training|validation}/{segment}/map_static.npz を探す。
    subset_hint が無ければ両方探す。
    """
    cand: List[Path] = []
    if subset_hint in ("training", "validation"):
        cand.append(maps_root / subset_hint / segment / "map_static.npz")
    cand.append(maps_root / "training" / segment / "map_static.npz")
    cand.append(maps_root / "validation" / segment / "map_static.npz")

    for p in cand:
        if p.exists():
            return p
    return None


def load_map_static(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """静的地図 JSON を読み込んで利用しやすい形へ整える。"""
    z = np.load(npz_path)
    if "xyz" not in z.files or "intensity" not in z.files:
        raise KeyError(f"{npz_path} must contain keys xyz and intensity, got {z.files}")
    xyz = z["xyz"].astype(np.float32)
    intensity = z["intensity"].astype(np.float32)
    return xyz, intensity


def read_gray_png(p: Path) -> Optional[np.ndarray]:
    """グレースケール PNG を NumPy 配列として読み込む。"""
    if not p.exists():
        return None
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return img


def make_overlay(cam_gray: np.ndarray, ri_u8: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    cam_gray と ri_u8 を同サイズ前提で重ねる。
    """
    a = float(alpha)
    a = max(0.0, min(1.0, a))
    cam = cam_gray.astype(np.float32)
    ri = ri_u8.astype(np.float32)
    out = (a * cam + (1.0 - a) * ri)
    return np.clip(out, 0, 255).astype(np.uint8)


def render_one_frame(
    xyz_world_all: np.ndarray,
    intensity_all: np.ndarray,
    up_val: float,
    cam: CamParams,
    W: int,
    H: int,
    out_dir: Path,
    cam_png_path: Path,
    src_json_path: Path,
    args: argparse.Namespace,
) -> Optional[dict]:
    """
    1フレーム分レンダリングして保存。meta を返す（失敗時None）。
    """
    # 投影
    u, v, depth, front = project_points_world_to_image(xyz_world_all, cam.Twc, cam.fx, cam.fy, cam.cx, cam.cy)

    ok = (
        front
        & (u >= 0) & (u < W)
        & (v >= 0) & (v < H)
        & np.isfinite(depth)
        & (depth >= float(args.min_depth))
        & (depth <= float(args.max_depth))
    )
    if not np.any(ok):
        return None

    u2 = u[ok]
    v2 = v[ok]
    depth_k = depth[ok]
    intens_k = intensity_all[ok]
    idx_k = np.where(ok)[0].astype(np.int32)

    # 距離補正（必要なら）
    if abs(float(args.range_corr_power)) > 1e-9:
        intens_k = intens_k * np.power(depth_k, float(args.range_corr_power))

    # 正規化（0..1）
    val01 = np.clip(intens_k / (up_val + 1e-6), 0.0, 1.0).astype(np.float32)

    base_map, depth_map, idx_map = rasterize_zbuffer(
        W, H, u2, v2, depth_k, val01, idx_k, int(args.point_size)
    )

    covered = (idx_map >= 0)
    coverage = float(np.mean(covered))

    # RI png（目視確認用）
    ri_png_path = out_dir / "ri_000.png"
    ri_json_path = out_dir / "ri_000.json"
    ri_idx_path = out_dir / "ri_000_index_depth.npz"
    overlay_path = out_dir / "overlay_cam_ri.png"

    if parse_bool(args.write_ri):
        img_u8 = np.clip((base_map * 255.0).round(), 0, 255).astype(np.uint8)

        # 後処理（histeq -> clahe）
        if parse_bool(args.post_mask_covered_only):
            mask_pp = covered
        else:
            mask_pp = None

        if parse_bool(args.post_hist_eq):
            img_u8 = masked_hist_equalize_uint8(img_u8, mask_pp)
        if parse_bool(args.post_clahe):
            img_u8 = apply_clahe_masked_uint8(img_u8, mask_pp, float(args.clahe_clip_limit), int(args.clahe_tile_grid))

        cv2.imwrite(str(ri_png_path), img_u8)

        # overlay を作る（任意）
        if parse_bool(args.write_overlay):
            cam_gray = read_gray_png(cam_png_path)
            if cam_gray is not None and cam_gray.shape[0] == H and cam_gray.shape[1] == W:
                ov = make_overlay(cam_gray, img_u8, alpha=float(args.overlay_alpha))
                cv2.imwrite(str(overlay_path), ov)

    # index/depth の NPZ
    if parse_bool(args.write_index):
        np.savez_compressed(
            ri_idx_path,
            point_index=idx_map.astype(np.int32),
            depth=depth_map.astype(np.float32),
        )

    meta = {
        "segment_id": str(args._current_segment),
        "subset": str(args._current_subset),
        "camera_name": str(args.cam).upper(),
        "frame_id": str(out_dir.name),
        "source": {
            "cam_export_png": str(cam_png_path),
            "cam_export_json": str(src_json_path),
            "map_static_npz": str(args._current_map_npz),
        },
        "projection": {
            "convention": "x-forward",
            "camera_axes": "X forward, Y left, Z up",
            "image_axes": "u right, v down",
            "depth_is": "Xc",
            "u_is": "-Yc",
            "v_is": "-Zc",
            "rounding": "astype(int32) (no np.round)",
        },
        "width": int(W),
        "height": int(H),
        "intrinsic": [float(cam.fx), float(cam.fy), float(cam.cx), float(cam.cy)],
        "world_to_cam": cam.Twc.astype(float).tolist(),
        "coverage": float(coverage),
        "render_profile": {
            "int_percentile": float(args.int_percentile),
            "global_up_val": float(up_val),
            "range_corr_power": float(args.range_corr_power),
            "point_size": int(args.point_size),
            "postprocess": {
                "hist_eq": bool(parse_bool(args.post_hist_eq)),
                "clahe": bool(parse_bool(args.post_clahe)),
                "clahe_clip_limit": float(args.clahe_clip_limit),
                "clahe_tile_grid": int(args.clahe_tile_grid),
                "mask_covered_only": bool(parse_bool(args.post_mask_covered_only)),
            },
        },
        "outputs": {
            "ri_png": str(ri_png_path) if parse_bool(args.write_ri) else None,
            "ri_json": str(ri_json_path),
            "ri_index_depth_npz": str(ri_idx_path) if parse_bool(args.write_index) else None,
            "overlay_png": str(overlay_path) if parse_bool(args.write_overlay) else None,
        },
    }

    save_json(ri_json_path, meta)
    return meta


def main() -> None:
    """CLI 引数を解釈し、入力の読み込みから結果保存までの一連の処理を実行する。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments-file", type=str, required=True, help="1行1segment のtxt")
    ap.add_argument("--cam-root", type=str, required=True, help="cam_gray/cross/.../q のroot")
    ap.add_argument("--maps-root", type=str, required=True, help="map_static.npz のroot（training/validation配下）")
    ap.add_argument("--out-root", type=str, required=True, help="出力root（training/validation配下に保存）")
    ap.add_argument("--cam", type=str, default="FRONT", help="カメラ名（例 FRONT）")

    ap.add_argument("--min-depth", type=float, default=0.5)
    ap.add_argument("--max-depth", type=float, default=200.0)

    ap.add_argument("--int-percentile", type=float, default=99.5)
    ap.add_argument("--range-corr-power", type=float, default=0.0)
    ap.add_argument("--point-size", type=int, default=2)

    ap.add_argument("--post-hist-eq", type=str, default="true")
    ap.add_argument("--post-clahe", type=str, default="true")
    ap.add_argument("--post-mask-covered-only", type=str, default="true")
    ap.add_argument("--clahe-clip-limit", type=float, default=2.0)
    ap.add_argument("--clahe-tile-grid", type=int, default=8)

    ap.add_argument("--write-ri", type=str, default="true", help="ri_000.png を保存")
    ap.add_argument("--write-index", type=str, default="true", help="ri_000_index_depth.npz を保存")
    ap.add_argument("--write-overlay", type=str, default="true", help="overlay_cam_ri.png を保存")
    ap.add_argument("--overlay-alpha", type=float, default=0.5)

    ap.add_argument("--skip-existing", type=str, default="true", help="出力が揃っていればスキップ")
    ap.add_argument("--write-manifest", type=str, default="false", help="処理結果CSVを書く")
    ap.add_argument("--manifest-out", type=str, default="", help="manifest出力先")

    args = ap.parse_args()

    segments_file = Path(args.segments_file)
    cam_root = Path(args.cam_root)
    maps_root = Path(args.maps_root)
    out_root = Path(args.out_root)
    cam_name = str(args.cam).upper()

    print("[BOOT] start")
    print(f"[INFO] segments-file = {segments_file}")
    print(f"[INFO] cam-root      = {cam_root}")
    print(f"[INFO] maps-root     = {maps_root}")
    print(f"[INFO] out-root      = {out_root}")
    print(f"[INFO] cam           = {cam_name}")
    print(f"[INFO] write-ri      = {parse_bool(args.write_ri)}")
    print(f"[INFO] write-index   = {parse_bool(args.write_index)}")
    print(f"[INFO] write-overlay = {parse_bool(args.write_overlay)}")

    segs = read_text_lines(segments_file)
    print(f"[INFO] segments n={len(segs)}")

    # manifest を読む
    manifest_rows: List[dict] = []
    want_manifest = parse_bool(args.write_manifest)
    if want_manifest and not args.manifest_out:
        raise ValueError("--write-manifest true のときは --manifest-out を指定してください。")

    for seg in segs:
        # subsetは cam_root 側で存在する方を回す（training/validation）
        subsets: List[str] = []
        for sub in ("training", "validation"):
            if (cam_root / sub / seg).exists():
                subsets.append(sub)
        if not subsets:
            print(f"[WARN] segment not found in cam-root: {seg}")
            continue

        # map_static.npz は subsetごとに探すが、無ければ反対側も探す
        for subset in subsets:
            map_npz = resolve_map_static_npz(maps_root, seg, subset)
            if map_npz is None:
                print(f"[WARN] map_static.npz not found for segment={seg} subset_hint={subset}")
                continue

            print("================================================================================")
            print(f"[INFO] segment={seg} subset={subset}")
            print(f"[INFO] map_static={map_npz}")

            # map load（巨大なのでsegmentごとに1回）
            xyz_world_all, intensity_all = load_map_static(map_npz)
            up_val = compute_up_value(intensity_all, float(args.int_percentile))
            print(f"[INFO] map points={xyz_world_all.shape[0]} | up_val(p{args.int_percentile})={up_val:.6f}")

            in_cam_dir = cam_root / subset / seg / cam_name
            if not in_cam_dir.exists():
                print(f"[WARN] camera dir not found: {in_cam_dir}")
                continue

            json_list = sorted(in_cam_dir.glob("*.json"))
            if not json_list:
                print(f"[WARN] no json found in: {in_cam_dir}")
                continue
            print(f"[INFO] frames(json)={len(json_list)} in {in_cam_dir}")

            out_seg_dir = out_root / subset / seg / cam_name
            ensure_dir(out_seg_dir)

            # meta 用に現在の文脈を退避する
            args._current_segment = seg
            args._current_subset = subset
            args._current_map_npz = str(map_npz)

            for jpath in json_list:
                frame_id = jpath.stem  # "00007"
                cam_png = in_cam_dir / f"{frame_id}.png"

                out_dir = out_seg_dir / frame_id
                ensure_dir(out_dir)

                ri_png_path = out_dir / "ri_000.png"
                ri_json_path = out_dir / "ri_000.json"
                ri_idx_path = out_dir / "ri_000_index_depth.npz"
                overlay_path = out_dir / "overlay_cam_ri.png"

                done = True
                if parse_bool(args.write_ri):
                    done = done and ri_png_path.exists()
                done = done and ri_json_path.exists()
                if parse_bool(args.write_index):
                    done = done and ri_idx_path.exists()
                if parse_bool(args.write_overlay):
                    done = done and overlay_path.exists()

                if parse_bool(args.skip_existing) and done:
                    continue

                # カメラパラメータを読み込む
                try:
                    cam_params = load_cam_params_from_export_json(jpath)
                except Exception as e:
                    print(f"[ERR] cam json parse failed: {jpath} | {type(e).__name__}: {e}")
                    continue

                W = int(cam_params.W)
                H = int(cam_params.H)

                # 描画する
                meta = render_one_frame(
                    xyz_world_all=xyz_world_all,
                    intensity_all=intensity_all,
                    up_val=up_val,
                    cam=cam_params,
                    W=W,
                    H=H,
                    out_dir=out_dir,
                    cam_png_path=cam_png,
                    src_json_path=jpath,
                    args=args,
                )
                if meta is None:
                    print(f"[WARN] no valid projection: {seg} {subset} frame={frame_id}")
                    continue

                if want_manifest:
                    manifest_rows.append({
                        "segment_id": seg,
                        "subset": subset,
                        "camera_name": cam_name,
                        "frame_id": frame_id,
                        "cam_png": str(cam_png),
                        "cam_json": str(jpath),
                        "map_static_npz": str(map_npz),
                        "ri_png": meta["outputs"]["ri_png"],
                        "ri_json": meta["outputs"]["ri_json"],
                        "ri_index_depth_npz": meta["outputs"]["ri_index_depth_npz"],
                        "overlay_png": meta["outputs"]["overlay_png"],
                        "coverage": meta["coverage"],
                    })

    if want_manifest:
        mout = Path(args.manifest_out)
        ensure_dir(mout.parent)
        cols = [
            "segment_id", "subset", "camera_name", "frame_id",
            "cam_png", "cam_json", "map_static_npz",
            "ri_png", "ri_json", "ri_index_depth_npz", "overlay_png",
            "coverage"
        ]
        with open(mout, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in manifest_rows:
                w.writerow(r)
        print(f"[OK] manifest written: {mout} rows={len(manifest_rows)}")

    print("[DONE]")


if __name__ == "__main__":
    main()
