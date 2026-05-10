#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 使い方:
#   python render_anchor_intensity_txt.py --help
#   必要な入力パスと出力先を引数で指定して実行する。

"""
Step 3: サブマップ・アンカー反射強度画像（RI）の生成

やること
- make_static_map.py が作った map_static.npz（セグメント全フレーム重畳点群）を読む
- make_submaps_distance.py が作った submaps.json を読む
- 各 submap_id（アンカー）ごとに、点群を投影して RI 画像を作る（z-buffer + point splat）
- 画像の後処理として「ヒストグラム平坦化 → CLAHE」を適用（デフォルトON）
- anchor.png / anchor.json / anchor_index.npz を保存

レジューム対応（今回の追加）:
- サブマップごとに anchor.png / anchor.json（および必要なら anchor_index.npz）が存在していれば、
  --overwrite false のときスキップする。


"""

import sys
import json
import argparse
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2 as open_dataset


# txt内がstemの場合に探索する Waymo individual_files のルート（例: <root>/<subset>/<stem>.tfrecord）
# 単一tfrecordを渡す従来運用には影響しない（txt入力のときだけ使用する）。
DEFAULT_TFRECORD_ROOT = "/mnt/e/waymo_tf/perception_v1.4.3/individual_files"


# 半径2pxのオフセット（中心以外の12点）
OFFSETS_R2 = [
    (-2, 0), (-1, -1), (-1, 0), (-1, 1),
    (0, -2), (0, -1),  (0, 1),  (0, 2),
    (1, -1), (1, 0),   (1, 1),  (2, 0),
]


def str2bool(s: str) -> bool:
    """CLI などで受け取った真偽値表現を bool に正規化する。"""
    return str(s).strip().lower() in ["1", "true", "t", "yes", "y"]


def ensure_dir(p: Path):
    """出力先ディレクトリを作成して存在を保証する。"""
    p.mkdir(parents=True, exist_ok=True)


def _read_nonempty_lines(txt_path: Path) -> List[str]:
    """txtを1行ずつ読み、空行・コメント行（#で開始）を除外して返す（順序維持で重複除去）。"""
    lines: List[str] = []
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for raw in f:
                s = raw.strip()
                if not s:
                    continue
                if s.startswith("#"):
                    continue
                lines.append(s)
    except UnicodeDecodeError:
        # 万一BOM付きなどだった場合の保険
        with open(txt_path, "r", encoding="utf-8-sig") as f:
            for raw in f:
                s = raw.strip()
                if not s:
                    continue
                if s.startswith("#"):
                    continue
                lines.append(s)

    uniq: List[str] = []
    seen = set()
    for s in lines:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def _looks_like_tfrecord_path(s: str) -> bool:
    """文字列が TFRecord パス表記かどうかを判定する。"""
    ls = s.lower()
    return ls.endswith(".tfrecord") or ls.endswith(".tfrecord.gz")


def resolve_tfrecord_path(
    item: str,
    subset: str,
    tfrecord_root: Path,
    txt_dir: Optional[Path] = None,
) -> Path:
    """
    txt内の1行（item）を tfrecord の実パスに解決する。

    解決順:
      1) item が絶対パスで存在
      2) txt_dir/item が存在（相対パス対応）
      3) item が相対パスで存在（カレントディレクトリ基準）
      4) stem とみなして <tfrecord_root>/<subset>/<stem>.tfrecord を探索
      5) item が .tfrecord 名（存在しない）なら <tfrecord_root>/<subset>/<name> を探索
    """
    s = item.strip()
    if not s:
        raise FileNotFoundError("Empty line")

    p = Path(s).expanduser()

    # 1) 絶対パスで存在
    if p.is_absolute() and p.exists():
        return p

    # 2) txt_dir基準の相対パス
    if txt_dir is not None:
        cand = (txt_dir / p).expanduser()
        if cand.exists():
            return cand.resolve()

    # 3) カレントディレクトリ基準の相対パス
    if p.exists():
        return p.resolve()

    # 4) stem から構築（.tfrecord 付与）
    tfroot = Path(tfrecord_root)
    if _looks_like_tfrecord_path(s):
        name = Path(s).name
    else:
        name = f"{s}.tfrecord"

    cand2 = tfroot / subset / name
    if cand2.exists():
        return cand2.resolve()

    raise FileNotFoundError(
        "tfrecord が見つかりません。\n"
        f"  item: {item}\n"
        f"  tried: {p}\n"
        f"  tried: {cand2}\n"
        "  ヒント: txtにフルパスを書くか、--tfrecord-root が正しいか確認してください。"
    )


def mat4_list_to_np(m) -> np.ndarray:
    """行優先の 16 要素配列を 4x4 の NumPy 行列に変換する。"""
    return np.array(m, dtype=np.float64).reshape(4, 4)


def camera_name_from_str(name: str) -> int:
    """文字列から Waymo のカメラ種別 ID を解決する。"""
    nm = name.strip().upper()
    if not hasattr(open_dataset.CameraName, nm):
        raise ValueError(f"Unknown camera name: {name}")
    return getattr(open_dataset.CameraName, nm)


def intrinsic_from(calib):
    """キャリブレーションやメタ情報から内部パラメータを取り出す。"""
    intr = list(calib.intrinsic)
    if len(intr) < 4:
        raise ValueError("intrinsic has less than 4 elements")
    fx, fy, cx, cy = float(intr[0]), float(intr[1]), float(intr[2]), float(intr[3])
    return fx, fy, cx, cy


def world_to_cam_from(frame, calib) -> np.ndarray:
    """
    既存実装に合わせる:
      frame.pose.transform      : world<-vehicle
      calib.extrinsic.transform : vehicle<-cam
    よって
      world->cam = (cam<-vehicle) @ (vehicle<-world)
               = inv(T_vc) @ inv(T_wv)
    """
    T_wv = mat4_list_to_np(frame.pose.transform)           # world<-vehicle
    T_vw = np.linalg.inv(T_wv)                             # vehicle<-world
    T_vc = mat4_list_to_np(calib.extrinsic.transform)      # vehicle<-cam
    T_cv = np.linalg.inv(T_vc)                             # cam<-vehicle
    T_wc = T_cv @ T_vw                                     # world -> cam
    return T_wc


def project_points_world_to_image(xyz_world: np.ndarray, Twc: np.ndarray, K: np.ndarray):
    """
    既存実装に合わせた投影（Waymo座標系の“軸の入れ替え”がある点に注意）:

      pts_c = (world_pts @ Twc^T)[:,:3]  # [Xc, Yc, Zc] in camera frame
      Z = Xc (>0) / U = -Yc / V = -Zc
      u = fx * (U/Z) + cx, v = fy * (V/Z) + cy

    戻り値:
      u(int32), v(int32), depth(Z=float32), front_mask(bool)
    """
    N = xyz_world.shape[0]
    if N == 0:
        return None

    pts_h = np.hstack([xyz_world, np.ones((N, 1), dtype=xyz_world.dtype)])  # (N,4)
    pts_c = (pts_h @ Twc.T)[:, :3]                                          # (N,3)

    Xc, Yc, Zc = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]
    Z = Xc
    U = -Yc
    V = -Zc

    front = Z > 0.1
    if not np.any(front):
        return None

    U = U[front]
    V = V[front]
    Z = Z[front].astype(np.float32)

    uvw = (K @ np.stack([U, V, Z], axis=0)).T
    u = (uvw[:, 0] / uvw[:, 2]).astype(np.int32)
    v = (uvw[:, 1] / uvw[:, 2]).astype(np.int32)
    return u, v, Z, front


def compute_up_value(intens: np.ndarray, pct: float) -> float:
    """姿勢や向きの判定に使う上方向の指標値を計算する。"""
    if intens.size == 0:
        return 1.0
    up = float(np.percentile(intens, pct))
    return max(up, 1e-6)


def apply_gamma_clip(intens_norm01: np.ndarray, gamma: float) -> np.ndarray:
    """ガンマ補正と値域クリップをまとめて適用する。"""
    if intens_norm01.size == 0:
        return intens_norm01
    x = np.clip(intens_norm01, 0.0, 1.0)
    x = np.power(x, gamma)
    return np.clip(x, 0.0, 1.0)


def get_camera_gray(frame, cam_name: int) -> np.ndarray:
    """Waymo Frame から指定カメラの JPEG を取り出し、H×Wのuint8グレースケールを返す。"""
    img_rec = next((im for im in frame.images if im.name == cam_name), None)
    if img_rec is None:
        return None
    buf = np.frombuffer(img_rec.image, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray


def percentile_pair(x: np.ndarray, lo: float, hi: float) -> Tuple[float, float]:
    """下限と上限のパーセンタイル値を組で計算する。"""
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))


def masked_hist_equalize_uint8(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    mask==True の画素だけでヒストグラム平坦化（global）。
    mask==False は 0 に戻す（背景を黒固定）。
    """
    if img.dtype != np.uint8:
        raise ValueError("masked_hist_equalize_uint8 expects uint8 image")
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)

    out = img.copy()
    vals = img[mask]
    if vals.size == 0:
        out[~mask] = 0
        return out

    hist = np.bincount(vals, minlength=256).astype(np.float64)
    cdf = hist.cumsum()
    nonzero = hist > 0
    if not np.any(nonzero):
        out[~mask] = 0
        return out

    cdf_min = cdf[nonzero][0]
    cdf_max = cdf[-1]
    if cdf_max <= cdf_min + 1e-12:
        out[~mask] = 0
        return out

    lut = np.floor((cdf - cdf_min) / (cdf_max - cdf_min) * 255.0 + 0.5)
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    out[mask] = lut[vals]
    out[~mask] = 0
    return out


def apply_clahe_uint8(img: np.ndarray, mask: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    """
    CLAHE を適用し、mask==False の画素は 0 に戻す。
    """
    if img.dtype != np.uint8:
        raise ValueError("apply_clahe_uint8 expects uint8 image")
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)

    tg = int(tile_grid)
    if tg <= 0:
        tg = 8
    cl = float(clip_limit)
    if cl <= 0:
        cl = 2.0

    clahe = cv2.createCLAHE(clipLimit=cl, tileGridSize=(tg, tg))
    out = clahe.apply(img)
    out[~mask] = 0
    return out


def is_anchor_done(out_dir: Path, need_index: bool) -> bool:
    """
    そのサブマップの anchor 出力が揃っているかを判定する。
      - anchor.png
      - anchor.json
      - need_index=True のときは anchor_index.npz
    """
    img_path = out_dir / "anchor.png"
    json_path = out_dir / "anchor.json"
    if not img_path.exists() or not json_path.exists():
        return False
    if need_index:
        idx_path = out_dir / "anchor_index.npz"
        if not idx_path.exists():
            return False
    return True


def _process_one_tfrecord(tfrecord_path: Path, args: argparse.Namespace, strict: bool) -> str:
    """単一tfrecordを従来通り処理する。list mode のときは strict=False で呼び出す。"""

    seg_stem = tfrecord_path.stem

    maps_dir = Path(args.maps_root) / args.subset / seg_stem
    submaps_dir = Path(args.submaps-root if hasattr(args, 'submaps-root') else args.submaps_root) / args.subset / seg_stem  # 互換性保護
    # ↑ただし通常は args.submaps_root を使う
    submaps_dir = Path(args.submaps_root) / args.subset / seg_stem
    map_npz = maps_dir / "map_static.npz"
    submaps_json = submaps_dir / "submaps.json"

    if not tfrecord_path.exists():
        msg = f"tfrecord が見つかりません: {tfrecord_path}"
        if strict:
            raise FileNotFoundError(str(tfrecord_path))
        print(f"[ERR] {msg}")
        return "err"
    if not map_npz.exists():
        msg = f"map_static.npz が見つかりません: {map_npz}"
        if strict:
            raise FileNotFoundError(str(map_npz))
        print(f"[ERR] {msg}")
        return "err"
    if not submaps_json.exists():
        msg = f"submaps.json が見つかりません: {submaps_json}"
        if strict:
            raise FileNotFoundError(str(submaps_json))
        print(f"[ERR] {msg}")
        return "err"

    # 1) フレームをキャッシュ（カメラキャリブと姿勢を参照するため）
    frames = []
    for rec in tf.data.TFRecordDataset(str(tfrecord_path), compression_type=""):
        fr = open_dataset.Frame()
        fr.ParseFromString(rec.numpy())
        frames.append(fr)

    # 2) セグメントの静的マップ（全フレーム重畳点群）
    with np.load(map_npz) as z:
        xyz_world_all = z["xyz"]
        intensity_all = z["intensity"]

    with open(submaps_json, "r", encoding="utf-8") as f:
        submap_obj = json.load(f)

    submaps = submap_obj["submaps"]
    meta_sub = submap_obj.get("meta", {})
    default_r = float(meta_sub.get("submap_radius_m", 10.0))

    cam_name = camera_name_from_str(args.cam)
    print(
        f"[INFO] segment={seg_stem} | frames={len(frames)} | pts={xyz_world_all.shape[0]} | submaps={len(submaps)} | cam={args.cam}"
    )
    print(
        f"[INFO] crop={args.crop_mode}, r={args.crop_radius_m}, margin={args.crop_margin_m}, z=({args.z_min},{args.z_max}), mode={args.int_mode}"
    )
    print(
        f"[INFO] post: hist_eq={args.post_hist_eq}, clahe={args.post_clahe}, mask_covered_only={args.post_mask_covered_only}"
    )
    print(f"[INFO] overwrite={args.overwrite} (サブマップ単位で anchor 出力の再計算を制御)")

    # global up 値（必要時のみ）
    global_up = None
    if args.int_mode in ("global_map", "camera_match", "camera_blend"):
        global_up = compute_up_value(intensity_all, args.int_percentile)

    for sm in tqdm(submaps, desc="render anchors", unit="submap"):
        sub_id = int(sm["submap_id"])
        anchor_idx = int(sm["anchor_frame_index"])
        anchor_pose = sm["anchor_pose_xyyaw"]
        ax, ay = float(anchor_pose["x"]), float(anchor_pose["y"])

        # 出力ディレクトリとレジューム判定
        out_dir = submaps_dir / str(sub_id)
        if out_dir.exists():
            if (not args.overwrite) and is_anchor_done(out_dir, args.write_index):
                print(f"[SKIP] submap {sub_id}: 既に anchor.png / anchor.json が存在するためスキップします。")
                continue
            # 既存だが不完全 or overwrite=True の場合は上書き（ファイルはそのまま上書き保存）

        ensure_dir(out_dir)

        # --- クロップ（円筒 or なし） ---
        if args.crop_mode == "circle":
            r = default_r if str(args.crop_radius_m).lower() == "auto" else float(args.crop_radius_m)
            r_eff = float(r + args.crop_margin_m)
            dx = xyz_world_all[:, 0] - ax
            dy = xyz_world_all[:, 1] - ay
            mask_xy = (dx * dx + dy * dy) <= (r_eff * r_eff)
        else:
            mask_xy = np.ones(xyz_world_all.shape[0], dtype=bool)

        if args.z_min is not None or args.z_max is not None:
            zc = xyz_world_all[:, 2]
            mask_z = np.ones_like(mask_xy)
            if args.z_min is not None:
                mask_z &= (zc >= args.z_min)
            if args.z_max is not None:
                mask_z &= (zc <= args.z_max)
            mask = mask_xy & mask_z
        else:
            mask = mask_xy

        if not np.any(mask):
            print(f"[WARN] submap {sub_id}: no points after crop, skip")
            continue

        xyz_world = xyz_world_all[mask]
        intensity = intensity_all[mask]
        orig_indices = np.flatnonzero(mask)

        # --- カメラパラメータ（FRONT固定） ---
        frame = frames[anchor_idx]
        calib = next((c for c in frame.context.camera_calibrations if c.name == cam_name), None)
        if calib is None:
            print(f"[WARN] camera calib not found for submap {sub_id}, skip")
            continue

        W, H = int(calib.width), int(calib.height)
        fx, fy, cx, cy = intrinsic_from(calib)
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float64)
        Twc = world_to_cam_from(frame, calib)

        cam_gray = get_camera_gray(frame, cam_name) if args.int_mode in ("camera_match", "camera_blend") else None
        if args.int_mode in ("camera_match", "camera_blend") and cam_gray is None:
            print(f"[WARN] submap {sub_id}: camera image not available; fallback to global_map")
            cam_gray = None

        # --- 投影（まずフロント点だけ） ---
        proj = project_points_world_to_image(xyz_world, Twc, K)
        if proj is None:
            print(f"[WARN] submap {sub_id}: no front points, skip")
            continue

        u, v, depth, front_mask = proj
        intens_front = intensity[front_mask]
        front_indices = orig_indices[front_mask]

        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not np.any(ok):
            print(f"[WARN] submap {sub_id}: no in-image points, skip")
            continue

        u = u[ok]
        v = v[ok]
        depth = depth[ok]
        intens_fov = intens_front[ok]
        pt_idx_fov = front_indices[ok]

        # --- 距離補正（任意） ---
        if abs(args.range_corr_power) > 1e-9:
            intens_fov = intens_fov * np.power(depth, args.range_corr_power)

        # --- 正規化（pointごと）---
        if args.int_mode == "per_submap":
            up_val = compute_up_value(intens_fov, args.int_percentile)
        else:
            up_val = global_up if global_up is not None else compute_up_value(intens_fov, args.int_percentile)

        base_norm_pts = np.clip(intens_fov / (up_val + 1e-6), 0.0, 1.0)
        base_norm_pts = apply_gamma_clip(base_norm_pts, args.gamma)  # [0,1]

        # =========================
        # A) base_norm の z-buffer マップを作成（float32）
        # =========================
        base_map = np.zeros((H, W), dtype=np.float32)
        depth_map = np.full((H, W), np.inf, dtype=np.float32)
        idx_map = -np.ones((H, W), dtype=np.int32) if args.write_index else None

        # 遠→近で更新（最後に近が残る）
        order = np.argsort(depth)[::-1]
        uo, vo = u[order], v[order]
        do = depth[order]
        bo = base_norm_pts[order]
        io = pt_idx_fov[order]

        base_map[vo, uo] = bo
        depth_map[vo, uo] = do
        if idx_map is not None:
            idx_map[vo, uo] = io.astype(np.int32)

        if args.point_size > 1:
            for du, dv in OFFSETS_R2:
                uu, vv = uo + du, vo + dv
                sel = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
                if not np.any(sel):
                    continue
                nearer = do[sel] < depth_map[vv[sel], uu[sel]]
                if not np.any(nearer):
                    continue
                rr_u = uu[sel][nearer]
                rr_v = vv[sel][nearer]
                rr_d = do[sel][nearer]
                rr_b = bo[sel][nearer]
                rr_i = io[sel][nearer]
                base_map[rr_v, rr_u] = rr_b
                depth_map[rr_v, rr_u] = rr_d
                if idx_map is not None:
                    idx_map[rr_v, rr_u] = rr_i.astype(np.int32)

        # =========================
        # B) camera_match: LiDARカバー画素のみで係数推定
        # =========================
        used_mode = args.int_mode
        match_a = match_b = None
        out_map = base_map.copy()

        if args.int_mode in ("camera_match", "camera_blend") and cam_gray is not None:
            if args.match_covered_only and idx_map is not None:
                mask_cov = (idx_map >= 0)
            else:
                mask_cov = np.ones_like(base_map, dtype=bool)

            if np.any(mask_cov):
                cam_vals = (cam_gray.astype(np.float32) / 255.0)[mask_cov]
                lidar_vals = base_map[mask_cov]
                p_lo, p_hi = percentile_pair(lidar_vals, args.match_pct_lo, args.match_pct_hi)
                q_lo, q_hi = percentile_pair(cam_vals, args.match_pct_lo, args.match_pct_hi)
                denom = (p_hi - p_lo)
                if np.isfinite(denom) and denom > 1e-6:
                    a = (q_hi - q_lo) / denom
                    b = q_lo - a * p_lo
                    match_a, match_b = float(a), float(b)
                    matched = np.clip(a * base_map + b, 0.0, 1.0)
                    if args.int_mode == "camera_match":
                        out_map = matched
                    else:
                        alpha = float(np.clip(args.match_blend, 0.0, 1.0))
                        out_map = np.clip((1.0 - alpha) * base_map + alpha * matched, 0.0, 1.0)
                else:
                    used_mode = "global_map" if args.int_mode == "camera_match" else "camera_blend(fallback->base)"
            else:
                used_mode = "global_map" if args.int_mode == "camera_match" else "camera_blend(fallback->base)"

        # --- 8bit へ ---
        img = np.clip((out_map * 255.0).round(), 0, 255).astype(np.uint8)

        # --- 後処理: ヒストグラム平坦化 → CLAHE ---
        if args.post_hist_eq or args.post_clahe:
            if args.post_mask_covered_only:
                if idx_map is not None:
                    mask_pp = (idx_map >= 0)
                else:
                    mask_pp = (img > 0)
            else:
                mask_pp = np.ones_like(img, dtype=bool)

            if args.post_hist_eq:
                img = masked_hist_equalize_uint8(img, mask_pp)
            if args.post_clahe:
                img = apply_clahe_uint8(img, mask_pp, args.clahe_clip_limit, args.clahe_tile_grid)

        # --- 保存 ---
        img_path = out_dir / "anchor.png"
        json_path = out_dir / "anchor.json"
        cv2.imwrite(str(img_path), img)

        if args.write_index:
            np.savez_compressed(
                out_dir / "anchor_index.npz",
                point_index=idx_map if idx_map is not None else np.full((H, W), -1, np.int32),
                depth=depth_map
            )

        coverage = float(np.mean((idx_map >= 0))) if (args.write_index and idx_map is not None) else float(np.mean(base_map > 0))

        meta = {
            "segment_id": seg_stem,
            "subset": args.subset,
            "submap_id": sub_id,
            "anchor_frame_index": anchor_idx,
            "anchor_pose_xyyaw": anchor_pose,
            "camera_name": open_dataset.CameraName.Name.Name(cam_name),
            "image_path": str(img_path),
            "width": int(W),
            "height": int(H),
            "intrinsic": [float(fx), float(fy), float(cx), float(cy)],
            "world_to_cam": Twc.tolist(),
            "coverage": coverage,
            "render_profile": {
                "int_mode": used_mode,
                "int_percentile": float(args.int_percentile),
                "intensity_up_val": float(up_val),
                "gamma": float(args.gamma),
                "range_corr_power": float(args.range_corr_power),
                "point_size": int(args.point_size),
                "crop_mode": args.crop_mode,
                "crop_radius_m": (default_r if str(args.crop_radius_m).lower() == "auto" else float(args.crop_radius_m)),
                "crop_margin_m": float(args.crop_margin_m),
                "z_min": None if args.z_min is None else float(args.z_min),
                "z_max": None if args.z_max is None else float(args.z_max),
                "camera_match": {
                    "match_pct_lo": float(args.match_pct_lo),
                    "match_pct_hi": float(args.match_pct_hi),
                    "match_blend": float(args.match_blend),
                    "match_covered_only": bool(args.match_covered_only),
                    "a": None if match_a is None else float(match_a),
                    "b": None if match_b is None else float(match_b),
                },
                "postprocess": {
                    "hist_equalize": bool(args.post_hist_eq),
                    "clahe": bool(args.post_clahe),
                    "clahe_clip_limit": float(args.clahe_clip_limit),
                    "clahe_tile_grid": int(args.clahe_tile_grid),
                    "mask_covered_only": bool(args.post_mask_covered_only),
                },
            }
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] anchors rendered (resume-aware): {len(submaps)} -> {submaps_dir}")
    return "ok"


def main():
    """CLI 引数を解釈し、入力の読み込みから結果保存までの一連の処理を実行する。"""
    ap = argparse.ArgumentParser(description="Step 3: アンカー反射強度画像（RI）を生成（FRONT固定 + hist_eq→CLAHE）")

    ap.add_argument(
        "--tfrecord",
        type=str,
        required=True,
        help=(
            "Waymo segment .tfrecord または tfrecordリスト .txt（1行1セグメント）。"
            " txtの各行は tfrecordのフルパス / 相対パス / セグメントstem を許可。"
        ),
    )
    ap.add_argument("--subset", type=str, choices=["training", "validation", "testing"], required=True)

    ap.add_argument(
        "--tfrecord-root",
        type=str,
        default=DEFAULT_TFRECORD_ROOT,
        help=(
            "txt内がstemの場合に探索するWaymo individual_filesのルート（既定: "
            f"{DEFAULT_TFRECORD_ROOT}）。例: <root>/<subset>/<stem>.tfrecord"
        ),
    )

    # ルート（あなたの運用に合わせて /mnt/e/waymo/data/ をデフォルトにする）
    ap.add_argument("--maps-root", type=str, default="/mnt/e/waymo/data/maps")
    ap.add_argument("--submaps-root", type=str, default="/mnt/e/waymo/data/submaps")

    # カメラは FRONT のみに固定（今の研究方針）
    ap.add_argument("--cam", type=str, choices=["FRONT"], default="FRONT")

    # 露出パラメータ（投影後のfloatマップ(out_map)を作る段階）
    ap.add_argument("--int-percentile", type=float, default=99.5,
                    help="up値の上限パーセンタイル（per_submap/global_map の基準）")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="ガンマ補正（0–1正規化後に適用）")
    ap.add_argument("--int-mode", type=str,
                    choices=["per_submap", "global_map", "camera_match", "camera_blend"],
                    default="global_map",
                    help="露出方式（基本はglobal_map推奨）")
    ap.add_argument("--match-pct-lo", type=float, default=5.0,
                    help="camera_match で合わせる下位パーセンタイル")
    ap.add_argument("--match-pct-hi", type=float, default=95.0,
                    help="camera_match で合わせる上位パーセンタイル")
    ap.add_argument("--match-blend", type=float, default=0.5,
                    help="camera_blend のときのブレンド率 α（0〜1）")
    ap.add_argument("--range-corr-power", type=float, default=0.0,
                    help="距離補正の指数 p（I'=I*depth^p, depthは投影Z）。0で無効")
    ap.add_argument("--match-covered-only", type=str2bool, default=True,
                    help="camera_match の係数推定を LiDARカバー画素（idx_map>=0）のみに限定")

    # 描画・クロップ
    ap.add_argument("--point-size", type=int, default=2, help="1:点のみ, 2:半径2pxスプラット（z比較あり）")
    ap.add_argument("--write-index", type=str2bool, default=True, help="anchor_index.npz を保存（pixel->3D index/depth）")
    ap.add_argument("--crop-mode", type=str, choices=["none", "circle"], default="circle",
                    help="circle: XY円筒クロップ（推奨：高速化）、none: 全点（遅い）")
    ap.add_argument("--crop-radius-m", type=str, default="auto",
                    help="circle時の半径[m]。autoならsubmaps.jsonのsubmap_radius_mを使う")
    ap.add_argument("--crop-margin-m", type=float, default=2.0,
                    help="circle時の半径に足す安全マージン[m]")
    ap.add_argument("--z-min", type=float, default=None)
    ap.add_argument("--z-max", type=float, default=None)

    # 追加：後処理（ヒストグラム平坦化 → CLAHE）
    ap.add_argument("--post-hist-eq", type=str2bool, default=True,
                    help="8bit化後にヒストグラム平坦化を適用")
    ap.add_argument("--post-clahe", type=str2bool, default=True,
                    help="8bit化後にCLAHEを適用（post-hist-eqの後に実行）")
    ap.add_argument("--clahe-clip-limit", type=float, default=2.0,
                    help="CLAHE clipLimit（大きいほど強くなる）")
    ap.add_argument("--clahe-tile-grid", type=int, default=8,
                    help="CLAHE tileGridSize（例：8なら(8,8)）")
    ap.add_argument("--post-mask-covered-only", type=str2bool, default=True,
                    help="後処理をLiDARカバー画素だけに適用し、背景は0に戻す（推奨）")

    # レジューム用：既存アンカーの扱い
    ap.add_argument("--overwrite", type=str2bool, default=False,
                    help="true なら既存 anchor.png/anchor.json を再計算して上書き、false なら揃っているサブマップはスキップ")

    args = ap.parse_args()

    tf_arg = Path(args.tfrecord)
    if not tf_arg.exists():
        print(f"[ERR] 入力が見つかりません: {tf_arg}")
        sys.exit(1)

    is_list_mode = tf_arg.suffix.lower() == ".txt"

    if not is_list_mode:
        # 従来通り: 単一tfrecord
        _process_one_tfrecord(tfrecord_path=tf_arg, args=args, strict=True)
        return

    # txtで複数tfrecord
    items = _read_nonempty_lines(tf_arg)
    if len(items) == 0:
        print(f"[ERR] txt が空です: {tf_arg}")
        sys.exit(1)

    tfrecord_root = Path(args.tfrecord_root)

    print(f"[INFO] list mode: {tf_arg}")
    print(f"[INFO] entries: {len(items)}")
    print(f"[INFO] tfrecord_root: {tfrecord_root}")
    print(f"[INFO] subset: {args.subset}")

    ok = 0
    err = 0

    for i, item in enumerate(items):
        print("\n" + "=" * 80)
        print(f"[INFO] ({i + 1}/{len(items)}) item: {item}")

        try:
            tf_path = resolve_tfrecord_path(
                item=item,
                subset=str(args.subset),
                tfrecord_root=tfrecord_root,
                txt_dir=tf_arg.parent,
            )
        except Exception as e:
            print(f"[ERR] tfrecord 解決に失敗: {item}")
            print(f"      {type(e).__name__}: {e}")
            err += 1
            continue

        try:
            st = _process_one_tfrecord(tfrecord_path=tf_path, args=args, strict=False)
        except Exception as e:
            print(f"[ERR] 処理に失敗: {tf_path}")
            print(f"      {type(e).__name__}: {e}")
            err += 1
            continue

        if st == "ok":
            ok += 1
        else:
            err += 1

    print("\n" + "=" * 80)
    print("[INFO] finished list mode")
    print(f"[INFO] ok={ok}, err={err}")

    if err > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
