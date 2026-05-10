#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 使い方:
#   python make_retrieval_shifted_ri_txt.py --help
#   必要な入力パスと出力先を引数で指定して実行する。

"""
step:5 Retrieval 学習用の「ずらし RI」データセットを作成するスクリプト（Step5-A）。

入力:
  - Waymo TFRecord (.tfrecord)
  - map_static.npz        (Step2: make_static_map.py の出力)
  - submaps.json          (Step1: make_submaps_distance.py の出力)
  - cam_gray/.../*_cam.json (Step4: export_cam_gray.py の出力, CLAHE済み画像へのパス)

出力:
  /mnt/e/waymo/data/retrieval_pairs/<subset>/<segment>/<submap_id>/
    cam.json
    ri_000.png / ri_000.json   (Δ=0: zero-ri-mode により reuse/render/skip)
    ri_001.png / ri_001.json
    ... (num_samples_per_anchor 枚)
    _done.json                 （今回追加：アンカー単位の完了フラグ）

ずらし設計（Retrieval 学習時）:
  Δs（前後）∈ [-0.45, +0.45] m
  Δl（左右）∈ [-1.0, +1.0] m
  Δψ（yaw）∈ [-2°, +2°] deg

  Δs:
    70%: |Δs| <= 0.25（一様）
    30%: 0.25 < |Δs| <= 0.45（一様）
  Δl:
    70%: |Δl| <= 0.6（一様）
    30%: 0.6 < |Δl| <= 1.0（一様）
  Δψ:
    一様 [-2°, +2°]

描画条件:
  - crop は行わない（全点群を使用）
  - intensity 正規化: map_static 全体から int_percentile% の上限 up を計算（外れ値に頑健）
  - range_corr_power = 0（距離補正なし）
  - z-buffer + point splat (point_size=2)
  - post-process: hist_eq + CLAHE（mask=LiDARカバー画素のみ）
  - gamma=1.0（=なし）

"""

import sys
import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2 as open_dataset


# txt（複数tfrecord）入力のときに、stem から tfrecord を解決するためのルート
# 例: <tfrecord_root>/<subset>/<stem>.tfrecord
DEFAULT_TFRECORD_ROOT = "/mnt/e/waymo_tf/perception_v1.4.3/individual_files"


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
        # BOM付きなど
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


# point_size=2 のスプラット用（半径2px近傍）
OFFSETS_R2 = [
    (-2, 0), (-1, -1), (-1, 0), (-1, 1),
    (0, -2), (0, -1),  (0, 1),  (0, 2),
    (1, -1), (1, 0),   (1, 1),  (2, 0),
]


def str2bool(v):
    """CLI などで受け取った真偽値表現を bool に正規化する。"""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {v}")


def ensure_dir(p: Path):
    """出力先ディレクトリを作成して存在を保証する。"""
    p.mkdir(parents=True, exist_ok=True)


def iter_tfrecord_bytes(tfrecord_path: Path):
    """
    TFRecord を順次 bytes で読む。

    備考:
      この環境 (TF 2.11) では `CUDA_VISIBLE_DEVICES=-1` 指定時に
      `tf.data.TFRecordDataset(...).numpy()` で glibc の double free が発生するため、
      安定動作する v1 iterator を使う。
    """
    # NOTE: tf.compat.v1.io.tf_record_iterator は deprecated だが、本用途では
    # 逐次読み出しのみで十分かつ CPU-only 実行で安定する。
    for rec in tf.compat.v1.io.tf_record_iterator(str(tfrecord_path)):
        yield rec


def mat4_list_to_np(m) -> np.ndarray:
    """行優先の 16 要素配列を 4x4 の NumPy 行列に変換する。"""
    return np.array(m, dtype=np.float64).reshape(4, 4)


def camera_name_from_str(name: str) -> int:
    """文字列から Waymo のカメラ種別 ID を解決する。"""
    nm = name.strip().upper()
    if not hasattr(open_dataset.CameraName, nm):
        raise ValueError(f"Unknown camera name: {name}")
    return getattr(open_dataset.CameraName, nm)


def intrinsic_from(calib) -> Tuple[float, float, float, float]:
    """キャリブレーションやメタ情報から内部パラメータを取り出す。"""
    intr = list(calib.intrinsic)
    if len(intr) < 4:
        raise ValueError("camera intrinsic has less than 4 elements")
    return float(intr[0]), float(intr[1]), float(intr[2]), float(intr[3])  # fx, fy, cx, cy


def yaw_from_T_wv(T_wv: np.ndarray) -> float:
    """車両姿勢行列（world<-vehicle）から水平ヨー角[deg]を取り出す。"""
    R = T_wv[:3, :3]
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    while yaw > 180.0:
        yaw -= 360.0
    while yaw < -180.0:
        yaw += 360.0
    return float(yaw)


def world_to_cam_from_Twv(T_wv: np.ndarray, calib) -> np.ndarray:
    """
    world<-vehicle (T_wv) と vehicle<-camera (extrinsic) から world->camera を作る。
      T_wv: world <- vehicle
      T_vc: vehicle <- camera

    world->camera = (camera<-vehicle) @ (vehicle<-world)
                  = inv(T_vc)        @ inv(T_wv)
    """
    T_vc = mat4_list_to_np(calib.extrinsic.transform)  # vehicle <- camera
    T_cv = np.linalg.inv(T_vc)                         # camera <- vehicle
    T_vw = np.linalg.inv(T_wv)                         # vehicle <- world
    T_wc = T_cv @ T_vw                                 # world -> camera
    return T_wc.astype(np.float32)


def compute_up_value(intensity_all: np.ndarray, percentile: float) -> float:
    """
    intensity の上限（正規化係数 up）をパーセンタイルで決める。
    例: 99.5 -> 上位 0.5% を飽和させる（外れ値に頑健）。
    """
    if intensity_all.size == 0:
        return 1.0
    pct = float(percentile)
    pct = max(0.0, min(100.0, pct))
    up = float(np.percentile(intensity_all, pct))
    return max(up, 1e-6)


def sample_delta_s(rng: np.random.RandomState) -> float:
    """Retrieval 用 Δs サンプリング（前後）"""
    if rng.rand() < 0.7:
        return float(rng.uniform(-0.25, 0.25))
    mag = float(rng.uniform(0.25, 0.45))
    return mag if (rng.rand() < 0.5) else -mag


def sample_delta_l(rng: np.random.RandomState) -> float:
    """Retrieval 用 Δl サンプリング（左右）"""
    if rng.rand() < 0.7:
        return float(rng.uniform(-0.6, 0.6))
    mag = float(rng.uniform(0.6, 1.0))
    return mag if (rng.rand() < 0.5) else -mag


def sample_delta_yaw_deg(rng: np.random.RandomState) -> float:
    """Δψ [deg] を一様にサンプリング（[-2, +2]）"""
    return float(rng.uniform(-2.0, +2.0))


def build_T_wv_with_offset(T_wv_base: np.ndarray, ds: float, dl: float, dyaw_deg: float) -> np.ndarray:
    """
    vehicle の基準姿勢 T_wv_base (world<-vehicle) に対して，
    ローカル座標（前後=Δs, 左右=Δl, yaw=Δψ）でオフセットを加えた T_wv_new を作る。

    - ds, dl は「基準yaw」に沿ってworld上で平行移動
    - yaw は世界Z軸回りの回転として加算
    - roll/pitchは無視（平坦路面近似）
    """
    p_base = T_wv_base[:3, 3].astype(np.float64)
    yaw_base_deg = yaw_from_T_wv(T_wv_base)
    yaw_base_rad = math.radians(yaw_base_deg)

    # ワールド XY 平面での前方 / 左方ベクトル
    forward = np.array([math.cos(yaw_base_rad), math.sin(yaw_base_rad), 0.0], dtype=np.float64)
    left = np.array([-math.sin(yaw_base_rad), math.cos(yaw_base_rad), 0.0], dtype=np.float64)

    offset = ds * forward + dl * left
    p_new = p_base + offset

    yaw_new_deg = yaw_base_deg + dyaw_deg
    yaw_new_rad = math.radians(yaw_new_deg)
    Rz = np.array([
        [math.cos(yaw_new_rad), -math.sin(yaw_new_rad), 0.0],
        [math.sin(yaw_new_rad),  math.cos(yaw_new_rad), 0.0],
        [0.0,                   0.0,                  1.0],
    ], dtype=np.float64)

    T_wv_new = np.eye(4, dtype=np.float64)
    T_wv_new[:3, :3] = Rz
    T_wv_new[:3, 3] = p_new
    return T_wv_new


def project_points_world_to_image(
    xyz_world: np.ndarray,
    Twc: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Waymo既存系と同じ投影:
      world->cam: [Xc,Yc,Zc]
      depth Z = Xc (>0)
      U=-Yc, V=-Zc
      u = fx*(U/Z)+cx
      v = fy*(V/Z)+cy

    戻り値:
      u(int32), v(int32), depth(Z=float32), front_mask(bool, len=N_world)
    """
    N = xyz_world.shape[0]
    if N == 0:
        return None

    # Twc: world から camera への変換
    R = Twc[:3, :3]
    t = Twc[:3, 3]

    x = xyz_world[:, 0]
    y = xyz_world[:, 1]
    z = xyz_world[:, 2]

    # pts_c = xyz @ R^T + t でカメラ座標へ変換する
    Xc = x * R[0, 0] + y * R[0, 1] + z * R[0, 2] + t[0]
    Yc = x * R[1, 0] + y * R[1, 1] + z * R[1, 2] + t[1]
    Zc = x * R[2, 0] + y * R[2, 1] + z * R[2, 2] + t[2]

    depth = Xc  # Z = Xc
    U = -Yc
    V = -Zc

    front = depth > 0.1
    if not np.any(front):
        return None

    depth_f = depth[front].astype(np.float32)
    U_f = U[front].astype(np.float32)
    V_f = V[front].astype(np.float32)

    invZ = 1.0 / (depth_f + 1e-6)
    u = (fx * (U_f * invZ) + cx).astype(np.int32)
    v = (fy * (V_f * invZ) + cy).astype(np.int32)
    return u, v, depth_f, front


def masked_hist_equalize_uint8(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    mask==True の画素だけを使ってヒストグラム平坦化。
    mask==False は 0 に戻す。
    """
    if img.dtype != np.uint8:
        raise ValueError("masked_hist_equalize_uint8 expects uint8")
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


def apply_clahe_masked_uint8(img: np.ndarray, mask: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    """
    CLAHE を適用し、mask==False の画素は 0 に戻す。
    """
    if img.dtype != np.uint8:
        raise ValueError("apply_clahe_masked_uint8 expects uint8")
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)

    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid), int(tile_grid)))
    out = clahe.apply(img)
    out[~mask] = 0
    return out


def rasterize_zbuffer(
    W: int,
    H: int,
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    val01: np.ndarray,
    pt_index: np.ndarray,
    point_size: int,
):
    """
    z-buffer + point splat で [0,1] の base_map を作る。
    戻り値:
      base_map(float32 [H,W]), depth_map(float32 [H,W]), idx_map(int32 [H,W])
    """
    base_map = np.zeros((H, W), dtype=np.float32)
    depth_map = np.full((H, W), np.inf, dtype=np.float32)
    idx_map = -np.ones((H, W), dtype=np.int32)

    # 遠方から近方へ処理する
    order = np.argsort(depth)[::-1]
    uo, vo = u[order], v[order]
    do = depth[order]
    bo = val01[order]
    io = pt_index[order]

    base_map[vo, uo] = bo
    depth_map[vo, uo] = do
    idx_map[vo, uo] = io

    if point_size > 1:
        for du, dv in OFFSETS_R2:
            uu = uo + du
            vv = vo + dv
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
            idx_map[rr_v, rr_u] = rr_i

    return base_map, depth_map, idx_map


def find_anchor_png(submaps_dir: Path, submap_id: int, anchor_png_name: str) -> Optional[Path]:
    """
    Step3(render_anchor_intensity.py) の出力（anchor.png）を submap_id から探す。
    ディレクトリ命名が環境差で揺れる可能性があるので候補を複数見る。
    """
    cand = []
    cand.append(submaps_dir / str(submap_id) / anchor_png_name)
    cand.append(submaps_dir / f"{submap_id:06d}" / anchor_png_name)
    cand.append(submaps_dir / f"{submap_id:05d}" / anchor_png_name)
    cand.append(submaps_dir / f"{submap_id:04d}" / anchor_png_name)
    cand.append(submaps_dir / f"submap_{submap_id}" / anchor_png_name)
    cand.append(submaps_dir / f"submap_{submap_id:06d}" / anchor_png_name)
    cand.append(submaps_dir / f"submap_{submap_id:04d}" / anchor_png_name)

    for p in cand:
        if p.exists():
            return p
    return None




def _process_one_tfrecord(tfrecord_path: Path, args: argparse.Namespace, strict: bool = True) -> str:
    """単一tfrecordを元の処理そのままで処理する（list mode用に関数化）。"""
    try:
        if args.num_shards <= 0:
            raise ValueError("--num-shards must be >= 1")
        if not (0 <= args.shard_id < args.num_shards):
            raise ValueError("--shard-id must be in [0, num_shards-1]")
    
        rng = np.random.RandomState(args.seed)
    
        tfrecord_path = Path(tfrecord_path)
        if not tfrecord_path.exists():
            raise FileNotFoundError(str(tfrecord_path))
        seg_stem = tfrecord_path.stem
    
        maps_dir = Path(args.maps_root) / args.subset / seg_stem
        submaps_dir = Path(args.submaps_root) / args.subset / seg_stem
        cam_gray_dir = Path(args.cam_gray_root) / args.subset / seg_stem / args.cam.upper()
        out_seg_dir = Path(args.out_root) / args.subset / seg_stem
    
        map_npz = maps_dir / "map_static.npz"
        submaps_json = submaps_dir / "submaps.json"
    
        if not map_npz.exists():
            raise FileNotFoundError(str(map_npz))
        if not submaps_json.exists():
            raise FileNotFoundError(str(submaps_json))
        if not cam_gray_dir.exists():
            raise FileNotFoundError(str(cam_gray_dir))
    
        ensure_dir(out_seg_dir)
    
        # TFRecord 全フレーム読み
        frames = []
        for rec in iter_tfrecord_bytes(tfrecord_path):
            fr = open_dataset.Frame()
            fr.ParseFromString(rec)
            frames.append(fr)
        n_frames = len(frames)
        print(f"[INFO] segment={seg_stem} frames={n_frames}")
    
        # export_cam_gray.py のゼロパディング幅に合わせる
        padw = max(4, len(str(n_frames - 1)))
    
        cam_name_id = camera_name_from_str(args.cam)
    
        # map_static 読み込み
        with np.load(map_npz) as zf:
            xyz_world_all = zf["xyz"].astype(np.float32)
            intensity_all = zf["intensity"].astype(np.float32)
        print(f"[INFO] map_static: pts={xyz_world_all.shape[0]}")
    
        # submaps.json 読み込み
        with open(submaps_json, "r", encoding="utf-8") as f:
            submap_obj = json.load(f)
        submaps = submap_obj["submaps"]
        print(f"[INFO] submaps: {len(submaps)} anchors")
    
        # グローバルな up 値
        up_val = compute_up_value(intensity_all, args.int_percentile)
        print(f"[INFO] int_percentile={args.int_percentile} -> global_up={up_val:.6f}")
    
        # shard適用（submapsの順番で割る）
        submaps_sharded = []
        for i, sm in enumerate(submaps):
            if (i % args.num_shards) == args.shard_id:
                submaps_sharded.append(sm)
        print(f"[INFO] shard: {args.shard_id}/{args.num_shards} -> anchors={len(submaps_sharded)}")
    
        if args.max_anchors > 0:
            submaps_sharded = submaps_sharded[:args.max_anchors]
            print(f"[INFO] max_anchors applied -> {len(submaps_sharded)}")
    
        # メインループ
        for sm in tqdm(submaps_sharded, desc="anchors", unit="anchor"):
            submap_id = int(sm["submap_id"])
            anchor_idx = int(sm["anchor_frame_index"])
    
            out_anchor_dir = out_seg_dir / str(submap_id)
            done_flag = out_anchor_dir / "_done.json"
    
            # 既存出力の状態に応じてレジューム判定
            if out_anchor_dir.exists():
                if done_flag.exists():
                    if not args.overwrite:
                        print(f"[SKIP] submap {submap_id}: _done.json があり、既にRetrieval RI生成済みのためスキップします。")
                        continue
                    else:
                        print(f"[INFO] submap {submap_id}: overwrite=true のため既存出力を削除して再生成します。")
                        shutil.rmtree(out_anchor_dir)
                else:
                    # ディレクトリはあるが done フラグが無い = 旧バージョン or 中途半端
                    print(f"[INFO] submap {submap_id}: 出力ディレクトリは存在するが _done.json が無いため再生成します。")
                    shutil.rmtree(out_anchor_dir)
    
            ensure_dir(out_anchor_dir)
    
            # 入力フレーム・キャリブ
            frame = frames[anchor_idx]
            calib = next((c for c in frame.context.camera_calibrations if c.name == cam_name_id), None)
            if calib is None:
                print(f"[WARN] submap {submap_id}: camera calib not found, skip")
                continue
    
            W, H = int(calib.width), int(calib.height)
            fx, fy, cx, cy = intrinsic_from(calib)
    
            # 基準vehicle pose
            T_wv_base = mat4_list_to_np(frame.pose.transform)
    
            # クエリカメラ（CLAHE済み）メタ
            cam_json_path = cam_gray_dir / f"{anchor_idx:0{padw}d}_cam.json"
            if not cam_json_path.exists():
                print(f"[WARN] submap {submap_id}: cam json not found ({cam_json_path}), skip")
                continue
            with open(cam_json_path, "r", encoding="utf-8") as f:
                cam_meta = json.load(f)
            cam_img_path = cam_meta["image_path"]
    
            # cam.json 保存（学習ローダが参照しやすい形）
            cam_out = {
                "segment_id": seg_stem,
                "subset": args.subset,
                "submap_id": submap_id,
                "anchor_frame_index": anchor_idx,
                "camera_name": args.cam.upper(),
                "cam_image_path": cam_img_path,
                "intrinsic": [fx, fy, cx, cy],
                "vehicle_pose_world_from_vehicle": T_wv_base.tolist(),
                "cam_meta": cam_meta,
            }
            with open(out_anchor_dir / "cam.json", "w", encoding="utf-8") as f:
                json.dump(cam_out, f, indent=2, ensure_ascii=False)
    
            # Δリスト生成（ri_000はΔ=0を想定。ただし zero-ri-mode=skip なら後で除外）
            num_target = max(1, int(args.num_samples_per_anchor))
            deltas = [(0.0, 0.0, 0.0)]
            while len(deltas) < num_target:
                ds = sample_delta_s(rng)
                dl = sample_delta_l(rng)
                dy = sample_delta_yaw_deg(rng)
                deltas.append((ds, dl, dy))
    
            # ri_000 の扱い
            start_k = 0
            if args.zero_ri_mode == "skip":
                start_k = 1  # 0番は作らない
    
            # zero reuse の場合はコピーする（ri_000.png）
            if args.zero_ri_mode == "reuse":
                anchor_png = find_anchor_png(submaps_dir, submap_id, args.anchor_png_name)
                if anchor_png is None:
                    print(f"[WARN] submap {submap_id}: anchor.png not found -> fallback to render for ri_000")
                    # フォールバック: 再描画する
                else:
                    dst_png = out_anchor_dir / "ri_000.png"
                    shutil.copy2(anchor_png, dst_png)
                    meta0 = {
                        "segment_id": seg_stem,
                        "subset": args.subset,
                        "submap_id": submap_id,
                        "anchor_frame_index": anchor_idx,
                        "sample_index": 0,
                        "camera_name": args.cam.upper(),
                        "cam_image_path": cam_img_path,
                        "ri_image_path": str(dst_png),
                        "delta": {"ds_m": 0.0, "dl_m": 0.0, "dyaw_deg": 0.0},
                        "note": "ri_000 reused from Step3 anchor.png",
                        "render_profile": {
                            "int_percentile": float(args.int_percentile),
                            "global_up_val": float(up_val),
                            "range_corr_power": float(args.range_corr_power),
                            "point_size": int(args.point_size),
                            "postprocess": {
                                "hist_eq": True,
                                "clahe": True,
                                "gamma": 1.0,
                                "mask_covered_only": bool(args.post_mask_covered_only),
                                "clahe_clip_limit": float(args.clahe_clip_limit),
                                "clahe_tile_grid": int(args.clahe_tile_grid),
                            },
                        },
                    }
                    with open(out_anchor_dir / "ri_000.json", "w", encoding="utf-8") as f:
                        json.dump(meta0, f, indent=2, ensure_ascii=False)
    
            # 以降のサンプルを生成（ri_000もrenderが必要な場合はここで生成）
            for k in range(start_k, len(deltas)):
                ds, dl, dy_deg = deltas[k]
    
                # reuseが成功した場合の ri_000 はスキップ
                if (k == 0) and (args.zero_ri_mode == "reuse"):
                    anchor_png = find_anchor_png(submaps_dir, submap_id, args.anchor_png_name)
                    if anchor_png is not None:
                        continue
    
                # render する必要がある（zero-ri-mode=render または reuse失敗時）
                T_wv_new = build_T_wv_with_offset(T_wv_base, ds, dl, dy_deg)
                Twc_new = world_to_cam_from_Twv(T_wv_new, calib)
    
                proj = project_points_world_to_image(xyz_world_all, Twc_new, fx, fy, cx, cy)
                if proj is None:
                    continue
                u, v, depth, front_mask = proj
    
                # front_mask は world点群に対するbool（len=N）
                intens_front = intensity_all[front_mask]
                idx_front = np.flatnonzero(front_mask).astype(np.int32)
    
                # 画面内フィルタ
                ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                if not np.any(ok):
                    continue
    
                u = u[ok]
                v = v[ok]
                depth_k = depth[ok]
                intens_k = intens_front[ok]
                idx_k = idx_front[ok]
    
                # 距離補正
                if abs(args.range_corr_power) > 1e-9:
                    intens_k = intens_k * np.power(depth_k, args.range_corr_power)
    
                # 正規化
                val01 = np.clip(intens_k / (up_val + 1e-6), 0.0, 1.0).astype(np.float32)
    
                base_map, depth_map, idx_map = rasterize_zbuffer(
                    W, H, u, v, depth_k, val01, idx_k, args.point_size
                )
    
                covered = (idx_map >= 0)
                coverage = float(np.mean(covered))
                if coverage < args.min_coverage:
                    continue
    
                # gamma=1.0（=なし）
                img_u8 = np.clip((base_map * 255.0).round(), 0, 255).astype(np.uint8)
    
                # 後処理（histeq -> clahe）
                if args.post_mask_covered_only:
                    mask_pp = covered
                else:
                    mask_pp = np.ones_like(img_u8, dtype=bool)
    
                img_u8 = masked_hist_equalize_uint8(img_u8, mask_pp)
                img_u8 = apply_clahe_masked_uint8(img_u8, mask_pp, args.clahe_clip_limit, args.clahe_tile_grid)
    
                fname = f"ri_{k:03d}"
                png_path = out_anchor_dir / f"{fname}.png"
                json_path = out_anchor_dir / f"{fname}.json"
    
                cv2.imwrite(str(png_path), img_u8)
    
                if args.write_index:
                    np.savez_compressed(
                        out_anchor_dir / f"{fname}_index_depth.npz",
                        point_index=idx_map,
                        depth=depth_map,
                    )
    
                meta = {
                    "segment_id": seg_stem,
                    "subset": args.subset,
                    "submap_id": submap_id,
                    "anchor_frame_index": anchor_idx,
                    "sample_index": int(k),
                    "camera_name": args.cam.upper(),
                    "cam_image_path": cam_img_path,
                    "ri_image_path": str(png_path),
                    "delta": {"ds_m": float(ds), "dl_m": float(dl), "dyaw_deg": float(dy_deg)},
                    "world_to_cam": Twc_new.astype(float).tolist(),
                    "coverage": coverage,
                    "render_profile": {
                        "int_percentile": float(args.int_percentile),
                        "global_up_val": float(up_val),
                        "range_corr_power": float(args.range_corr_power),
                        "point_size": int(args.point_size),
                        "postprocess": {
                            "hist_eq": True,
                            "clahe": True,
                            "gamma": 1.0,
                            "mask_covered_only": bool(args.post_mask_covered_only),
                            "clahe_clip_limit": float(args.clahe_clip_limit),
                            "clahe_tile_grid": int(args.clahe_tile_grid),
                        },
                    },
                }
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)
    
            # アンカー単位の完了フラグを書き出す
            ri_jsons = sorted(out_anchor_dir.glob("ri_*.json"))
            done_info = {
                "segment_id": seg_stem,
                "subset": args.subset,
                "submap_id": submap_id,
                "anchor_frame_index": anchor_idx,
                "num_ri": len(ri_jsons),
                "num_samples_per_anchor_arg": int(args.num_samples_per_anchor),
                "zero_ri_mode": args.zero_ri_mode,
            }
            with open(done_flag, "w", encoding="utf-8") as f:
                json.dump(done_info, f, indent=2, ensure_ascii=False)
    
        print(f"[OK] done (resume-aware): {out_seg_dir}")

        return "ok"
    except Exception as e:
        if strict:
            raise
        print(f"[ERR] failed: {tfrecord_path}")
        print(f"      {type(e).__name__}: {e}")
        return "err"


def main():
    """CLI 引数を解釈し、入力の読み込みから結果保存までの一連の処理を実行する。"""
    ap = argparse.ArgumentParser(description="Retrieval 学習用ずらしRI生成 (crop-mode none固定)")

    ap.add_argument(
        "--tfrecord",
        type=str,
        required=True,
        help=(
            "Waymo segment .tfrecord のフルパス、または tfrecord リスト .txt（1行1セグメント）\n"
            "txt の各行は以下いずれでもOK: \n"
            "  - tfrecord のフルパス\n"
            "  - txtからの相対パス\n"
            "  - segment stem（例: segment-..._with_camera_labels）\n"
        ),
    )
    ap.add_argument("--subset", type=str, choices=["training", "validation", "testing"], required=True)

    ap.add_argument(
        "--tfrecord-root",
        type=str,
        default=DEFAULT_TFRECORD_ROOT,
        help=(
            "txt 内が stem の場合に探索する Waymo individual_files のルート（既定: "
            f"{DEFAULT_TFRECORD_ROOT}）。例: <root>/<subset>/<stem>.tfrecord"
        ),
    )

    ap.add_argument("--maps-root", type=str, default="/mnt/e/waymo/data/maps", help="map_static.npz ルート")
    ap.add_argument("--submaps-root", type=str, default="/mnt/e/waymo/data/submaps", help="submaps.json ルート（Step3のanchor.pngもここ）")
    ap.add_argument("--cam-gray-root", type=str, default="/mnt/e/waymo/data/cam_gray", help="export_cam_gray.py の出力ルート")
    ap.add_argument("--out-root", type=str, default="/mnt/e/waymo/data/retrieval_pairs", help="出力ルート")
    ap.add_argument("--cam", type=str, default="FRONT", choices=["FRONT"], help="カメラ名（現状 FRONTのみ）")

    # 反射強度の正規化
    ap.add_argument("--int-percentile", type=float, default=99.5, help="global up のパーセンタイル")
    ap.add_argument("--range-corr-power", type=float, default=0.0, help="I'=I*depth^p の p（0で無効）")

    # ラスタライズ
    ap.add_argument("--point-size", type=int, default=2, help="1:点のみ, 2:半径2pxスプラット")
    ap.add_argument("--write-index", type=str2bool, default=False, help="trueなら idx_map/depth_map も保存（Retrievalだけなら通常不要）")

    # post-process (固定: histeq + clahe, gamma=1.0)
    ap.add_argument("--post-mask-covered-only", type=str2bool, default=True, help="LiDARカバー画素だけ後処理して背景を0に戻す")
    ap.add_argument("--clahe-clip-limit", type=float, default=2.0, help="CLAHE clipLimit")
    ap.add_argument("--clahe-tile-grid", type=int, default=8, help="CLAHE tileGridSize の1辺")

    # データセット規模 / サンプリング
    ap.add_argument("--num-samples-per-anchor", type=int, default=16, help="各アンカーあたり生成するRI枚数（ri_000含む）")
    ap.add_argument("--min-coverage", type=float, default=0.01, help="coverageがこれ未満のRIは捨てる")
    ap.add_argument("--seed", type=int, default=42, help="乱数seed")

    # (0,0,0) の扱い
    ap.add_argument(
        "--zero-ri-mode",
        type=str,
        choices=["reuse", "render", "skip"],
        default="reuse",
        help="ri_000(Δ=0)の作り方: reuse=Step3のanchor.pngをコピー, render=再レンダ, skip=作らない",
    )
    ap.add_argument("--anchor-png-name", type=str, default="anchor.png", help="Step3のアンカー強度画像ファイル名")

    # io
    ap.add_argument("--overwrite", type=str2bool, default=False, help="trueなら出力フォルダを作り直す（_done.json 無視で再計算）")
    ap.add_argument("--max-anchors", type=int, default=-1, help="デバッグ用: 処理するアンカー数上限（-1で全て）")

    # CPU シャーディング
    ap.add_argument("--num-shards", type=int, default=1, help="アンカーを分割する総シャード数")
    ap.add_argument("--shard-id", type=int, default=0, help="自分が処理するシャードID（0..num_shards-1）")

    args = ap.parse_args()

    if args.num_shards <= 0:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("--shard-id must be in [0, num_shards-1]")

    in_path = Path(args.tfrecord)
    if not in_path.exists():
        print(f"[ERR] input not found: {in_path}")
        sys.exit(1)

    is_list_mode = in_path.suffix.lower() == ".txt"

    # 従来通り: 単一tfrecord
    if not is_list_mode:
        _process_one_tfrecord(tfrecord_path=in_path, args=args, strict=True)
        return

    # 追加: txtで複数tfrecord
    items = _read_nonempty_lines(in_path)
    if len(items) == 0:
        print(f"[ERR] txt が空です: {in_path}")
        sys.exit(1)

    tfrecord_root = Path(args.tfrecord_root)

    print(f"[INFO] list mode: {in_path}")
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
                txt_dir=in_path.parent,
            )
        except Exception as e:
            print(f"[ERR] tfrecord 解決に失敗: {item}")
            print(f"      {type(e).__name__}: {e}")
            err += 1
            continue

        st = _process_one_tfrecord(tfrecord_path=tf_path, args=args, strict=False)
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
