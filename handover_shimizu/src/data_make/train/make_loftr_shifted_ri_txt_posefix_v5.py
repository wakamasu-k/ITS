#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 使い方:
#   python make_loftr_shifted_ri_txt_posefix_v5.py --help
#   必要な入力パスと出力先を引数で指定して実行する。

"""
make_loftr_shifted_ri_txt_posefix_v5.py

Waymo Open Dataset の submap と LiDAR 静的地図 (map_static.npz) から，
LoFTR/SFPPR 学習用の「ずらし反射強度画像 (RI)」を生成するスクリプト。

【入力】
  - Waymo TFRecord (.tfrecord)
  - map_static.npz             (Step2: make_static_map.py の出力)
  - submaps.json               (Step1: make_submaps_distance.py の出力)
  - cam_gray/.../*_cam.json    (Step4: export_cam_gray.py の出力)

【出力】
  <out-root>/<subset>/<segment>/<submap_id>/
    cam.json
    ri_000.png / ri_000.json
    ri_001.png / ri_001.json
    ...
    ri_xxx_index_depth.npz  (--write-index true の場合)

【モード】
  1) 単一セグメント:
     --tfrecord <path/to/segment-xxx.tfrecord>

  2) 複数セグメント一括:
     --segments-file <train_clear.txt> --tfrecord-root <dir>

train_clear.txt は 1行1セグメント名（拡張子なし）を想定：
  segment-xxxxxxxx_with_camera_labels

【再開性（途中停止→再実行）】
  - submapディレクトリが存在してもスキップしない
  - ri_xxx.png/json/(index_depth.npz) が揃っているサンプルだけスキップし，不足分だけ生成
  - Δサンプリングは submap 単位の安定seedにしているので，再開しても ri_005 が別Δになる事故を防ぐ

【重要設定】
  - 投影の丸め: np.round を使わず astype(np.int32)（render_anchor_intensity.py 側に合わせる）
  - --write-index はデフォルト true
  - ri_XXX.json に width / height / intrinsic=[fx,fy,cx,cy] を保存
  - depth欠損の最近傍埋め（hole filling）は入れていない（要望どおり）


"""

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2 as open_dataset


# 投影規約（後段GT作成で『どの式で投影したか』を明示するためにメタへ保存する）
PROJECTION_CONVENTION = {
    'name': 'waymo_anchor_like_v1',
    'camera_frame': 'Waymo camera frame: X forward, Y left, Z up',
    'image_frame': 'Image frame: u right, v down',
    'depth_definition': 'depth = Xc (forward axis)',
    'uv_definition': 'U=-Yc, V=-Zc; u=fx*(U/depth)+cx; v=fy*(V/depth)+cy',
    'rounding': 'astype(int32) truncation (no round)',
}


# 半径2pxのオフセット（中心以外の12点）
OFFSETS_R2 = [
    (-2, 0), (-1, -1), (-1, 0), (-1, 1),
    (0, -2), (0, -1), (0, 1), (0, 2),
    (1, -1), (1, 0), (1, 1), (2, 0),
]


def str2bool(s: str) -> bool:
    """CLI などで受け取った真偽値表現を bool に正規化する。"""
    return str(s).strip().lower() in ["1", "true", "t", "yes", "y"]


def ensure_dir(p: Path) -> None:
    """出力先ディレクトリを作成して存在を保証する。"""
    p.mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    """JSON を一時ファイル経由で安全に書き出す。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp.json", dir=str(path.parent))
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def atomic_savez_compressed(path: Path, **arrays: Any) -> None:
    """NumPy 配列群を一時ファイル経由で安全に圧縮保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp.npz", dir=str(path.parent))
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def get_file_fingerprint(path: Path, mode: str = 'head', head_bytes: int = 1048576) -> dict:
    """
    後段で『参照している map_static が本当に同じか』を確認しやすくするための軽量フィンガープリント。

    mode:
      - 'none': パス/サイズ/mtime のみ
      - 'head': 先頭 head_bytes だけ読んで md5
      - 'full': 全体を読んで md5（重いので注意）
    """
    path = Path(path)
    st = path.stat()
    info = {
        'path': str(path),
        'size_bytes': int(st.st_size),
        'mtime_unix': float(st.st_mtime),
        'hash_mode': str(mode),
    }

    mode = str(mode).lower()
    if mode == 'none':
        return info

    import hashlib
    md5 = hashlib.md5()

    if mode == 'head':
        n = max(0, int(head_bytes))
        with open(path, 'rb') as f:
            data = f.read(n)
        md5.update(data)
        info['md5'] = md5.hexdigest()
        info['hash_bytes'] = int(len(data))
        return info

    if mode == 'full':
        with open(path, 'rb') as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                md5.update(chunk)
        info['md5'] = md5.hexdigest()
        return info

    raise ValueError(f'Unknown hash mode: {mode}')


def stable_uint32_from_text(text: str) -> int:
    """
    Python の hash() はプロセスごとに変わり得るので使わない。
    md5 から安定した 32-bit seed を作る。
    """
    h = hashlib.md5(text.encode("utf-8")).digest()
    return int.from_bytes(h[:4], byteorder="little", signed=False)


def mat4_list_to_np(m) -> np.ndarray:
    """行優先の 16 要素配列を 4x4 の NumPy 行列に変換する。"""
    return np.array(m, dtype=np.float64).reshape(4, 4)


def _as_4x4(x: Any, name: str = "") -> np.ndarray:
    """入力を 4x4 の同次変換行列に整形する。"""
    a = np.asarray(x, dtype=np.float64)
    if a.shape == (4, 4):
        return a
    if a.size == 16:
        return a.reshape(4, 4)
    raise ValueError(f"{name} must be 4x4, got shape={a.shape}")


def _read_intrinsic_from_meta(meta: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    # 推奨する簡潔な形式
    """メタ JSON からカメラ内部パラメータを読み取る。"""
    if "intrinsic" in meta:
        arr = np.asarray(meta["intrinsic"], dtype=np.float64).reshape(-1)
        if arr.size == 4:
            return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
    # よくある代替形式
    if "intrinsics" in meta:
        arr = np.asarray(meta["intrinsics"], dtype=np.float64)
        if arr.shape == (4,):
            return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
        if arr.shape == (3, 3) or arr.size == 9:
            arr = arr.reshape(3, 3)
            return float(arr[0, 0]), float(arr[1, 1]), float(arr[0, 2]), float(arr[1, 2])
    if "K" in meta:
        arr = np.asarray(meta["K"], dtype=np.float64)
        if arr.shape == (3, 3) or arr.size == 9:
            arr = arr.reshape(3, 3)
            return float(arr[0, 0]), float(arr[1, 1]), float(arr[0, 2]), float(arr[1, 2])
    return None


def read_pose_intrinsic_from_cam_json(
    cam_meta: Dict[str, Any],
    Twc_fallback: np.ndarray,
    intr_fallback: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, Tuple[float, float, float, float], bool, bool]:
    """カメラ JSON から姿勢と内部パラメータをまとめて読み込む。"""
    Twc = np.asarray(Twc_fallback, dtype=np.float64)
    intr = tuple(float(x) for x in intr_fallback)
    pose_from_cam = False
    intr_from_cam = False

    # 姿勢
    try:
        if "world_to_cam" in cam_meta:
            Twc = _as_4x4(cam_meta["world_to_cam"], "world_to_cam")
            pose_from_cam = True
        elif "cam_to_world" in cam_meta:
            T_cw = _as_4x4(cam_meta["cam_to_world"], "cam_to_world")
            Twc = np.linalg.inv(T_cw)
            pose_from_cam = True
        elif "world_from_cam" in cam_meta:
            T_w_from_c = _as_4x4(cam_meta["world_from_cam"], "world_from_cam")
            Twc = np.linalg.inv(T_w_from_c)
            pose_from_cam = True
    except Exception:
        pose_from_cam = False

    # 内部パラメータ
    intr_cam = _read_intrinsic_from_meta(cam_meta)
    if intr_cam is not None:
        intr = intr_cam
        intr_from_cam = True

    return Twc.astype(np.float64), intr, pose_from_cam, intr_from_cam


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


def sample_delta_loftr_sl(rng: np.random.RandomState) -> Tuple[float, float, str]:
    """
    LoFTR Train（design.txt準拠）
      Inner 70% : |ds|<=0.5 and |dl|<=0.5
      Outer 25% : [-1,1]^2 \ (Inner ∪ Corner)
      Corner 5% : |ds|>0.8 and |dl|>0.8
    """
    r = float(rng.rand())

    P_INNER = 0.70
    P_OUTER = 0.25
    INNER_MAX = 0.5
    CORNER_MIN = 0.8
    MAX_ABS = 1.0

    if r < P_INNER:
        ds = float(rng.uniform(-INNER_MAX, INNER_MAX))
        dl = float(rng.uniform(-INNER_MAX, INNER_MAX))
        return ds, dl, "inner"

    if r < P_INNER + P_OUTER:
        # 外側領域: 内側領域と角領域を除いた [-1,1]^2 から棄却サンプリングする
        for _ in range(2000):
            ds = float(rng.uniform(-MAX_ABS, MAX_ABS))
            dl = float(rng.uniform(-MAX_ABS, MAX_ABS))
            if (abs(ds) <= INNER_MAX) and (abs(dl) <= INNER_MAX):
                continue
            if (abs(ds) > CORNER_MIN) and (abs(dl) > CORNER_MIN):
                continue
            return ds, dl, "outer"
        ds = float(rng.uniform(-MAX_ABS, MAX_ABS))
        dl = float(rng.uniform(-MAX_ABS, MAX_ABS))
        return ds, dl, "outer_fallback"

    # 角領域
    ds = float(rng.uniform(CORNER_MIN, MAX_ABS))
    dl = float(rng.uniform(CORNER_MIN, MAX_ABS))
    if rng.rand() < 0.5:
        ds = -ds
    if rng.rand() < 0.5:
        dl = -dl
    return ds, dl, "corner"


def sample_delta_yaw_deg(rng: np.random.RandomState) -> float:
    """LoFTR 学習用の Δψ を一様分布 [-2°, +2°] からサンプリングする。"""
    return float(rng.uniform(-2.0, 2.0))


def build_T_wv_with_offset(
    T_wv_base: np.ndarray,
    ds_m: float,
    dl_m: float,
    dyaw_deg: float,
    preserve_roll_pitch: bool = True,
) -> np.ndarray:
    """
    base姿勢を中心に、車体座標の (s=前後, l=左右) と yaw をずらした world<-vehicle を作る。

    注意:
      - 平面ヨーのみで回す（roll/pitchは無視）
    """
    R_base = T_wv_base[:3, :3].astype(np.float64)
    t_base = T_wv_base[:3, 3].astype(np.float64)
    ds = float(ds_m)
    dl = float(dl_m)

    # 基準車両の局所座標系で並進する: s は前後（x軸）, l は左方向（y軸）。
    t_new = t_base + R_base[:, 0] * ds + R_base[:, 1] * dl

    if preserve_roll_pitch:
        yaw = math.radians(float(dyaw_deg))
        c = math.cos(yaw)
        s = math.sin(yaw)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        R_new = R_base @ Rz
    else:
        # 旧挙動: yaw のみを持つ姿勢へ平坦化する
        yaw_base_deg = yaw_from_T_wv(T_wv_base)
        yaw_new_rad = math.radians(yaw_base_deg + float(dyaw_deg))
        c = math.cos(yaw_new_rad)
        s = math.sin(yaw_new_rad)
        R_new = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    T_new = np.eye(4, dtype=np.float64)
    T_new[:3, :3] = R_new
    T_new[:3, 3] = t_new
    return T_new


def project_points_world_to_image(
    xyz_world: np.ndarray, T_wc: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    world->camera (T_wc) で点群をカメラ座標へ変換し、Waymoの軸定義に合わせて投影する。

    Waymoカメラ座標（想定）:
      X forward, Y left, Z up
    画像座標:
      u right, v down

    変換:
      depth Zimg = Xc
      U = -Yc
      V = -Zc

    丸め:
      render_anchor_intensity.py に合わせて np.round は使わず astype(int32)
    """
    R = T_wc[:3, :3].astype(np.float32)
    t = T_wc[:3, 3].astype(np.float32)

    Xc = (xyz_world.astype(np.float32) @ R.T) + t[None, :]

    Z = Xc[:, 0]         # forward
    U = -Xc[:, 1]        # right
    V = -Xc[:, 2]        # down

    front = (Z > 0) & np.isfinite(Z)
    Z_safe = np.where(front, Z, 1.0)

    u = fx * (U / Z_safe) + cx
    v = fy * (V / Z_safe) + cy

    # roundしない（astype int32）
    u = u.astype(np.int32)
    v = v.astype(np.int32)

    depth = Z.astype(np.float32)
    return u, v, depth, front


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
    if mask is None:
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid), int(tile_grid)))
        return clahe.apply(img_u8)

    mask = mask.astype(bool)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid), int(tile_grid)))
    out = clahe.apply(img_u8)
    out[~mask] = 0
    return out


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

    order = np.argsort(depth)[::-1]  # far -> near（near が後に来て上書きされる狙い）
    uo = u[order]
    vo = v[order]
    do = depth[order].astype(np.float32)
    bo = base_norm_pts[order].astype(np.float32)
    io = pt_idx[order].astype(np.int32)

    base_map[vo, uo] = bo
    depth_map[vo, uo] = do
    idx_map[vo, uo] = io

    if int(point_size) > 1:
        for du, dv in OFFSETS_R2:
            uu = uo + int(du)
            vv = vo + int(dv)
            sel = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            if not np.any(sel):
                continue

            uu2 = uu[sel]
            vv2 = vv[sel]
            do2 = do[sel]
            bo2 = bo[sel]
            io2 = io[sel]

            nearer = do2 < depth_map[vv2, uu2]
            if not np.any(nearer):
                continue

            uu3 = uu2[nearer]
            vv3 = vv2[nearer]
            base_map[vv3, uu3] = bo2[nearer]
            depth_map[vv3, uu3] = do2[nearer]
            idx_map[vv3, uu3] = io2[nearer]

    return base_map, depth_map, idx_map


def rasterize_index_depth_nosplat(
    W: int,
    H: int,
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    pt_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    point_size=1 での point_index/depth 生成（非スプラット）。
    """
    depth_map = np.full((H, W), np.inf, dtype=np.float32)
    idx_map = -np.ones((H, W), dtype=np.int32)
    if u.size == 0:
        return depth_map, idx_map

    # 遠方から近方へ上書きし、z-buffer で最近傍のみを残す挙動と等価にする
    order = np.argsort(depth)[::-1]
    uo = u[order]
    vo = v[order]
    do = depth[order].astype(np.float32)
    io = pt_idx[order].astype(np.int32)
    depth_map[vo, uo] = do
    idx_map[vo, uo] = io
    return depth_map, idx_map


def find_anchor_png(submaps_dir: Path, submap_id: int, anchor_png_name: str) -> Optional[Path]:
    # サブマップディレクトリの命名が (str) or (06d) どちらでも拾えるようにする
    """候補ディレクトリから対応するアンカー画像を探す。"""
    cand1 = submaps_dir / str(submap_id) / anchor_png_name
    if cand1.exists():
        return cand1
    cand2 = submaps_dir / f"{submap_id:06d}" / anchor_png_name
    if cand2.exists():
        return cand2
    return None


def load_segments_list(txt_path: Path) -> List[str]:
    """セグメント一覧テキストを読み込む。"""
    segs: List[str] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            segs.append(s)
    return segs


def resolve_tfrecord_path(item: str, tfrecord_root: Optional[Path]) -> Optional[Path]:
    """
    segments-file の 1行(item) から tfrecord の実ファイルを解決する。

    - item が /path/.../*.tfrecord の場合: それを使う（存在チェック）
    - item が segment 名の場合: tfrecord_root / (item + ".tfrecord") を探す
    """
    p = Path(item)

    # すでに .tfrecord のパスっぽい
    if p.suffix == ".tfrecord":
        if p.exists():
            return p
        if tfrecord_root is not None:
            cand = tfrecord_root / p.name
            if cand.exists():
                return cand
        return None

    # segment 名として扱う
    if tfrecord_root is None:
        return None
    cand2 = tfrecord_root / f"{item}.tfrecord"
    if cand2.exists():
        return cand2
    return None


def process_one_segment(args: argparse.Namespace, tfrecord_path: Path) -> bool:
    """
    1セグメント分を処理する。
    成功 True / 失敗 False（例：必要ファイルが無い等）
    """
    tfrecord_path = Path(tfrecord_path)
    seg_stem = tfrecord_path.stem

    maps_dir = Path(args.maps_root) / args.subset / seg_stem
    submaps_dir = Path(args.submaps_root) / args.subset / seg_stem
    cam_gray_dir = Path(args.cam_gray_root) / args.subset / seg_stem / args.cam.upper()
    out_seg_dir = Path(args.out_root) / args.subset / seg_stem

    map_npz = maps_dir / "map_static.npz"
    submaps_json = submaps_dir / "submaps.json"

    if not tfrecord_path.exists():
        print(f"[WARN] tfrecord not found: {tfrecord_path}")
        return False
    if not map_npz.exists():
        print(f"[WARN] map_static not found: {map_npz}")
        return False
    if not submaps_json.exists():
        print(f"[WARN] submaps.json not found: {submaps_json}")
        return False
    if not cam_gray_dir.exists():
        print(f"[WARN] cam_gray_dir not found: {cam_gray_dir}")
        return False

    # 1) フレームをキャッシュ
    frames: List[open_dataset.Frame] = []
    for rec in tf.data.TFRecordDataset(str(tfrecord_path), compression_type=""):
        fr = open_dataset.Frame()
        fr.ParseFromString(rec.numpy())
        frames.append(fr)

    if len(frames) == 0:
        print(f"[WARN] no frames in tfrecord: {tfrecord_path}")
        return False

    padw = max(4, len(str(len(frames) - 1)))

    # 2) map_static 読み込み
    npz = np.load(str(map_npz))
    if "xyz" not in npz or "intensity" not in npz:
        print(f"[WARN] map_static.npz missing keys: {map_npz}")
        return False
    xyz_world_all = np.asarray(npz["xyz"], dtype=np.float32)
    intensity_all = np.asarray(npz["intensity"], dtype=np.float32)
    up_val = compute_up_value(intensity_all, args.int_percentile)

    map_static_info = get_file_fingerprint(map_npz, mode=args.map_static_hash_mode, head_bytes=args.map_static_hash_bytes)
    try:
        map_static_info.update({
            'npz_keys': list(getattr(npz, 'files', [])),
            'xyz_shape': [int(x) for x in xyz_world_all.shape],
            'intensity_shape': [int(x) for x in intensity_all.shape],
            'xyz_dtype': str(xyz_world_all.dtype),
            'intensity_dtype': str(intensity_all.dtype),
        })
    except Exception as e:
        map_static_info['note'] = f'failed to add array meta: {e}'

    # 3) submaps.json 読み込み
    with open(submaps_json, "r", encoding="utf-8") as f:
        submaps_obj = json.load(f)

    if isinstance(submaps_obj, dict) and "submaps" in submaps_obj:
        submaps_list = submaps_obj["submaps"]
    elif isinstance(submaps_obj, list):
        submaps_list = submaps_obj
    else:
        print(f"[WARN] unknown submaps.json format: {submaps_json}")
        return False

    if not isinstance(submaps_list, list):
        print(f"[WARN] submaps_list is not list: {submaps_json}")
        return False

    # 4) submap のシャーディング
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("--shard-id must satisfy 0 <= shard_id < num_shards")

    submaps_shard = [sm for i, sm in enumerate(submaps_list) if (i % args.num_shards) == args.shard_id]
    if args.max_anchors is not None and int(args.max_anchors) > 0:
        submaps_shard = submaps_shard[: int(args.max_anchors)]

    ensure_dir(out_seg_dir)
    stats = {
        "segment_id": seg_stem,
        "submaps_in_shard": int(len(submaps_shard)),
        "anchors_processed": 0,
        "anchors_skipped": 0,
        "ri_generated": 0,
        "ri_skipped_done": 0,
        "ri_skipped_low_coverage": 0,
        "ri_skipped_no_points": 0,
        "k0_pose_from_cam_json": 0,
    }

    # 5) 各submap(=anchor)ごとに生成
    pbar = tqdm(submaps_shard, desc=f"[SEG] {seg_stem} (submaps shard {args.shard_id}/{args.num_shards})", leave=False)
    for sm in pbar:
        if not isinstance(sm, dict):
            stats["anchors_skipped"] += 1
            continue

        if "submap_id" in sm:
            submap_id = int(sm["submap_id"])
        elif "id" in sm:
            submap_id = int(sm["id"])
        else:
            print(f"[WARN] submap entry has no id: {seg_stem}")
            stats["anchors_skipped"] += 1
            continue

        if "anchor_frame_index" in sm:
            anchor_idx = int(sm["anchor_frame_index"])
        elif "anchor_idx" in sm:
            anchor_idx = int(sm["anchor_idx"])
        else:
            print(f"[WARN] submap entry has no anchor index: {seg_stem} submap={submap_id}")
            stats["anchors_skipped"] += 1
            continue

        if not (0 <= anchor_idx < len(frames)):
            print(f"[WARN] anchor_idx out of range: {anchor_idx} (frames={len(frames)}), skip submap {submap_id}")
            stats["anchors_skipped"] += 1
            continue

        out_anchor_dir = out_seg_dir / str(submap_id)
        if out_anchor_dir.exists() and args.overwrite:
            shutil.rmtree(out_anchor_dir)
        ensure_dir(out_anchor_dir)

        # cam.json をコピー（Step4のメタ情報をそのまま使う）
        cam_json_path = cam_gray_dir / f"{anchor_idx:0{padw}d}_cam.json"
        if not cam_json_path.exists():
            alt = cam_gray_dir / f"{anchor_idx}_cam.json"
            if alt.exists():
                cam_json_path = alt
            else:
                print(f"[WARN] cam json not found: {cam_json_path}, skip submap {submap_id}")
                stats["anchors_skipped"] += 1
                continue
        shutil.copy2(str(cam_json_path), str(out_anchor_dir / "cam.json"))

        with open(cam_json_path, "r", encoding="utf-8") as f:
            cam_meta = json.load(f)
        cam_img_path = cam_meta.get("image_path", "")

        # anchorフレームとキャリブレーション
        frame = frames[anchor_idx]
        cam_enum = camera_name_from_str(args.cam)
        calib = None
        for c in frame.context.camera_calibrations:
            if c.name == cam_enum:
                calib = c
                break
        if calib is None:
            print(f"[WARN] camera calib not found for {args.cam} in frame {anchor_idx}, skip submap {submap_id}")
            stats["anchors_skipped"] += 1
            continue

        W = int(calib.width)
        H = int(calib.height)
        fx_cal, fy_cal, cx_cal, cy_cal = intrinsic_from(calib)
        intr_default = (fx_cal, fy_cal, cx_cal, cy_cal)
        T_wv_base = mat4_list_to_np(frame.pose.transform).astype(np.float64)
        Twc_cal = world_to_cam_from_Twv(T_wv_base, calib).astype(np.float64)

        # k=0 では cam.json を使う（必要なら全 k で内部パラメータも使う）
        Twc_cam, intr_cam, pose_from_cam, intr_from_cam = read_pose_intrinsic_from_cam_json(
            cam_meta=cam_meta,
            Twc_fallback=Twc_cal,
            intr_fallback=intr_default,
        )
        if bool(args.use_cam_intrinsic) and intr_from_cam:
            fx_base, fy_base, cx_base, cy_base = intr_cam
        else:
            fx_base, fy_base, cx_base, cy_base = intr_default

        # 再開でΔがズレないように：submap単位の安定seed
        seed_text = f"{args.seed}|{seg_stem}|{submap_id}|{anchor_idx}"
        sub_seed = stable_uint32_from_text(seed_text)
        sub_rng = np.random.RandomState(sub_seed)

        # Δリスト（ri_000 は Δ=0）
        deltas: List[Tuple[float, float, float, str]] = [(0.0, 0.0, 0.0, "zero")]
        for _ in range(max(0, int(args.num_samples_per_anchor) - 1)):
            ds_raw, dl_raw, region = sample_delta_loftr_sl(sub_rng)
            ds = float(ds_raw)
            dl = float(dl_raw) * float(args.dl_max_m)
            dy_deg = sample_delta_yaw_deg(sub_rng)
            deltas.append((ds, dl, dy_deg, region))

        start_k = 0
        if args.zero_ri_mode == "skip":
            start_k = 1
        elif args.zero_ri_mode == "reuse":
            ri0_png = out_anchor_dir / "ri_000.png"
            ri0_json = out_anchor_dir / "ri_000.json"
            ri0_idx = out_anchor_dir / "ri_000_index_depth.npz"
            ri0_idx_nosplat_alias = out_anchor_dir / f"ri_000_index_depth{args.nosplat_suffix}.npz"
            ri0_idx_splat = out_anchor_dir / f"ri_000_index_depth{args.splat_suffix}.npz"
            ri0_done = ri0_png.exists() and ri0_json.exists() and (
                (not args.write_index) or (
                    ri0_idx.exists()
                    and (not args.write_nosplat_alias or ri0_idx_nosplat_alias.exists() or ri0_idx_nosplat_alias == ri0_idx)
                    and (not args.write_splatted_index_depth or ri0_idx_splat.exists())
                )
            )
            if ri0_done:
                start_k = 1
            else:
                anchor_png = find_anchor_png(submaps_dir, submap_id, args.anchor_png_name)
                if anchor_png is None:
                    print(f"[WARN] anchor.png not found for reuse: submap {submap_id} -> fallback to render ri_000")
                    start_k = 0
                else:
                    shutil.copy2(str(anchor_png), str(ri0_png))
                    if args.write_index:
                        cand = anchor_png.parent / "anchor_index.npz"
                        if cand.exists():
                            shutil.copy2(str(cand), str(ri0_idx))
                    meta0 = {
                        "segment_id": seg_stem,
                        "subset": args.subset,
                        "submap_id": submap_id,
                        "anchor_frame_index": anchor_idx,
                        "sample_index": 0,
                        "camera_name": args.cam.upper(),
                        "cam_image_path": cam_img_path,
                        "source": {
                            "tfrecord_path": str(tfrecord_path),
                            "submaps_json": str(submaps_json),
                            "cam_json": str(cam_json_path),
                            "map_static_npz": str(map_npz),
                        },
                        "map_static": map_static_info,
                        "projection": dict(PROJECTION_CONVENTION),
                        "width": int(W),
                        "height": int(H),
                        "intrinsic": [float(fx_base), float(fy_base), float(cx_base), float(cy_base)],
                        "ri_image_path": str(ri0_png),
                        "delta": {"ds_m": 0.0, "dl_m": 0.0, "dyaw_deg": 0.0, "region": "zero"},
                        "world_to_cam": (Twc_cam if args.k0_align_cam_json else Twc_cal).astype(float).tolist(),
                        "coverage": None,
                        "render_profile": {
                            "int_percentile": float(args.int_percentile),
                            "global_up_val": float(up_val),
                            "range_corr_power": float(args.range_corr_power),
                            "point_size_image": int(args.point_size),
                            "point_size_index": int(args.index_point_size),
                            "postprocess": {
                                "hist_eq": True,
                                "clahe": True,
                                "gamma": 1.0,
                                "mask_covered_only": bool(args.post_mask_covered_only),
                                "clahe_clip_limit": float(args.clahe_clip_limit),
                                "clahe_tile_grid": int(args.clahe_tile_grid),
                            },
                        },
                        "seed_info": {
                            "base_seed": int(args.seed),
                            "sub_seed": int(sub_seed),
                            "seed_text": seed_text,
                        },
                    }
                    atomic_write_json(ri0_json, meta0)
                    start_k = 1

        # 6) 描画（render）
        for k in range(start_k, len(deltas)):
            ds, dl, dy_deg, region = deltas[k]
            fname = f"ri_{k:03d}"
            png_path = out_anchor_dir / f"{fname}.png"
            json_path = out_anchor_dir / f"{fname}.json"
            idx_path = out_anchor_dir / f"{fname}_index_depth.npz"
            idx_nosplat_alias = out_anchor_dir / f"{fname}_index_depth{args.nosplat_suffix}.npz"
            idx_splat_path = out_anchor_dir / f"{fname}_index_depth{args.splat_suffix}.npz"

            done = png_path.exists() and json_path.exists() and (
                (not args.write_index) or (
                    idx_path.exists()
                    and (not args.write_nosplat_alias or idx_nosplat_alias.exists() or idx_nosplat_alias == idx_path)
                    and (not args.write_splatted_index_depth or idx_splat_path.exists())
                )
            )
            if done:
                stats["ri_skipped_done"] += 1
                continue

            # 姿勢 / intrinsic
            if int(k) == 0 and bool(args.k0_align_cam_json):
                Twc_use = Twc_cam.astype(np.float64)
                if pose_from_cam:
                    stats["k0_pose_from_cam_json"] += 1
            else:
                T_wv_new = build_T_wv_with_offset(
                    T_wv_base,
                    ds,
                    dl,
                    dy_deg,
                    preserve_roll_pitch=bool(args.preserve_roll_pitch),
                )
                Twc_use = world_to_cam_from_Twv(T_wv_new, calib).astype(np.float64)

            u, v, depth, front = project_points_world_to_image(
                xyz_world_all, Twc_use.astype(np.float32), fx_base, fy_base, cx_base, cy_base
            )
            ok = front & (u >= 0) & (u < W) & (v >= 0) & (v < H) & np.isfinite(depth)
            if not np.any(ok):
                stats["ri_skipped_no_points"] += 1
                continue

            u2 = u[ok]
            v2 = v[ok]
            depth_k = depth[ok]
            intens_k = intensity_all[ok]
            idx_k = np.where(ok)[0].astype(np.int32)

            if abs(args.range_corr_power) > 1e-9:
                intens_k = intens_k * np.power(depth_k, args.range_corr_power)
            val01 = np.clip(intens_k / (up_val + 1e-6), 0.0, 1.0).astype(np.float32)

            base_map, depth_map_splat, idx_map_splat = rasterize_zbuffer(
                W, H, u2, v2, depth_k, val01, idx_k, int(args.point_size)
            )
            if int(args.index_point_size) <= 1:
                depth_map_idx, idx_map_idx = rasterize_index_depth_nosplat(
                    W, H, u2, v2, depth_k, idx_k
                )
            elif int(args.index_point_size) == int(args.point_size):
                # 両方の point size が同じ場合は、計算済みのスプラット済みマップを再利用する。
                depth_map_idx, idx_map_idx = depth_map_splat, idx_map_splat
            else:
                # index/depth のスプラット出力を互換目的で有効化する任意モード。
                _, depth_map_idx, idx_map_idx = rasterize_zbuffer(
                    W, H, u2, v2, depth_k, val01, idx_k, int(args.index_point_size)
                )

            covered = (idx_map_splat >= 0)
            coverage = float(np.mean(covered))
            if coverage < float(args.min_coverage):
                stats["ri_skipped_low_coverage"] += 1
                continue

            img_u8 = np.clip((base_map * 255.0).round(), 0, 255).astype(np.uint8)
            mask_pp = covered if args.post_mask_covered_only else None
            img_u8 = masked_hist_equalize_uint8(img_u8, mask_pp)
            img_u8 = apply_clahe_masked_uint8(img_u8, mask_pp, args.clahe_clip_limit, args.clahe_tile_grid)

            ok_write = cv2.imwrite(str(png_path), img_u8)
            if not ok_write:
                print(f"[WARN] cv2.imwrite failed: {png_path}")
                continue

            if args.write_index:
                atomic_savez_compressed(idx_path, point_index=idx_map_idx, depth=depth_map_idx)
                if args.write_nosplat_alias and (idx_nosplat_alias != idx_path):
                    atomic_savez_compressed(idx_nosplat_alias, point_index=idx_map_idx, depth=depth_map_idx)
                if args.write_splatted_index_depth:
                    atomic_savez_compressed(idx_splat_path, point_index=idx_map_splat, depth=depth_map_splat)

            meta = {
                "segment_id": seg_stem,
                "subset": args.subset,
                "submap_id": submap_id,
                "anchor_frame_index": anchor_idx,
                "sample_index": int(k),
                "camera_name": args.cam.upper(),
                "cam_image_path": cam_img_path,
                "source": {
                    "tfrecord_path": str(tfrecord_path),
                    "submaps_json": str(submaps_json),
                    "cam_json": str(cam_json_path),
                    "map_static_npz": str(map_npz),
                },
                "map_static": map_static_info,
                "projection": dict(PROJECTION_CONVENTION),
                "width": int(W),
                "height": int(H),
                "intrinsic": [float(fx_base), float(fy_base), float(cx_base), float(cy_base)],
                "ri_image_path": str(png_path),
                "delta": {"ds_m": float(ds), "dl_m": float(dl), "dyaw_deg": float(dy_deg), "region": str(region)},
                "world_to_cam": Twc_use.astype(float).tolist(),
                "coverage": coverage,
                "render_profile": {
                    "int_percentile": float(args.int_percentile),
                    "global_up_val": float(up_val),
                    "range_corr_power": float(args.range_corr_power),
                    "point_size_image": int(args.point_size),
                    "point_size_index": int(args.index_point_size),
                    "postprocess": {
                        "hist_eq": True,
                        "clahe": True,
                        "gamma": 1.0,
                        "mask_covered_only": bool(args.post_mask_covered_only),
                        "clahe_clip_limit": float(args.clahe_clip_limit),
                        "clahe_tile_grid": int(args.clahe_tile_grid),
                    },
                },
                "seed_info": {
                    "base_seed": int(args.seed),
                    "sub_seed": int(sub_seed),
                    "seed_text": seed_text,
                },
                "compat_sampling_args": {
                    "dist_profile": str(args.dist_profile),
                    "dl_max_m": float(args.dl_max_m),
                    "dl_sampling": str(args.dl_sampling),
                    "max_resample": int(args.max_resample),
                },
            }
            atomic_write_json(json_path, meta)
            stats["ri_generated"] += 1

        stats["anchors_processed"] += 1

    if bool(args.save_summary_json):
        summary_path = out_seg_dir / "_summary_make_loftr_shifted_ri_txt_posefix_v5.json"
        summary = {
            "segment_id": seg_stem,
            "subset": args.subset,
            "stats": stats,
            "config": {
                "point_size_image": int(args.point_size),
                "point_size_index": int(args.index_point_size),
                "k0_align_cam_json": bool(args.k0_align_cam_json),
                "use_cam_intrinsic": bool(args.use_cam_intrinsic),
                "preserve_roll_pitch": bool(args.preserve_roll_pitch),
                "write_index": bool(args.write_index),
                "write_nosplat_alias": bool(args.write_nosplat_alias),
                "write_splatted_index_depth": bool(args.write_splatted_index_depth),
                "nosplat_suffix": str(args.nosplat_suffix),
                "splat_suffix": str(args.splat_suffix),
            },
        }
        atomic_write_json(summary_path, summary)

    print(f"[OK] done segment: {seg_stem} -> {out_seg_dir}")
    return True


def main():
    """CLI 引数を解釈し、入力の読み込みから結果保存までの一連の処理を実行する。"""
    ap = argparse.ArgumentParser()

    # 入力指定：単一セグメント or segments-file
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tfrecord", type=str, default=None, help="単一セグメントの .tfrecord パス")
    g.add_argument("--segments-file", type=str, default=None, help="train_clear.txt のようなセグメント一覧テキスト")

    ap.add_argument("--tfrecord-root", type=str, default="", help="segments-file がセグメント名のみの場合に使う .tfrecord ディレクトリ")

    ap.add_argument("--subset", type=str, required=True, choices=["training", "validation", "testing"])
    ap.add_argument("--maps-root", type=str, required=True)
    ap.add_argument("--submaps-root", type=str, required=True)
    ap.add_argument("--cam-gray-root", type=str, required=True)
    ap.add_argument("--out-root", type=str, required=True)

    # map_static の参照同一性を後段で確認しやすくするための情報（jsonへ保存）
    ap.add_argument("--map-static-hash-mode", type=str, choices=["none", "head", "full"], default="head",
                    help="map_static.npz の同一性確認用ハッシュ。head=先頭Nバイト, full=全体。")
    ap.add_argument("--map-static-hash-bytes", type=int, default=1048576,
                    help="hash-mode=head のときに読むバイト数（デフォルト1MiB）。")

    ap.add_argument("--cam", type=str, choices=["FRONT"], default="FRONT")

    # 露出/描画（RI画像）
    ap.add_argument("--int-percentile", type=float, default=99.5)
    ap.add_argument("--range-corr-power", type=float, default=0.0)
    ap.add_argument("--point-size", type=int, default=2, help="RI画像の描画スプラット半径（従来互換は2）")

    # index/depth 出力（GT幾何用）
    ap.add_argument("--index-point-size", type=int, default=1,
                    help="index/depth のpoint_size。幾何の一貫性のため1推奨。")
    ap.add_argument("--write-nosplat-alias", type=str2bool, default=False,
                    help="trueで *_index_depth{nosplat_suffix}.npz も追加保存。")
    ap.add_argument("--write-splatted-index-depth", type=str2bool, default=False,
                    help="trueで *_index_depth{splat_suffix}.npz を追加保存（診断用）。")
    ap.add_argument("--nosplat-suffix", type=str, default="_nosplat")
    ap.add_argument("--splat-suffix", type=str, default="_splat")

    # 姿勢/intrinsic の整合性
    ap.add_argument("--k0-align-cam-json", type=str2bool, default=True,
                    help="k=0 は cam.json の world_to_cam を優先して整合させる。")
    ap.add_argument("--use-cam-intrinsic", type=str2bool, default=True,
                    help="trueで cam.json の intrinsic を優先。falseでcalib intrinsicを使用。")
    ap.add_argument("--preserve-roll-pitch", type=str2bool, default=True,
                    help="k>0のオフセット生成で base 姿勢の roll/pitch を保持。")

    # 出力
    ap.add_argument("--write-index", type=str2bool, default=True)
    ap.add_argument("--post-mask-covered-only", type=str2bool, default=True)
    ap.add_argument("--clahe-clip-limit", type=float, default=2.0)
    ap.add_argument("--clahe-tile-grid", type=int, default=8)

    # サンプリング
    ap.add_argument("--num-samples-per-anchor", type=int, default=32)
    ap.add_argument("--min-coverage", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dist-profile", type=str, choices=["v3"], default="v3",
                    help="互換引数。現行実装では v3 のみ受理。")
    ap.add_argument("--dl-max-m", type=float, default=2.0,
                    help="dlスケール[m]。内部の正規化dlに乗算。")
    ap.add_argument("--dl-sampling", type=str, choices=["triangular", "uniform", "normal_clip"], default="triangular",
                    help="互換引数。現行実装ではメタ保存用。")
    ap.add_argument("--max-resample", type=int, default=50,
                    help="互換引数。現行実装ではメタ保存用。")
    ap.add_argument("--save-summary-json", action="store_true",
                    help="セグメントごとのサマリJSONを保存。")

    # Δ=0 の扱い
    ap.add_argument("--zero-ri-mode", type=str, choices=["reuse", "render", "skip"], default="render",
                    help="render: ri_000 も再レンダリング（推奨）。reuse: anchor.png をコピー。")
    ap.add_argument("--anchor-png-name", type=str, default="anchor.png")

    # 実行制御
    ap.add_argument("--overwrite", type=str2bool, default=False)

    # submap のシャーディング
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--max-anchors", type=int, default=-1, help="デバッグ用。-1で無制限。")

    # segments-file の segment sharding（任意：並列化用）
    ap.add_argument("--segments-num-shards", type=int, default=1)
    ap.add_argument("--segments-shard-id", type=int, default=0)

    # エラー時の挙動
    ap.add_argument("--stop-on-error", type=str2bool, default=False)

    args = ap.parse_args()

    # 方針固定: RI 画像はスプラットを維持し、index/depth は厳密に非スプラットのままにする。
    if int(args.index_point_size) != 1:
        print(f"[WARN] policy forces --index-point-size=1 (got {args.index_point_size})")
        args.index_point_size = 1
    if bool(args.write_splatted_index_depth):
        print("[WARN] policy disables --write-splatted-index-depth (forcing false)")
        args.write_splatted_index_depth = False

    # 方針固定: ri_000 は常に再描画し、再利用やスキップはしない。
    if str(args.zero_ri_mode) != "render":
        print(f"[WARN] policy forces --zero-ri-mode=render (got {args.zero_ri_mode})")
        args.zero_ri_mode = "render"

    # segments モード
    if args.segments_file is not None:
        seg_file = Path(args.segments_file)
        if not seg_file.exists():
            raise FileNotFoundError(seg_file)

        segments_all = load_segments_list(seg_file)
        if len(segments_all) == 0:
            raise RuntimeError(f"segments-file is empty: {seg_file}")

        if args.segments_num_shards <= 0:
            raise ValueError("--segments-num-shards must be >= 1")
        if not (0 <= args.segments_shard_id < args.segments_num_shards):
            raise ValueError("--segments-shard-id must satisfy 0 <= id < segments-num-shards")

        # セグメントのシャーディング
        segments = [s for i, s in enumerate(segments_all) if (i % args.segments_num_shards) == args.segments_shard_id]

        tfroot = Path(args.tfrecord_root) if args.tfrecord_root.strip() else None

        ok_count = 0
        ng_count = 0

        print(f"[INFO] segments total={len(segments_all)} shard={args.segments_shard_id}/{args.segments_num_shards} -> run={len(segments)}")
        for seg in segments:
            tfp = resolve_tfrecord_path(seg, tfroot)
            if tfp is None:
                print(f"[WARN] cannot resolve tfrecord for segment: {seg} (tfrecord-root={tfroot})")
                ng_count += 1
                if args.stop_on_error:
                    break
                continue

            print(f"\n========== SEGMENT START ==========")
            print(f"segment  : {seg}")
            print(f"tfrecord : {tfp}")
            print(f"==================================")
            ok = process_one_segment(args, tfp)
            if ok:
                ok_count += 1
            else:
                ng_count += 1
                if args.stop_on_error:
                    break

        print(f"\n[SUMMARY] ok={ok_count} ng={ng_count} (shard {args.segments_shard_id}/{args.segments_num_shards})")
        if ng_count > 0:
            raise SystemExit(1)
        raise SystemExit(0)

    # single segment モード
    tfrecord_path = Path(args.tfrecord)
    ok = process_one_segment(args, tfrecord_path)
    if not ok:
        raise SystemExit(1)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
