# 使い方:
#   python eval_cross_match_to_pose_zalign_multi_visall_v2.py --help
#   必要な入力パスと出力先を引数で指定して実行する。


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 目的: クロス評価で LoFTR マッチング、PnP 姿勢推定、可視化、集計を行う。
"""
eval_cross_match_to_pose.py

Cross-eval（GT v3）用の推論スクリプト。LoFTR で対応点を求め、PnP でカメラ姿勢を推定する。

- image0: anchor / RI（反射強度画像）   => backbone0
- image1: query camera grayscale        => backbone1

manifest 内の各 pair について、以下を順に実行する:
  1) 画像をリサイズする（アスペクト比維持、長辺 = --resize-long、かつ --stride で割り切れるサイズへ調整）
  2) matcher（SFPPRLoFTR）を実行する
  3) 2D-3D 対応を構築する:
       - 2D: 予測マッチから得た query 画素
       - 3D: anchor の depth/index map と anchor pose から復元した world 座標点
         注: 投影・逆投影には GT 生成時と同じ x-forward 規約を使う:
              u = fx * (-Y/X) + cx
              v = fy * (-Z/X) + cy
         ここで X は *_index_depth.npz に保存されている「depth」
  4) solvePnPRansac（OpenCV）で query 姿勢を推定し、必要なら追加で refine する
  5) 可能であれば q_meta_json 内の GT 姿勢と比較する

出力:
  - out_dir/per_pair.csv
  - out_dir/per_query.csv（各 query について最良候補を保存。判定基準は inlier 数、その後 reproj 誤差）
  - out_dir/summary.json

"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from loftr import default_cfg
from loftr.loftr_sfppr import SFPPRLoFTR

# ---- cv2 のスレッド制御 ----
try:
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass


# ----------------------------
# ユーティリティ
# ----------------------------

def _read_gray_u8(path: str) -> np.ndarray:
    """グレースケール画像を uint8 配列として読み込む。"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def _resize_long_keep_aspect(gray_u8: np.ndarray, long_side: int, divisor: int = 8) -> Tuple[np.ndarray, float, float]:
    """グレースケール画像を、最大辺が long_side になるよう縦横比を保って縮尺し、さらに divisor で割り切れる大きさへ切り下げる。"""
    H, W = gray_u8.shape[:2]
    if long_side is None or int(long_side) <= 0:
        # それでも divisor 制約は維持する
        newH = max(divisor, (H // divisor) * divisor)
        newW = max(divisor, (W // divisor) * divisor)
    else:
        s = float(long_side) / float(max(H, W))
        newH = int(round(H * s))
        newW = int(round(W * s))
        newH = max(divisor, (newH // divisor) * divisor)
        newW = max(divisor, (newW // divisor) * divisor)

    if newH == H and newW == W:
        return gray_u8, 1.0, 1.0

    resized = cv2.resize(gray_u8, (newW, newH), interpolation=cv2.INTER_AREA)
    sx = float(newW) / float(W)
    sy = float(newH) / float(H)
    return resized, sx, sy


def _to_tensor_gray01(gray_u8: np.ndarray) -> torch.Tensor:
    """(H,W) の uint8 画像を、[0,1] 範囲の (1,H,W) float32 テンソルへ変換する。"""
    t = torch.from_numpy(gray_u8).float() / 255.0
    return t.unsqueeze(0)


def _safe_float_list(x: Any, n: int) -> List[float]:
    """数列を float の list として安全に取り出す。"""
    if isinstance(x, (list, tuple)) and len(x) >= n:
        return [float(x[i]) for i in range(n)]
    raise ValueError(f"Expected list/tuple of length >= {n}, got: {type(x)} len={len(x) if isinstance(x,(list,tuple)) else 'n/a'}")


def _parse_intrinsic(meta: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """GT 生成時と同じ規則で内部パラメータを解釈する。

    対応形式:
      - meta['intrinsic'] = [fx, fy, cx, cy]
      - meta['cam_intrinsic'] = [fx, fy, cx, cy, k1, k2, p1, p2, k3]（Waymo 形式） -> 先頭 4 要素を使う
      - 3x3 の K を一次元化したもの（len==9）: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
      - ネストした 3x3 の list/array、または np.ndarray(3,3)
      - fx/fy/cx/cy キーを持つ dict
    """
    if meta is None:
        raise ValueError("meta is None")

    intr = meta.get("intrinsic", None)
    if intr is None and "cam_intrinsic" in meta:
        intr = meta["cam_intrinsic"]
    if intr is None:
        for k in ("K", "camera_intrinsic", "camera_matrix"):
            if k in meta:
                intr = meta[k]
                break
    if intr is None:
        raise KeyError(f"intrinsic not found in meta keys={list(meta.keys())[:40]}")

    if isinstance(intr, dict):
        for keys in [("fx", "fy", "cx", "cy"), ("f_x", "f_y", "c_x", "c_y")]:
            if all(k in intr for k in keys):
                return (
                    float(intr[keys[0]]),
                    float(intr[keys[1]]),
                    float(intr[keys[2]]),
                    float(intr[keys[3]]),
                )
        if "K" in intr:
            intr = intr["K"]
        else:
            raise ValueError(f"intrinsic dict missing fx/fy/cx/cy. keys={list(intr.keys())}")

    if (
        isinstance(intr, (list, tuple))
        and len(intr) == 3
        and all(isinstance(r, (list, tuple)) and len(r) == 3 for r in intr)
    ):
        K = __import__("numpy").array(intr, dtype=float)
        return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    import numpy as np
    arr = np.array(intr, dtype=np.float64)
    if arr.shape == (3, 3):
        return float(arr[0, 0]), float(arr[1, 1]), float(arr[0, 2]), float(arr[1, 2])

    vals = arr.reshape(-1)
    if vals.size == 4:
        return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])

    if vals.size >= 9:
        if (
            abs(float(vals[8]) - 1.0) < 1e-3
            and abs(float(vals[1])) < 1e-3
            and abs(float(vals[3])) < 1e-3
            and abs(float(vals[6])) < 1e-3
            and abs(float(vals[7])) < 1e-3
        ):
            return float(vals[0]), float(vals[4]), float(vals[2]), float(vals[5])
        return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])

    if vals.size > 4:
        return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])

    raise ValueError(f"intrinsic format unsupported: shape={arr.shape} vals={vals.tolist()}")


def _parse_Twc(meta: Dict[str, Any]) -> np.ndarray:
    """各種表現からカメラ姿勢行列 Twc を復元する。"""
    if "world_to_cam" in meta:
        arr = np.array(meta["world_to_cam"], dtype=np.float64).reshape(-1)
        if arr.size == 16:
            return arr.reshape(4, 4)
    # あり得る別名も試す
    for k in ["T_wc", "T_w2c", "Twc", "w2c"]:
        if k in meta:
            arr = np.array(meta[k], dtype=np.float64).reshape(-1)
            if arr.size == 16:
                return arr.reshape(4, 4)
    raise KeyError("Cannot find 4x4 world_to_cam in json.")


def _invert_T(T: np.ndarray) -> np.ndarray:
    """4x4 変換行列を反転する。"""
    R = T[:3, :3]
    t = T[:3, 3:4]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3:4] = -R.T @ t
    return Ti


def _transform_points(T: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """4x4 変換を適用し、xyz (N,3) を (N,3) へ写す。"""
    N = int(xyz.shape[0])
    xyz_h = np.concatenate([xyz.astype(np.float64), np.ones((N, 1), dtype=np.float64)], axis=1)
    out = xyz_h @ T.T
    return out[:, :3].astype(np.float64)


def backproject_xforward(u: np.ndarray, v: np.ndarray, depth_x: np.ndarray,
                         fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    x-forward pinhole:
      u = fx * (-Y/X) + cx
      v = fy * (-Z/X) + cy
    given u,v and X (=depth_x), recover (X,Y,Z) in camera coords (X forward, Y left, Z up).
    """
    X = depth_x
    Y = -(u - cx) * X / fx
    Z = -(v - cy) * X / fy
    return np.stack([X, Y, Z], axis=1)


def project_xforward(xyz_cam_xf: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    x-forward projection:
      u = fx * (-Y/X) + cx
      v = fy * (-Z/X) + cy
    """
    X = xyz_cam_xf[:, 0]
    Y = xyz_cam_xf[:, 1]
    Z = xyz_cam_xf[:, 2]
    eps = 1e-9
    u = fx * (-Y / (X + eps)) + cx
    v = fy * (-Z / (X + eps)) + cy
    return u, v


def _wrap_deg(x: float) -> float:
    """角度を [-180, 180) に折りたたむ。"""
    y = (x + 180.0) % 360.0 - 180.0
    return float(y)


def _rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Ra と Rb（どちらも 3x3）間の回転角を度単位で返す。"""
    R = Ra @ Rb.T
    tr = float(np.trace(R))
    c = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    return float(math.degrees(math.acos(c)))


def _camera_center_world(Twc: np.ndarray) -> np.ndarray:
    """カメラ姿勢からワールド座標系でのカメラ中心を求める。"""
    R = Twc[:3, :3]
    t = Twc[:3, 3]
    C = -R.T @ t
    return C.astype(np.float64)


def _forward_axis_world_from_Twc_xforward(Twc_xf: np.ndarray) -> np.ndarray:
    """
    x-forward カメラ座標系では、前方軸は +X_cam = [1,0,0] である。
    ワールド座標系での前方ベクトルは R_cw * [1,0,0] で、R_cw = R_wc^T とする。
    """
    R_wc = Twc_xf[:3, :3]
    fwd = R_wc.T @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    n = float(np.linalg.norm(fwd) + 1e-12)
    return fwd / n


# x-forward カメラ座標系と OpenCV カメラ座標系の固定回転
# p_cv = S * p_xf。ここで p_cv = [右方向, 下方向, 前方向]
S_XF_TO_CV = np.array([[0.0, -1.0,  0.0],
                       [0.0,  0.0, -1.0],
                       [1.0,  0.0,  0.0]], dtype=np.float64)
S_CV_TO_XF = S_XF_TO_CV.T


def _Twc_cv_to_xf(Twc_cv: np.ndarray) -> np.ndarray:
    """OpenCV カメラ軸で表された world_to_cam を x-forward カメラ軸へ変換する。"""
    R_cv = Twc_cv[:3, :3]
    t_cv = Twc_cv[:3, 3]
    R_xf = S_CV_TO_XF @ R_cv
    t_xf = S_CV_TO_XF @ t_cv
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R_xf
    out[:3, 3] = t_xf
    return out


def _id_to_pixel_xy(ids: np.ndarray, hw: Tuple[int, int], stride: int, cell_point: str) -> Tuple[np.ndarray, np.ndarray]:
    """平坦化された coarse id を、リサイズ後画像座標系の pixel x,y へ変換する。"""
    H, W = int(hw[0]), int(hw[1])
    Hc, Wc = H // int(stride), W // int(stride)
    ids = ids.astype(np.int64, copy=False)
    iy = (ids // Wc).astype(np.int64)
    ix = (ids % Wc).astype(np.int64)

    if cell_point == "center":
        x = (ix.astype(np.float32) + 0.5) * float(stride)
        y = (iy.astype(np.float32) + 0.5) * float(stride)
    else:
        x = ix.astype(np.float32) * float(stride)
        y = iy.astype(np.float32) * float(stride)
    return x, y


def _ensure_dir(p: str) -> None:
    """出力先ディレクトリを作成して存在を保証する。"""
    Path(p).mkdir(parents=True, exist_ok=True)


def _sanitize_for_filename(s: str, max_len: int = 120) -> str:
    """任意文字列をファイル名に安全な表現へ変換する。"""
    s = str(s)
    s = re.sub(r"[^0-9a-zA-Z_\-\.]+", "_", s)
    if len(s) <= max_len:
        return s
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
    head = s[:max_len - 1 - len(h)]
    return f"{head}_{h}"


def _stable_seed_from_string(s: str) -> int:
    """文字列から再現可能な乱数シードを生成する。"""
    h = hashlib.sha1(str(s).encode("utf-8")).hexdigest()[:8]
    return int(h, 16)


def _parse_dist_bins(s: str) -> List[float]:
    """距離ビン設定を数値境界列へ変換する。"""
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    vals = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        vals.append(float(p))
    vals = sorted(list({float(v) for v in vals}))
    return vals


def _fmt_bin(v: float) -> str:
    """距離ビンの表示文字列を作る。"""
    s = f"{float(v):g}"
    return s.replace(".", "p").replace("-", "m")


def _dist_bin_label(dist: float, bins: List[float]) -> str:
    """距離値に対応するビンラベルを返す。"""
    if not np.isfinite(dist):
        return "dist_unknown"
    if not bins or len(bins) < 2:
        return "dist_all"
    for i in range(len(bins) - 1):
        lo = bins[i]
        hi = bins[i + 1]
        if dist >= lo and dist < hi:
            return f"dist_{_fmt_bin(lo)}_{_fmt_bin(hi)}"
    if dist < bins[0]:
        return f"dist_lt_{_fmt_bin(bins[0])}"
    return f"dist_ge_{_fmt_bin(bins[-1])}"


def _draw_text_with_outline(img: np.ndarray, text: str, org: Tuple[int, int],
                            font_scale: float = 0.5, thickness: int = 1) -> None:
    """可視化画像へ縁取り付きテキストを描画する。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = int(org[0]), int(org[1])
    # 輪郭（黒）
    cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # 本体（白）
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _visualize_matches_gt(
    img0_gray: np.ndarray,
    img1_gray: np.ndarray,
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    correct_mask: np.ndarray,
    out_path: str,
    pair_title_lines: List[str],
    max_draw: int = 0,
    seed_key: str = "",
    line_thickness: int = 1,
    point_radius: int = 2,
    draw_text: bool = True,
    layout: str = "horizontal",
) -> None:
    """
    Save a match visualization:
      - horizontal: left=img0(anchor), right=img1(query)
      - vertical:   top=img0(anchor), bottom=img1(query)
      green: correct w.r.t GT reprojection
      red:   incorrect (also includes invalid depth/behind-camera)
    """
    H0, W0 = img0_gray.shape[:2]
    H1, W1 = img1_gray.shape[:2]

    # BGR へ変換する
    img0_bgr = cv2.cvtColor(img0_gray, cv2.COLOR_GRAY2BGR)
    img1_bgr = cv2.cvtColor(img1_gray, cv2.COLOR_GRAY2BGR)

    layout = str(layout).strip().lower()
    if layout not in ("horizontal", "vertical"):
        layout = "horizontal"

    if layout == "vertical":
        H = H0 + H1
        W = max(W0, W1)
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[:H0, :W0] = img0_bgr
        canvas[H0:H0 + H1, :W1] = img1_bgr
    else:
        H = max(H0, H1)
        W = W0 + W1
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[:H0, :W0] = img0_bgr
        canvas[:H1, W0:W0 + W1] = img1_bgr

    N = int(mkpts0.shape[0])
    if N == 0:
        if bool(draw_text):
            for i, t in enumerate(pair_title_lines[:8]):
                _draw_text_with_outline(canvas, t, (10, 20 + 18 * i), font_scale=0.55, thickness=1)
        _ensure_dir(str(Path(out_path).parent))
        cv2.imwrite(out_path, canvas)
        return

    # 描画対象 index を決める
    idx = np.arange(N, dtype=np.int64)
    if max_draw is not None and int(max_draw) > 0 and N > int(max_draw):
        rng = np.random.default_rng(_stable_seed_from_string(seed_key))
        idx = rng.choice(idx, size=int(max_draw), replace=False)
        idx = np.sort(idx)

    # 描画する
    for k in idx:
        x0, y0 = float(mkpts0[k, 0]), float(mkpts0[k, 1])
        x1, y1 = float(mkpts1[k, 0]), float(mkpts1[k, 1])
        col = (0, 255, 0) if bool(correct_mask[k]) else (0, 0, 255)  # BGR: green/red
        p0 = (int(round(x0)), int(round(y0)))
        if layout == "vertical":
            p1 = (int(round(x1)), int(round(y1 + H0)))
        else:
            p1 = (int(round(x1 + W0)), int(round(y1)))
        cv2.line(canvas, p0, p1, col, max(1, int(line_thickness)), cv2.LINE_AA)
        cv2.circle(canvas, p0, max(1, int(point_radius)), col, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, max(1, int(point_radius)), col, -1, cv2.LINE_AA)

    # タイトル
    if bool(draw_text):
        for i, t in enumerate(pair_title_lines[:10]):
            _draw_text_with_outline(canvas, t, (10, 20 + 18 * i), font_scale=0.55, thickness=1)

    _ensure_dir(str(Path(out_path).parent))
    cv2.imwrite(out_path, canvas)


# ----------------------------
# ICP 処理 の解析（任意）
# ----------------------------

@dataclass
class IcpDBtoQ:
    """DB から query への ICP 変換と関連メタ情報を保持する。"""
    table: Dict[Tuple[str, str], np.ndarray]

    def get(self, db_seg: str, q_seg: str) -> Optional[np.ndarray]:
        """セグメント対に対応する変換や設定値を取り出す。"""
        return self.table.get((str(db_seg), str(q_seg)))


def _try_parse_mat_from_string(s: str) -> Optional[np.ndarray]:
    """文字列化された行列を可能なら NumPy 行列へ変換する。"""
    s2 = str(s).strip()
    if not s2:
        return None
    # JSON 配列、または "a b c ..." / "a, b, c" 形式を許容する
    nums = None
    try:
        obj = json.loads(s2)
        if isinstance(obj, (list, tuple)):
            arr = np.array(obj, dtype=np.float64).reshape(-1)
            if arr.size == 16:
                return arr.reshape(4, 4)
    except Exception:
        pass
    parts = re.split(r"[,\s]+", s2)
    try:
        arr = np.array([float(x) for x in parts if x != ""], dtype=np.float64).reshape(-1)
        if arr.size == 16:
            return arr.reshape(4, 4)
        if arr.size == 12:
            # 3x4
            M = np.eye(4, dtype=np.float64)
            M[:3, :4] = arr.reshape(3, 4)
            return M
    except Exception:
        return None
    return None


def load_icp_db_to_q(
    icp_csv: str,
    db_col: Optional[str] = None,
    q_col: Optional[str] = None,
    mat_col: Optional[str] = None,
    mat_cols: Optional[List[str]] = None,
) -> IcpDBtoQ:
    """
    CSV から db->q の 4x4 変換を読み込む。

    対応形式（自動判定）:
      1) 16 個の float / JSON 配列を持つ単一列（例: 'T_db_to_q'）
      2) 16 個の数値列（T00..T33 など）
      3) 12 個の数値列（r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz）

    自動判定を上書きしたい場合は次を使う:
      --icp-db-col / --icp-q-col / --icp-mat-col / --icp-mat-cols
    """
    df = pd.read_csv(icp_csv)
    cols = list(df.columns)

    def _auto_seg_col(kind: str) -> Optional[str]:
        """DataFrame からセグメント列名を自動判定する。"""
        kind = kind.lower()
        cands = []
        for c in cols:
            cl = c.lower()
            if kind in cl and ("seg" in cl or "segment" in cl):
                cands.append(c)
        if cands:
            # まず完全一致を優先する
            for pref in [f"{kind}_seg", f"{kind}_segment", kind]:
                for c in cands:
                    if c.lower() == pref:
                        return c
            return cands[0]
        return None

    dbc = db_col or _auto_seg_col("db") or ("db_seg" if "db_seg" in cols else None)
    qc = q_col or _auto_seg_col("q") or ("q_seg" if "q_seg" in cols else None)
    if dbc is None or qc is None:
        raise ValueError(f"Could not detect db/q segment columns in icp csv. columns={cols}")

    # 行列表現を検出する
    if mat_cols is None and mat_col is None:
        # まず一般的な単一列候補を試す
        for c in cols:
            cl = c.lower()
            if ("db" in cl and "q" in cl and ("t" in cl or "mat" in cl or "se3" in cl)) or cl in ["t_db_to_q", "t_db2q", "t_db_q", "t_db_to_q_4x4"]:
                mat_col = c
                break

    table: Dict[Tuple[str, str], np.ndarray] = {}

    if mat_cols is not None and len(mat_cols) == 16:
        missing = [c for c in mat_cols if c not in cols]
        if missing:
            raise ValueError(f"--icp-mat-cols has missing columns: {missing}")
        for _, r in df.iterrows():
            dbs = str(r[dbc])
            qs = str(r[qc])
            arr = np.array([float(r[c]) for c in mat_cols], dtype=np.float64)
            T = arr.reshape(4, 4)
            table[(dbs, qs)] = T
        return IcpDBtoQ(table)

    if mat_col is not None and mat_col in cols:
        for _, r in df.iterrows():
            dbs = str(r[dbc])
            qs = str(r[qc])
            T = _try_parse_mat_from_string(r[mat_col])
            if T is None:
                continue
            table[(dbs, qs)] = T
        if table:
            return IcpDBtoQ(table)

    # 16 列形式を自動判定する（T00..T33 など）
    # 行列らしい 16 列を探す
    def _find_mat16_cols() -> Optional[List[str]]:
        # 厳密に T00..T33 を探す
        """16 要素行列を表す列群を検出する。"""
        patt_sets = [
            [f"T{i}{j}" for i in range(4) for j in range(4)],
            [f"t{i}{j}" for i in range(4) for j in range(4)],
            [f"m{i}{j}" for i in range(4) for j in range(4)],
        ]
        for ps in patt_sets:
            if all(c in cols for c in ps):
                return ps
        # 小文字バリアントも許容する
        lower_map = {c.lower(): c for c in cols}
        for ps in patt_sets:
            if all(p.lower() in lower_map for p in ps):
                return [lower_map[p.lower()] for p in ps]
        return None

    mat16 = _find_mat16_cols()
    if mat16 is not None:
        for _, r in df.iterrows():
            dbs = str(r[dbc])
            qs = str(r[qc])
            arr = np.array([float(r[c]) for c in mat16], dtype=np.float64).reshape(4, 4)
            table[(dbs, qs)] = arr
        return IcpDBtoQ(table)

    # 3x4 の RT 列も自動判定する
    rt12_cands = ["r00","r01","r02","tx","r10","r11","r12","ty","r20","r21","r22","tz"]
    if all(c in cols for c in rt12_cands):
        for _, r in df.iterrows():
            dbs = str(r[dbc])
            qs = str(r[qc])
            arr = np.array([float(r[c]) for c in rt12_cands], dtype=np.float64).reshape(3, 4)
            T = np.eye(4, dtype=np.float64)
            T[:3, :4] = arr
            table[(dbs, qs)] = T
        return IcpDBtoQ(table)

    # yaw+translation 形式も自動判定する（2D ICP でよくある）
    # 注意: この分岐は元々 SE(2) ICP 結果向けに作られている。
    # 大きな一定 z オフセットを避けるため、tz があればそれも使うよう拡張している。
    def _pick_col(prefer: List[str], fallback_contains: List[str]) -> Optional[str]:
        # 1) 完全一致（大文字小文字は無視）
        """候補列名の中から実在する列を選ぶ。"""
        lower_map = {c.lower(): c for c in cols}
        for k in prefer:
            if k.lower() in lower_map:
                return lower_map[k.lower()]
        # 2) 部分一致（error 列を拾わないようにする）
        cands = []
        for c in cols:
            cl = c.lower()
            if any(s in cl for s in fallback_contains) and ("err" not in cl) and ("error" not in cl):
                cands.append(c)
        # 'final' を含む列を優先する
        for c in cands:
            if "final" in c.lower():
                return c
        return cands[0] if cands else None

    yaw_c = _pick_col(["yaw", "yaw_deg", "final_yaw_deg", "final_yaw", "yaw_rad"], ["yaw"])
    tx_c  = _pick_col(["tx", "t_x", "x", "dx", "final_tx"], ["tx", "_tx"])
    ty_c  = _pick_col(["ty", "t_y", "y", "dy", "final_ty"], ["ty", "_ty"])
    tz_c  = _pick_col(["tz", "t_z", "z", "dz", "final_tz"], ["tz", "_tz"])

    if yaw_c is not None and tx_c is not None and ty_c is not None:
        # 列名に 'deg' が含まれていれば度とみなす
        is_deg = ("deg" in yaw_c.lower())
        for _, r in df.iterrows():
            dbs = str(r[dbc])
            qs = str(r[qc])
            yaw = float(r[yaw_c])
            if is_deg:
                yaw = math.radians(yaw)

            # +Z 軸まわりの回転
            c = math.cos(yaw); s = math.sin(yaw)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = np.array([[c, -s, 0.0],
                                  [s,  c, 0.0],
                                  [0.0, 0.0, 1.0]], dtype=np.float64)
            T[0, 3] = float(r[tx_c])
            T[1, 3] = float(r[ty_c])
            if tz_c is not None:
                T[2, 3] = float(r[tz_c])
            table[(dbs, qs)] = T
        if table:
            return IcpDBtoQ(table)

    raise ValueError(
        "Could not parse ICP CSV into db->q 4x4 matrices.\n"
        f"Columns: {cols}\n"
        "Try specifying --icp-mat-col or --icp-mat-cols explicitly."
    )


# ----------------------------
# 深度 / index の読み込み
# ----------------------------

@dataclass
class AnchorIndex:
    """アンカー画像に対応するインデックス・深度・姿勢情報を束ねる。"""
    depth_x: np.ndarray         # (H,W) float32
    valid_mask: np.ndarray      # (H,W) bool


def load_anchor_index_npz(npz_path: str) -> AnchorIndex:
    """アンカー画像に対応する index/depth 情報を読み込む。"""
    z = np.load(npz_path)
    # depth キー
    depth = None
    for k in ["depth", "depth_x", "depth_map", "dep"]:
        if k in z:
            depth = z[k]
            break
    if depth is None:
        raise KeyError(f"Cannot find depth in npz: {npz_path}, keys={list(z.keys())}")
    depth = depth.astype(np.float32, copy=False)
    # 妥当性キー（point_index/pid）
    pid = None
    for k in ["point_index", "pid", "point_id", "index"]:
        if k in z:
            pid = z[k]
            break
    if pid is not None:
        pid = pid.astype(np.int64, copy=False)
        valid = (pid >= 0)
    else:
        valid = np.isfinite(depth)

    valid = valid & np.isfinite(depth) & (depth > 0.0)
    return AnchorIndex(depth_x=depth, valid_mask=valid)


# ----------------------------
# モデル読み込みとマッチング
# ----------------------------

def _looks_like_dual_backbone(sd: Dict[str, torch.Tensor]) -> bool:
    """経験則として、dual-backbone checkpoint には 'matcher.backbone0.*' と 'matcher.backbone1.*' のようなキーが入る。"""
    keys = list(sd.keys())
    has0 = any(k.endswith("backbone0.pos_encoding.pe") or ".backbone0." in k for k in keys)
    has1 = any(k.endswith("backbone1.pos_encoding.pe") or ".backbone1." in k for k in keys)
    return bool(has0 and has1)


def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """state_dict などの共通接頭辞を取り除く。"""
    out = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out


def _convert_single_to_dual_backbone(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    single-backbone の state_dict を dual-backbone 形式へ変換する。
    backbone.* を backbone0.* と backbone1.* へ複製する。
    旧 checkpoint 向けの best-effort な補助関数である。
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if ".backbone0." in k or ".backbone1." in k:
            out[k] = v
            continue
        if ".backbone." in k:
            out[k.replace(".backbone.", ".backbone0.")] = v
            out[k.replace(".backbone.", ".backbone1.")] = v
            continue
        if k.startswith("backbone."):
            out[k.replace("backbone.", "backbone0.")] = v
            out[k.replace("backbone.", "backbone1.")] = v
            continue
        out[k] = v
    return out


def _convert_dual_to_single_backbone(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    dual-backbone の state_dict を single-backbone 形式へ変換する。
    backbone0.* を backbone.* として使い、backbone1.* は捨てる。
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if ".backbone0." in k:
            out[k.replace(".backbone0.", ".backbone.")] = v
        elif k.startswith("backbone0."):
            out[k.replace("backbone0.", "backbone.")] = v
        elif ".backbone1." in k or k.startswith("backbone1."):
            # 捨てる
            continue
        else:
            out[k] = v
    return out


def load_ckpt_flexible(matcher: torch.nn.Module, ckpt_obj: Any) -> Tuple[List[str], List[str]]:
    """
    よくある state_dict のラッパー / 接頭辞 / backbone バリアントを吸収しながら、checkpoint を matcher に読み込む。
    戻り値は load_state_dict の (missing_keys, unexpected_keys)。
    """
    # ラッパーを外す
    sd: Any
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
        sd = ckpt_obj["state_dict"]
    elif isinstance(ckpt_obj, dict) and "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        sd = ckpt_obj["model"]
    elif isinstance(ckpt_obj, dict):
        # すでに state_dict かもしれない
        sd = {k: v for k, v in ckpt_obj.items() if torch.is_tensor(v)}
        if not sd:
            sd = ckpt_obj
    else:
        sd = ckpt_obj

    if not isinstance(sd, dict):
        missing, unexpected = matcher.load_state_dict(sd, strict=False)
        return missing, unexpected

    # 共通接頭辞を取り除く
    if any(k.startswith("matcher.") for k in sd.keys()):
        sd = _strip_prefix(sd, "matcher.")

    matcher_dual = _looks_like_dual_backbone(matcher.state_dict())
    sd_dual = _looks_like_dual_backbone(sd)

    if matcher_dual and not sd_dual:
        sd = _convert_single_to_dual_backbone(sd)
    elif (not matcher_dual) and sd_dual:
        sd = _convert_dual_to_single_backbone(sd)

    missing, unexpected = matcher.load_state_dict(sd, strict=False)
    return missing, unexpected


def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """辞書を再帰的に更新する。"""
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def load_matcher(
    ckpt_path: str,
    device: torch.device,
    enable_fine: bool,
    enable_repeatability: bool = False,
) -> SFPPRLoFTR:
    """推論に使う特徴マッチャを読み込んで初期化する。"""
    import copy
    ckpt = torch.load(ckpt_path, map_location="cpu")

    cfg = copy.deepcopy(default_cfg)
    if isinstance(ckpt, dict) and "config" in ckpt and isinstance(ckpt["config"], dict):
        cfg = deep_update(cfg, ckpt["config"])

    matcher = SFPPRLoFTR(
        config=cfg,
        enable_fine=bool(enable_fine),
        enable_repeatability=bool(enable_repeatability),
    )

    missing, unexpected = load_ckpt_flexible(matcher, ckpt)
    if len(unexpected) > 0:
        print(f"[WARN] unexpected keys in ckpt: {unexpected[:8]} ... ({len(unexpected)})")
    if len(missing) > 0:
        print(f"[WARN] missing keys in ckpt: {missing[:8]} ... ({len(missing)})")

    matcher = matcher.to(device)
    matcher.eval()
    return matcher


def infer_matches(
    matcher: SFPPRLoFTR,
    img0_u8: np.ndarray,
    img1_u8: np.ndarray,
    resize_long: int,
    stride: int,
    cell_point: str,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    match_source: str,
    conf_th: float,
    max_matches: int,
) -> Dict[str, Any]:
    """
    次を含む辞書を返す:
      - mkpts0: 元の image0 画素座標系にある (Nx2)
      - mkpts1: 元の image1 画素座標系にある (Nx2)
      - conf  : (N,) の float32（利用できなければ 0 埋め）
      - aux   : サイズ / スケール / 生の件数情報
    """
    H0o, W0o = img0_u8.shape[:2]
    H1o, W1o = img1_u8.shape[:2]

    # matcher 用にリサイズする
    img0_r, sx0, sy0 = _resize_long_keep_aspect(img0_u8, resize_long, divisor=stride)
    img1_r, sx1, sy1 = _resize_long_keep_aspect(img1_u8, resize_long, divisor=stride)
    H0, W0 = img0_r.shape[:2]
    H1, W1 = img1_r.shape[:2]

    t0 = _to_tensor_gray01(img0_r).to(device, non_blocking=True)
    t1 = _to_tensor_gray01(img1_r).to(device, non_blocking=True)

    b: Dict[str, Any] = {"image0": t0.unsqueeze(0), "image1": t1.unsqueeze(0)}

    # autocast（既定では CUDA のみ）
    if amp_enabled:
        autocast_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True)
    else:
        autocast_ctx = torch.autocast(device_type=device.type, enabled=False)

    with torch.inference_mode():
        with autocast_ctx:
            matcher(b)

    mk0 = None
    mk1 = None
    conf = None

    def _np(k: str) -> Optional[np.ndarray]:
        """辞書中の値を NumPy 配列へ変換して返す。"""
        if k not in b:
            return None
        x = b[k]
        if torch.is_tensor(x):
            x = x.detach().to('cpu').numpy()
        return np.array(x)

    if match_source in ["auto", "fine"]:
        mk0_f = _np("mkpts0_f")
        mk1_f = _np("mkpts1_f")
        if mk0_f is not None and mk1_f is not None and mk0_f.size > 0 and mk1_f.size > 0:
            mk0 = mk0_f.reshape(-1, 2).astype(np.float32, copy=False)
            mk1 = mk1_f.reshape(-1, 2).astype(np.float32, copy=False)
            conf = _np("mconf")
            if conf is not None:
                conf = conf.reshape(-1).astype(np.float32, copy=False)

    if (mk0 is None or mk1 is None) and match_source in ["auto", "coarse"]:
        i_ids = _np("i_ids")
        j_ids = _np("j_ids")
        if i_ids is None or j_ids is None:
            mk0 = np.zeros((0, 2), dtype=np.float32)
            mk1 = np.zeros((0, 2), dtype=np.float32)
            conf = np.zeros((0,), dtype=np.float32)
        else:
            i_ids = i_ids.reshape(-1).astype(np.int64, copy=False)
            j_ids = j_ids.reshape(-1).astype(np.int64, copy=False)

            # 重複を避けるため (i,j) で一意化する
            order = None
            uniq_mask = None
            if i_ids.size > 0:
                key = i_ids * (H1 // stride * (W1 // stride)) + j_ids
                order = np.argsort(key)
                key_s = key[order]
                i_s = i_ids[order]
                j_s = j_ids[order]
                uniq_mask = np.ones_like(key_s, dtype=bool)
                uniq_mask[1:] = key_s[1:] != key_s[:-1]
                i_u = i_s[uniq_mask]
                j_u = j_s[uniq_mask]
            else:
                i_u = i_ids
                j_u = j_ids

            x0r, y0r = _id_to_pixel_xy(i_u, (H0, W0), stride, cell_point)
            x1r, y1r = _id_to_pixel_xy(j_u, (H1, W1), stride, cell_point)
            mk0 = np.stack([x0r, y0r], axis=1).astype(np.float32)
            mk1 = np.stack([x1r, y1r], axis=1).astype(np.float32)

            conf = _np("mconf")
            if conf is not None:
                conf = conf.reshape(-1).astype(np.float32, copy=False)
                if order is not None and uniq_mask is not None and conf.size == i_ids.size:
                    conf = conf[order][uniq_mask]
                else:
                    conf = None
            if conf is None:
                cm = b.get("conf_matrix", None)
                if torch.is_tensor(cm):
                    try:
                        cm0 = cm[0].detach().float().cpu().numpy()
                        conf = cm0[i_u, j_u].astype(np.float32, copy=False)
                    except Exception:
                        conf = np.zeros((mk0.shape[0],), dtype=np.float32)
                else:
                    conf = np.zeros((mk0.shape[0],), dtype=np.float32)

    if mk0 is None or mk1 is None:
        mk0 = np.zeros((0, 2), dtype=np.float32)
        mk1 = np.zeros((0, 2), dtype=np.float32)
    if conf is None:
        conf = np.zeros((mk0.shape[0],), dtype=np.float32)

    # confidence でフィルタする
    if conf_th > 0.0 and conf.size == mk0.shape[0]:
        keep = conf >= float(conf_th)
        mk0 = mk0[keep]
        mk1 = mk1[keep]
        conf = conf[keep]

    # top-k でフィルタする
    if max_matches is not None and int(max_matches) > 0 and mk0.shape[0] > int(max_matches) and conf.size == mk0.shape[0]:
        k = int(max_matches)
        idx = np.argsort(-conf)[:k]
        mk0 = mk0[idx]
        mk1 = mk1[idx]
        conf = conf[idx]

    # 元の画像座標系へ戻す
    mk0[:, 0] /= float(sx0)
    mk0[:, 1] /= float(sy0)
    mk1[:, 0] /= float(sx1)
    mk1[:, 1] /= float(sy1)

    # 範囲内に clamp する
    mk0[:, 0] = np.clip(mk0[:, 0], 0.0, float(W0o - 1))
    mk0[:, 1] = np.clip(mk0[:, 1], 0.0, float(H0o - 1))
    mk1[:, 0] = np.clip(mk1[:, 0], 0.0, float(W1o - 1))
    mk1[:, 1] = np.clip(mk1[:, 1], 0.0, float(H1o - 1))

    return {
        "mkpts0": mk0,
        "mkpts1": mk1,
        "conf": conf.astype(np.float32, copy=False),
        "aux": {
            "orig_hw0": (H0o, W0o),
            "orig_hw1": (H1o, W1o),
            "resized_hw0": (H0, W0),
            "resized_hw1": (H1, W1),
            "scale0": (sx0, sy0),
            "scale1": (sx1, sy1),
        }
    }


# ----------------------------
# 姿勢推定
# ----------------------------

@dataclass
class PoseResult:
    """PnP 推定の結果と評価指標をまとめて保持する。"""
    success: bool
    n_match_in: int
    n_3d: int
    n_inlier: int
    inlier_ratio: float
    reproj_rmse: float
    reproj_med: float
    reproj_p95: float
    Twc_pred_xf: Optional[np.ndarray]
    C_pred: Optional[np.ndarray]


def estimate_pose_pnp_xforward(
    mkpts0: np.ndarray,  # anchor pixels (orig)
    mkpts1: np.ndarray,  # query pixels (orig)
    anchor_index: AnchorIndex,
    anchor_Twc_xf: np.ndarray,
    anchor_K: Tuple[float, float, float, float],
    query_K: Tuple[float, float, float, float],
    T_db2q: Optional[np.ndarray],
    min_depth: float,
    max_depth: float,
    ransac_reproj_th: float,
    ransac_max_iter: int,
    ransac_conf: float,
    refine: bool,
) -> PoseResult:
    """x-forward 投影系で PnP による姿勢推定を実行する。"""
    fx0, fy0, cx0, cy0 = anchor_K
    fx1, fy1, cx1, cy1 = query_K

    n_in = int(mkpts0.shape[0])
    if n_in == 0:
        return PoseResult(False, 0, 0, 0, 0.0, float("inf"), float("inf"), float("inf"), None, None)

    # anchor 画素位置で深度をサンプリングする
    H0, W0 = anchor_index.depth_x.shape[:2]
    u0 = mkpts0[:, 0]
    v0 = mkpts0[:, 1]
    ui = np.clip(np.round(u0).astype(np.int64), 0, W0 - 1)
    vi = np.clip(np.round(v0).astype(np.int64), 0, H0 - 1)

    depth_x = anchor_index.depth_x[vi, ui]
    valid = anchor_index.valid_mask[vi, ui] & np.isfinite(depth_x)
    valid = valid & (depth_x >= float(min_depth)) & (depth_x <= float(max_depth))

    if not np.any(valid):
        return PoseResult(False, n_in, 0, 0, 0.0, float("inf"), float("inf"), float("inf"), None, None)

    mk0v = mkpts0[valid]
    mk1v = mkpts1[valid]
    depth_v = depth_x[valid].astype(np.float64, copy=False)
    n_3d = int(mk0v.shape[0])
    if n_3d < 6:
        return PoseResult(False, n_in, n_3d, 0, 0.0, float("inf"), float("inf"), float("inf"), None, None)

    # anchor カメラ座標系（x-forward）での 3D 点
    xyz0_cam = backproject_xforward(mk0v[:, 0].astype(np.float64),
                                    mk0v[:, 1].astype(np.float64),
                                    depth_v,
                                    fx0, fy0, cx0, cy0)

    # world（db frame）へ変換する
    Tcw0 = _invert_T(anchor_Twc_xf)
    xyz_w = _transform_points(Tcw0, xyz0_cam)  # (N,3)

    # 任意の db->q 変換: world 点を q world frame で表す
    if T_db2q is not None:
        xyz_w = _transform_points(T_db2q, xyz_w)

    # OpenCV カメラ軸で PnP を解く
    K = np.array([[fx1, 0.0, cx1],
                  [0.0, fy1, cy1],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    obj_pts = xyz_w.astype(np.float64, copy=False).reshape(-1, 1, 3)
    img_pts = mk1v.astype(np.float64, copy=False).reshape(-1, 1, 2)

    # RANSAC による PnP
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=obj_pts,
            imagePoints=img_pts,
            cameraMatrix=K,
            distCoeffs=None,
            iterationsCount=int(ransac_max_iter),
            reprojectionError=float(ransac_reproj_th),
            confidence=float(ransac_conf),
            flags=cv2.SOLVEPNP_EPNP,
        )
    except Exception:
        ok = False
        rvec = None
        tvec = None
        inliers = None

    if not ok or rvec is None or tvec is None or inliers is None or int(inliers.size) < 6:
        return PoseResult(False, n_in, n_3d, 0, 0.0, float("inf"), float("inf"), float("inf"), None, None)

    inliers = inliers.reshape(-1).astype(np.int64, copy=False)
    n_inlier = int(inliers.size)

    # inlier に対して任意の refinement を行う
    if refine:
        try:
            obj_in = obj_pts[inliers]
            img_in = img_pts[inliers]
            ok2, rvec2, tvec2 = cv2.solvePnP(
                objectPoints=obj_in,
                imagePoints=img_in,
                cameraMatrix=K,
                distCoeffs=None,
                rvec=rvec,
                tvec=tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok2:
                rvec, tvec = rvec2, tvec2
        except Exception:
            pass

    R_cv, _ = cv2.Rodrigues(rvec)
    t_cv = tvec.reshape(3)

    Twc_cv = np.eye(4, dtype=np.float64)
    Twc_cv[:3, :3] = R_cv
    Twc_cv[:3, 3] = t_cv

    Twc_xf = _Twc_cv_to_xf(Twc_cv)
    C_pred = _camera_center_world(Twc_xf)

    # inlier 上での再投影誤差統計
    try:
        proj, _ = cv2.projectPoints(obj_pts[inliers], rvec, tvec, K, None)
        proj = proj.reshape(-1, 2)
        obs = img_pts[inliers].reshape(-1, 2)
        err = np.linalg.norm(proj - obs, axis=1).astype(np.float64)
        rmse = float(np.sqrt(np.mean(err * err))) if err.size > 0 else float("inf")
        med = float(np.median(err)) if err.size > 0 else float("inf")
        p95 = float(np.percentile(err, 95)) if err.size > 0 else float("inf")
    except Exception:
        rmse, med, p95 = float("inf"), float("inf"), float("inf")

    return PoseResult(
        success=True,
        n_match_in=n_in,
        n_3d=n_3d,
        n_inlier=n_inlier,
        inlier_ratio=float(n_inlier / max(1, n_3d)),
        reproj_rmse=rmse,
        reproj_med=med,
        reproj_p95=p95,
        Twc_pred_xf=Twc_xf,
        C_pred=C_pred,
    )


def classify_matches_by_gt_reproj(
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    anchor_index: AnchorIndex,
    anchor_Twc_xf: np.ndarray,
    anchor_K: Tuple[float, float, float, float],
    query_K: Tuple[float, float, float, float],
    Twc_gt_xf: np.ndarray,
    T_db2q: Optional[np.ndarray],
    min_depth: float,
    max_depth: float,
    gt_reproj_th_px: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Classify each match as correct/incorrect by GT reprojection.

    Steps:
      - sample depth X at anchor pixels (mkpts0)
      - backproject to anchor camera (x-forward)
      - transform to world (db), optionally to query-world via T_db2q
      - project into query image using GT pose (Twc_gt_xf) with x-forward projection
      - compare to mkpts1 (query matched pixel)
      - correct if pixel error <= gt_reproj_th_px

    Returns:
      correct_mask: (N,) bool
      err_px: (N,) float64 (inf if invalid)
      valid_3d: (N,) bool  (depth valid and within range, and GT projection has X>0)
    """
    N = int(mkpts0.shape[0])
    correct = np.zeros((N,), dtype=bool)
    err_px = np.full((N,), np.inf, dtype=np.float64)
    valid_3d = np.zeros((N,), dtype=bool)

    if N == 0:
        return correct, err_px, valid_3d

    fx0, fy0, cx0, cy0 = anchor_K
    fx1, fy1, cx1, cy1 = query_K

    H0, W0 = anchor_index.depth_x.shape[:2]
    u0 = mkpts0[:, 0]
    v0 = mkpts0[:, 1]
    ui = np.clip(np.round(u0).astype(np.int64), 0, W0 - 1)
    vi = np.clip(np.round(v0).astype(np.int64), 0, H0 - 1)

    depth_x = anchor_index.depth_x[vi, ui]
    vmask = anchor_index.valid_mask[vi, ui] & np.isfinite(depth_x)
    vmask = vmask & (depth_x >= float(min_depth)) & (depth_x <= float(max_depth))

    if not np.any(vmask):
        return correct, err_px, valid_3d

    mk0v = mkpts0[vmask]
    mk1v = mkpts1[vmask]
    depth_v = depth_x[vmask].astype(np.float64, copy=False)

    xyz0_cam = backproject_xforward(
        mk0v[:, 0].astype(np.float64),
        mk0v[:, 1].astype(np.float64),
        depth_v,
        fx0, fy0, cx0, cy0
    )

    Tcw0 = _invert_T(anchor_Twc_xf)
    xyz_w = _transform_points(Tcw0, xyz0_cam)

    if T_db2q is not None:
        xyz_w = _transform_points(T_db2q, xyz_w)

    # GT のカメラ座標（x-forward）
    R = Twc_gt_xf[:3, :3].astype(np.float64, copy=False)
    t = Twc_gt_xf[:3, 3].astype(np.float64, copy=False).reshape(3, 1)
    xyz_cam_gt = (R @ xyz_w.T + t).T  # (M,3)

    X = xyz_cam_gt[:, 0]
    proj_ok = np.isfinite(X) & (X > 1e-6) & np.all(np.isfinite(xyz_cam_gt), axis=1)

    u_gt = np.full((xyz_cam_gt.shape[0],), np.nan, dtype=np.float64)
    v_gt = np.full((xyz_cam_gt.shape[0],), np.nan, dtype=np.float64)
    if np.any(proj_ok):
        uu, vv = project_xforward(xyz_cam_gt[proj_ok], fx1, fy1, cx1, cy1)
        u_gt[proj_ok] = uu
        v_gt[proj_ok] = vv

    # 画素誤差を計算する
    e = np.full((xyz_cam_gt.shape[0],), np.inf, dtype=np.float64)
    if np.any(proj_ok):
        dx = u_gt[proj_ok] - mk1v[proj_ok, 0].astype(np.float64)
        dy = v_gt[proj_ok] - mk1v[proj_ok, 1].astype(np.float64)
        e[proj_ok] = np.sqrt(dx * dx + dy * dy)

    # 全配列へ書き戻す
    idx_full = np.where(vmask)[0]
    valid_3d[idx_full] = proj_ok
    err_px[idx_full] = e
    correct[idx_full] = proj_ok & (e <= float(gt_reproj_th_px))

    return correct, err_px, valid_3d


# ----------------------------
# メイン処理
# ----------------------------

def parse_args() -> argparse.Namespace:
    """CLI 引数を定義して解析結果を返す。"""
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, required=True, help="manifest_cross_eval_gt_v3.csv (or filtered version)")
    p.add_argument("--ckpt", type=str, required=True, help="LoFTR checkpoint .pth")
    p.add_argument("--out-dir", type=str, required=True, help="output directory")

    p.add_argument("--resize-long", type=int, default=840)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--cell-point", type=str, default="center", choices=["topleft", "center"])

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp-dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])

    p.add_argument("--match-source", type=str, default="auto", choices=["auto", "fine", "coarse"])
    p.add_argument("--model-enable-fine", type=str, default="auto", choices=["auto","true","false"],
                   help="Instantiate matcher with fine enabled/disabled. auto: enable unless --match-source=coarse.")
    p.add_argument("--match-conf-th", type=float, default=0.0)
    p.add_argument("--max-matches", type=int, default=4096)

    p.add_argument("--min-depth", type=float, default=0.5)
    p.add_argument("--max-depth", type=float, default=200.0)

    p.add_argument("--pnp-reproj-th", type=float, default=4.0)
    p.add_argument("--pnp-max-iter", type=int, default=2000)
    p.add_argument("--pnp-confidence", type=float, default=0.999)
    p.add_argument("--pnp-refine", action="store_true")

    p.add_argument("--gt-type", type=str, default="strict", choices=["all", "strict", "t1m", "t2m", "t5m", "t10m"],
                   help="Filter manifest rows by GT type flag columns (is_strict_gt, is_gt_t1m, ...).")
    p.add_argument("--q-segments-file", type=str, default="", help="Optional file listing q segments to include (one per line).")

    p.add_argument("--assume-same-frame", action="store_true",
                   help="Compute pose errors even without ICP. If not set, errors are only computed when frames are known comparable (db_seg==q_seg or ICP available).")

    # ICP 処理
    p.add_argument("--icp-csv", type=str, default="", help="Optional db->q ICP transform CSV (for comparing to GT in query frame).")
    p.add_argument("--icp-db-col", type=str, default="", help="Override db segment column name in icp csv.")
    p.add_argument("--icp-q-col", type=str, default="", help="Override q segment column name in icp csv.")
    p.add_argument("--icp-mat-col", type=str, default="", help="Override 4x4 matrix column name in icp csv (string/list of 16).")
    p.add_argument("--icp-mat-cols", type=str, default="", help="Override 16 matrix columns, comma-separated (T00,T01,...,T33).")

    p.add_argument("--skip-error-rows", action="store_true")
    p.add_argument("--no-verify-files", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Process only first N rows (0=all).")

    # 同じ per-pair 結果から、複数候補半径にまたがる事後集約を行う
    p.add_argument("--summary-radii", type=str, default="1,2,5,10",
                   help="Comma-separated radii [m] for per-query selection summary (e.g., \"1,2,5,10\").")
    p.add_argument("--dist-bins", type=str, default="0,1,2,3,4,5,6,10",
                   help="Distance bins [m] for per-pair summary (e.g., \"0,1,2,3,4,5,6,10\").")
    p.add_argument("--z-align", type=str, default="segpair_median_nearest",
                   choices=["none", "segpair_median_nearest"],
                   help="If enabled, compute per-(db_seg,q_seg) median dz using nearest-anchor pairs and report z-aligned 3D errors.")
    p.add_argument("--no-write-per-radius", action="store_true",
                   help="If set, do NOT write per_query_strict.csv and per_query_t{r}m.csv files (summary.json still contains stats).")
    p.add_argument("--write-oracle", action="store_true",
                   help="If set, also output oracle per-radius results (min translation error among candidates) for analysis.")

    # ---- 追加: match 可視化の保存 ----
    p.add_argument("--save-match-vis", action="store_true",
                   help="If set, save match visualization image for each pair (GT-based green/red).")
    p.add_argument("--match-vis-dir", type=str, default="match_vis",
                   help="Subdirectory under --out-dir to save match visualization images.")
    p.add_argument("--match-vis-gt-th", type=float, default=4.0,
                   help="Pixel threshold for GT reprojection to classify match correctness (<=th => green).")
    p.add_argument("--match-vis-max-draw", type=int, default=0,
                   help="Max number of match lines to draw per pair (0=draw all). Useful to reduce clutter/time.")
    p.add_argument("--match-vis-line-thickness", type=int, default=1,
                   help="Line thickness for match visualization (px).")
    p.add_argument("--match-vis-point-radius", type=int, default=2,
                   help="Endpoint point radius for match visualization (px).")
    p.add_argument("--match-vis-no-text", action="store_true",
                   help="If set, do not draw overlay text on match visualization images.")
    p.add_argument("--match-vis-save-vertical", action="store_true",
                   help="If set, also save vertical layout visualization with '_v.png' suffix.")
    p.add_argument("--match-vis-only-pnp-fail", action="store_true",
                   help="If set, save visualization only when PnP failed for this pair.")
    p.add_argument("--match-vis-only-gt-bad", action="store_true",
                   help="If set, save visualization only when comparable and t_err_xy > --match-vis-gt-bad-th.")
    p.add_argument("--match-vis-gt-bad-th", type=float, default=2.0,
                   help="Threshold [m] for --match-vis-only-gt-bad.")
    p.add_argument("--match-vis-dist-bins", type=str, default="0,0.5,1,2,5,10",
                   help="Comma-separated distance bins (meters) for match vis subdirs. Empty disables binning.")

    return p.parse_args()


def _load_json_cached(path: str, cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """JSON をキャッシュ付きで読み込む。"""
    if path in cache:
        return cache[path]
    with open(path, "r") as f:
        obj = json.load(f)
    cache[path] = obj
    return obj


def _load_npz_cached(path: str, cache: Dict[str, AnchorIndex]) -> AnchorIndex:
    """NPZ をキャッシュ付きで読み込む。"""
    if path in cache:
        return cache[path]
    idx = load_anchor_index_npz(path)
    cache[path] = idx
    return idx


def _read_segments_file(path: str) -> List[str]:
    """セグメント一覧ファイルを読み込む。"""
    out = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
    return out


def _filter_manifest(df: pd.DataFrame, gt_type: str, q_segments: Optional[set]) -> pd.DataFrame:
    """評価条件に合わせて manifest を絞り込む。"""
    if q_segments is not None:
        df = df[df["q_seg"].astype(str).isin(q_segments)].copy()

    if gt_type == "all":
        return df

    col = {
        "strict": "is_strict_gt",
        "t1m": "is_gt_t1m",
        "t2m": "is_gt_t2m",
        "t5m": "is_gt_t5m",
        "t10m": "is_gt_t10m",
    }[gt_type]

    if col not in df.columns:
        raise ValueError(f"Manifest has no column '{col}'. columns={list(df.columns)}")

    df = df[df[col].astype(bool)].copy()
    return df


def _select_best_per_query(per_pair_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pick best candidate row per query (q_seg, q_frame_index) based on:
      1) pnp_success (True preferred)
      2) n_inlier (desc)
      3) reproj_med (asc)
      4) n_3d (desc)
    """
    if per_pair_df.empty:
        return per_pair_df.copy()

    d = per_pair_df.copy()
    d["pnp_success_int"] = d["pnp_success"].astype(int)
    for c in ["n_inlier", "reproj_med", "n_3d"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["reproj_med"] = d["reproj_med"].fillna(np.inf)

    sort_cols = ["pnp_success_int", "n_inlier", "reproj_med", "n_3d"]
    asc = [False, False, True, False]
    d = d.sort_values(sort_cols, ascending=asc)

    gcols = ["q_seg", "q_frame_index"]
    best = d.groupby(gcols, as_index=False).head(1).copy()
    best = best.drop(columns=["pnp_success_int"], errors="ignore")
    return best


def _select_nearest_per_query(per_pair_df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the nearest-anchor row per query using.
    """
    if per_pair_df.empty:
        return per_pair_df.copy()
    d = per_pair_df.copy()
    d["dist_xy_m"] = pd.to_numeric(d["dist_xy_m"], errors="coerce")
    d = d.sort_values(["q_seg", "q_frame_index", "dist_xy_m", "anchor_id"], ascending=[True, True, True, True])
    nearest = d.groupby(["q_seg", "q_frame_index"], as_index=False).head(1).copy()
    return nearest


def _parse_radii_arg(s: str) -> List[float]:
    """
    Parse comma-separated radii string into a sorted list of unique floats.
    """
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    radii: List[float] = []
    for p in parts:
        try:
            radii.append(float(p))
        except Exception:
            raise ValueError(f"Invalid --summary-radii item: {p}")
    radii = sorted(list({float(r) for r in radii if r > 0}))
    return radii


def _compute_zalign_offsets_from_nearest(per_pair_df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """
    Compute dz offsets per (db_seg, q_seg) using nearest-anchor pairs:
        dz = pred_Cz - gt_Cz
    We use the median across queries within each segment pair.
    """
    if per_pair_df.empty:
        return {}

    nearest = _select_nearest_per_query(per_pair_df)
    d = nearest.copy()
    d = d[d["pnp_success"].astype(bool)]
    d["pred_Cz"] = pd.to_numeric(d["pred_Cz"], errors="coerce")
    d["gt_Cz"] = pd.to_numeric(d["gt_Cz"], errors="coerce")
    d = d[np.isfinite(d["pred_Cz"].to_numpy()) & np.isfinite(d["gt_Cz"].to_numpy())].copy()
    if d.empty:
        return {}

    d["dz"] = d["pred_Cz"] - d["gt_Cz"]
    dz_med = d.groupby(["db_seg", "q_seg"])["dz"].median()
    return {(str(k[0]), str(k[1])): float(v) for k, v in dz_med.items()}


def _apply_zalign(per_pair_df: pd.DataFrame, mode: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Add z-aligned error columns to per_pair_df.
    mode:
      - "none": no-op (still returns df, info)
      - "segpair_median_nearest": subtract per-(db_seg,q_seg) median dz estimated from nearest-anchor results
    """
    info: Dict[str, Any] = {"mode": mode}
    d = per_pair_df.copy()

    if mode == "none":
        return d, info

    if mode != "segpair_median_nearest":
        raise ValueError(f"Unsupported --z-align mode: {mode}")

    dz_map = _compute_zalign_offsets_from_nearest(d)
    info["n_seg_pairs_with_offset"] = int(len(dz_map))

    if dz_map:
        dz_vals = np.array(list(dz_map.values()), dtype=np.float64)
        info["dz_offset_stats"] = {
            "median": float(np.median(dz_vals)),
            "min": float(np.min(dz_vals)),
            "max": float(np.max(dz_vals)),
            "mean": float(np.mean(dz_vals)),
            "std": float(np.std(dz_vals)),
        }
        info["dz_offset_by_segpair"] = [
            {"db_seg": k[0], "q_seg": k[1], "dz_med": float(v)}
            for k, v in sorted(dz_map.items(), key=lambda kv: (kv[0][0], kv[0][1]))
        ]
    else:
        info["dz_offset_stats"] = {}
        info["dz_offset_by_segpair"] = []

    def _lookup_dz(row: pd.Series) -> float:
        """高さ合わせに使う z 補正量を検索する。"""
        key = (str(row.get("db_seg", "")), str(row.get("q_seg", "")))
        return float(dz_map.get(key, 0.0))

    d["dz_off"] = d.apply(_lookup_dz, axis=1)

    for c in ["pred_Cx", "pred_Cy", "pred_Cz", "gt_Cx", "gt_Cy", "gt_Cz"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    d["pred_Cz_zalign"] = d["pred_Cz"] - d["dz_off"]
    d["t_err_z_zalign"] = (d["pred_Cz_zalign"] - d["gt_Cz"]).abs()
    d["t_err_xyz_zalign"] = np.sqrt(
        (d["pred_Cx"] - d["gt_Cx"]) ** 2 +
        (d["pred_Cy"] - d["gt_Cy"]) ** 2 +
        (d["pred_Cz_zalign"] - d["gt_Cz"]) ** 2
    )
    return d, info


def _select_oracle_min_xy(per_pair_df: pd.DataFrame) -> pd.DataFrame:
    """
    Oracle selection (for analysis): per query, pick the candidate with minimum t_err_xy among PnP-success rows.
    If all candidates failed, the query may be missing in the oracle output.
    """
    if per_pair_df.empty:
        return per_pair_df.copy()
    d = per_pair_df.copy()
    d = d[d["pnp_success"].astype(bool)].copy()
    d["t_err_xy"] = pd.to_numeric(d["t_err_xy"], errors="coerce")
    d = d[np.isfinite(d["t_err_xy"].to_numpy())].copy()
    if d.empty:
        return d
    idx = d.groupby(["q_seg", "q_frame_index"])["t_err_xy"].idxmin()
    idx = idx.dropna().astype(int)
    return d.loc[idx].reset_index(drop=True)


def _summarize_loc(df_query: pd.DataFrame) -> Dict[str, Any]:
    """位置誤差に関する集計指標をまとめる。"""
    if df_query.empty:
        return {"n_query": 0}

    out: Dict[str, Any] = {}
    out["n_query"] = int(df_query.shape[0])
    out["pnp_success_rate"] = float(df_query["pnp_success"].astype(bool).mean())

    for k in ["t_err_xy", "t_err_xyz", "t_err_xyz_zalign", "t_err_z_zalign", "rot_err_deg", "yaw_err_deg"]:
        if k in df_query.columns:
            arr = pd.to_numeric(df_query[k], errors="coerce").to_numpy(dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            out[f"{k}_median"] = float(np.median(arr))
            out[f"{k}_p95"] = float(np.percentile(arr, 95))

    if "t_err_xy" in df_query.columns:
        e = pd.to_numeric(df_query["t_err_xy"], errors="coerce").to_numpy(dtype=np.float64)
        e = e[np.isfinite(e)]
        for th in [0.5, 1.0, 2.0, 5.0, 10.0]:
            out[f"success_xy@{th}m"] = float(np.mean(e <= th)) if e.size > 0 else 0.0

    return out


def _summarize_pair_errors(df: pd.DataFrame) -> Dict[str, Any]:
    """pair 単位の誤差分布を要約する。"""
    out: Dict[str, Any] = {"n_pairs": int(df.shape[0])}
    if df.empty:
        return out

    if "pnp_success" in df.columns:
        out["pnp_success_rate"] = float(df["pnp_success"].astype(bool).mean())
        out["n_pnp_success"] = int(df["pnp_success"].astype(bool).sum())

    if "comparable" in df.columns:
        out["n_comparable"] = int(df["comparable"].astype(bool).sum())

    # 誤差統計には comparable かつ pnp_success のものだけを使う
    if ("pnp_success" in df.columns) and ("comparable" in df.columns):
        df_ok = df[df["pnp_success"].astype(bool) & df["comparable"].astype(bool)].copy()
    elif "pnp_success" in df.columns:
        df_ok = df[df["pnp_success"].astype(bool)].copy()
    else:
        df_ok = df.copy()

    if df_ok.empty:
        return out

    def _stat(arr: np.ndarray) -> Dict[str, float]:
        """数値配列の要約統計を辞書として返す。"""
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {}
        return {
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "mean": float(np.mean(arr)),
        }

    for c in ["t_err_xy", "t_err_xyz", "t_err_z", "t_err_xyz_zalign", "t_err_z_zalign", "rot_err_deg", "yaw_err_deg"]:
        if c in df_ok.columns:
            out[c] = _stat(pd.to_numeric(df_ok[c], errors="coerce").to_numpy(dtype=np.float64))

    # 距離改善量: dist_xy_m -> t_err_xy
    if "dist_xy_m" in df_ok.columns and "t_err_xy" in df_ok.columns:
        dist = pd.to_numeric(df_ok["dist_xy_m"], errors="coerce").to_numpy(dtype=np.float64)
        err = pd.to_numeric(df_ok["t_err_xy"], errors="coerce").to_numpy(dtype=np.float64)
        mask = np.isfinite(dist) & np.isfinite(err) & (dist >= 0)
        if np.any(mask):
            dist = dist[mask]
            err = err[mask]
            delta = dist - err
            ratio = np.full_like(dist, np.nan, dtype=np.float64)
            ratio[dist > 1e-6] = err[dist > 1e-6] / dist[dist > 1e-6]
            out["dist_xy_m"] = _stat(dist)
            out["delta_xy"] = _stat(delta)
            out["improved_rate_xy"] = float(np.mean(delta > 0))
            out["ratio_xy"] = _stat(ratio)

    # 距離改善量: dist_xyz_m -> t_err_xyz
    if "dist_xyz_m" in df_ok.columns and "t_err_xyz" in df_ok.columns:
        dist3 = pd.to_numeric(df_ok["dist_xyz_m"], errors="coerce").to_numpy(dtype=np.float64)
        err3 = pd.to_numeric(df_ok["t_err_xyz"], errors="coerce").to_numpy(dtype=np.float64)
        mask3 = np.isfinite(dist3) & np.isfinite(err3) & (dist3 >= 0)
        if np.any(mask3):
            dist3 = dist3[mask3]
            err3 = err3[mask3]
            delta3 = dist3 - err3
            ratio3 = np.full_like(dist3, np.nan, dtype=np.float64)
            ratio3[dist3 > 1e-6] = err3[dist3 > 1e-6] / dist3[dist3 > 1e-6]
            out["dist_xyz_m"] = _stat(dist3)
            out["delta_xyz"] = _stat(delta3)
            out["improved_rate_xyz"] = float(np.mean(delta3 > 0))
            out["ratio_xyz"] = _stat(ratio3)

    return out


def main() -> None:
    """CLI 引数を解釈し、入力の読み込みから結果保存までの一連の処理を実行する。"""
    args = parse_args()
    t0 = time.time()

    out_dir = Path(args.out_dir)
    _ensure_dir(str(out_dir))

    df = pd.read_csv(args.manifest)
    if args.limit and int(args.limit) > 0:
        df = df.head(int(args.limit)).copy()

    q_segments = None
    if args.q_segments_file:
        q_segments = set(_read_segments_file(args.q_segments_file))

    df = _filter_manifest(df, args.gt_type, q_segments)

    if not args.no_verify_files:
        for col in ["q_image_png", "q_meta_json", "anchor_png", "anchor_json", "anchor_index_npz"]:
            if col in df.columns:
                missing = df[~df[col].astype(str).apply(lambda p: Path(p).exists())]
                if not missing.empty:
                    print(f"[WARN] Missing files for column {col}: {missing.shape[0]} rows. First: {missing.iloc[0][col]}")
                    if not args.skip_error_rows:
                        raise FileNotFoundError(f"Missing files detected in manifest column {col}. Use --skip-error-rows or --no-verify-files.")
            else:
                print(f"[WARN] manifest has no column '{col}'")

    # ICP 処理 の読み込み（任意）
    icp: Optional[IcpDBtoQ] = None
    if args.icp_csv:
        try:
            mat_cols = [c.strip() for c in args.icp_mat_cols.split(",") if c.strip()] if args.icp_mat_cols else None
            if mat_cols is not None and len(mat_cols) != 16:
                raise ValueError("--icp-mat-cols must have exactly 16 comma-separated columns.")
            icp = load_icp_db_to_q(
                args.icp_csv,
                db_col=(args.icp_db_col or None),
                q_col=(args.icp_q_col or None),
                mat_col=(args.icp_mat_col or None),
                mat_cols=mat_cols,
            )
            print(f"[INFO] Loaded ICP transforms: {len(icp.table)} pairs")
        except Exception as e:
            print(f"[WARN] Failed to load/parse ICP csv: {e}")
            icp = None

    # device と AMP の設定
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    amp_enabled = (device.type == "cuda") and (args.amp_dtype.lower() != "fp32")
    amp_dtype = torch.bfloat16
    if args.amp_dtype.lower() == "fp16":
        amp_dtype = torch.float16
    elif args.amp_dtype.lower() == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16  # unused

    # fine を有効にして matcher を組むか決める
    if str(args.model_enable_fine).lower() == "true":
        enable_fine = True
    elif str(args.model_enable_fine).lower() == "false":
        enable_fine = False
    else:
        enable_fine = (str(args.match_source) != "coarse")

    matcher = load_matcher(args.ckpt, device=device, enable_fine=enable_fine, enable_repeatability=False)

    json_cache: Dict[str, Dict[str, Any]] = {}
    idx_cache: Dict[str, AnchorIndex] = {}

    per_rows: List[Dict[str, Any]] = []

    # 可視化出力ディレクトリ（任意）
    vis_dir = out_dir / str(args.match_vis_dir) / str(args.gt_type)
    if args.save_match_vis:
        _ensure_dir(str(vis_dir))
    dist_bins = _parse_dist_bins(args.match_vis_dist_bins)
    dist_bins_summary = _parse_dist_bins(args.dist_bins)

    pbar = tqdm(df.itertuples(index=False), total=int(df.shape[0]), ncols=120)
    for row in pbar:
        try:
            rdict = row._asdict()
        except Exception:
            rdict = dict(zip(df.columns, row))

        pair_id = rdict.get("pair_id", "")
        q_seg = str(rdict.get("q_seg", ""))
        db_seg = str(rdict.get("db_seg", ""))
        q_frame_index = int(rdict.get("q_frame_index", -1))
        dist_xy_m = float(rdict.get("dist_xy_m", float("nan")))

        q_img_path = str(rdict["q_image_png"])
        a_img_path = str(rdict["anchor_png"])
        q_meta_path = str(rdict["q_meta_json"])
        a_json_path = str(rdict["anchor_json"])
        a_idx_path = str(rdict["anchor_index_npz"])

        try:
            img0 = _read_gray_u8(a_img_path)  # anchor
            img1 = _read_gray_u8(q_img_path)  # query

            q_meta = _load_json_cached(q_meta_path, json_cache)
            a_meta = _load_json_cached(a_json_path, json_cache)

            K0 = _parse_intrinsic(a_meta)
            K1 = _parse_intrinsic(q_meta)
            Twc0_xf = _parse_Twc(a_meta)
            Twc_gt_xf = _parse_Twc(q_meta)

            anchor_index = _load_npz_cached(a_idx_path, idx_cache)

            matches = infer_matches(
                matcher=matcher,
                img0_u8=img0,
                img1_u8=img1,
                resize_long=int(args.resize_long),
                stride=int(args.stride),
                cell_point=str(args.cell_point),
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                match_source=str(args.match_source),
                conf_th=float(args.match_conf_th),
                max_matches=int(args.max_matches),
            )

            mk0 = matches["mkpts0"]
            mk1 = matches["mkpts1"]

            # db->q 変換（任意）
            T_db2q = None
            if icp is not None:
                T_db2q = icp.get(db_seg, q_seg)

            pose = estimate_pose_pnp_xforward(
                mkpts0=mk0,
                mkpts1=mk1,
                anchor_index=anchor_index,
                anchor_Twc_xf=Twc0_xf,
                anchor_K=K0,
                query_K=K1,
                T_db2q=T_db2q,
                min_depth=float(args.min_depth),
                max_depth=float(args.max_depth),
                ransac_reproj_th=float(args.pnp_reproj_th),
                ransac_max_iter=int(args.pnp_max_iter),
                ransac_conf=float(args.pnp_confidence),
                refine=bool(args.pnp_refine),
            )

            # pose 誤差が意味を持つのは、world frame 同士が比較可能な場合だけである。
            if args.icp_csv:
                comparable = (T_db2q is not None)
            else:
                comparable = bool(args.assume_same_frame or (db_seg == q_seg))

            out: Dict[str, Any] = {
                "pair_id": pair_id,
                "q_seg": q_seg,
                "db_seg": db_seg,
                "q_frame_index": q_frame_index,
                "anchor_id": rdict.get("anchor_id", ""),
                "dist_xy_m": dist_xy_m,
                "gt_type": args.gt_type,
                "match_source": args.match_source,
                "n_match_in": int(pose.n_match_in),
                "n_3d": int(pose.n_3d),
                "n_inlier": int(pose.n_inlier),
                "inlier_ratio": float(pose.inlier_ratio),
                "reproj_rmse": float(pose.reproj_rmse),
                "reproj_med": float(pose.reproj_med),
                "reproj_p95": float(pose.reproj_p95),
                "pnp_success": bool(pose.success),
                "comparable": bool(comparable),
                "n_green": float("nan"),
                "n_valid3d": float("nan"),
                "n_match_total": float("nan"),
                "green_ratio": float("nan"),
            }
            # manifest に strict フラグがあれば引き継ぐ
            if "is_strict_gt" in rdict:
                try:
                    out["is_strict_gt"] = bool(int(rdict.get("is_strict_gt")))
                except Exception:
                    out["is_strict_gt"] = bool(rdict.get("is_strict_gt"))
            # 任意の pair 単位 distance bin ラベル
            if dist_bins_summary:
                out["dist_bin"] = _dist_bin_label(dist_xy_m, dist_bins_summary)

            if pose.success and pose.Twc_pred_xf is not None:
                Twc_pred_xf = pose.Twc_pred_xf
                C_pred = pose.C_pred if pose.C_pred is not None else _camera_center_world(Twc_pred_xf)
                C_gt = _camera_center_world(Twc_gt_xf)
                # db frame における anchor 中心（anchor pose 由来）
                C_anchor_db = _camera_center_world(Twc0_xf)
                if comparable:
                    if T_db2q is not None:
                        C_anchor_q = _transform_points(T_db2q, C_anchor_db.reshape(1, 3))[0]
                    else:
                        C_anchor_q = C_anchor_db
                    out["dist_xyz_m"] = float(np.linalg.norm((C_anchor_q - C_gt).astype(np.float64)))
                else:
                    out["dist_xyz_m"] = float("nan")

                out.update({
                    "pred_Cx": float(C_pred[0]),
                    "pred_Cy": float(C_pred[1]),
                    "pred_Cz": float(C_pred[2]),
                    "gt_Cx": float(C_gt[0]),
                    "gt_Cy": float(C_gt[1]),
                    "gt_Cz": float(C_gt[2]),
                })

                if comparable:
                    d = (C_pred - C_gt).astype(np.float64)
                    out["t_err_xyz"] = float(np.linalg.norm(d))
                    out["t_err_xy"] = float(np.linalg.norm(d[:2]))
                    out["t_err_z"] = float(abs(d[2]))
                    if np.isfinite(out.get("dist_xyz_m", np.nan)):
                        out["delta_xyz"] = float(out["dist_xyz_m"] - out["t_err_xyz"])

                    out["rot_err_deg"] = _rot_angle_deg(Twc_pred_xf[:3, :3], Twc_gt_xf[:3, :3])

                    fwd_p = _forward_axis_world_from_Twc_xforward(Twc_pred_xf)
                    fwd_g = _forward_axis_world_from_Twc_xforward(Twc_gt_xf)
                    yaw_p = math.degrees(math.atan2(float(fwd_p[1]), float(fwd_p[0])))
                    yaw_g = math.degrees(math.atan2(float(fwd_g[1]), float(fwd_g[0])))
                    out["yaw_pred_deg"] = float(yaw_p)
                    out["yaw_gt_deg"] = float(yaw_g)
                    out["yaw_err_deg"] = _wrap_deg(yaw_p - yaw_g)
                else:
                    out["t_err_xyz"] = float("nan")
                    out["t_err_xy"] = float("nan")
                    out["t_err_z"] = float("nan")
                    out["delta_xyz"] = float("nan")
                    out["rot_err_deg"] = float("nan")
                    out["yaw_pred_deg"] = float("nan")
                    out["yaw_gt_deg"] = float("nan")
                    out["yaw_err_deg"] = float("nan")
            else:
                out.update({
                    "pred_Cx": float("nan"), "pred_Cy": float("nan"), "pred_Cz": float("nan"),
                    "gt_Cx": float("nan"), "gt_Cy": float("nan"), "gt_Cz": float("nan"),
                    "t_err_xyz": float("nan"), "t_err_xy": float("nan"), "t_err_z": float("nan"),
                    "dist_xyz_m": float("nan"), "delta_xyz": float("nan"),
                    "rot_err_deg": float("nan"),
                    "yaw_pred_deg": float("nan"), "yaw_gt_deg": float("nan"), "yaw_err_deg": float("nan"),
                })

            # ---- 追加: match 可視化を保存する（GT ベースで緑 / 赤） ----
            if args.save_match_vis:
                save_this = True

                if args.match_vis_only_pnp_fail and bool(pose.success):
                    save_this = False

                if args.match_vis_only_gt_bad:
                    if not comparable:
                        save_this = False
                    else:
                        try:
                            te = float(out.get("t_err_xy", float("nan")))
                            if (not np.isfinite(te)) or (te <= float(args.match_vis_gt_bad_th)):
                                save_this = False
                        except Exception:
                            save_this = False

                if save_this:
                    dist_label = _dist_bin_label(dist_xy_m, dist_bins)
                    vis_dir_bin = vis_dir / dist_label
                    if args.save_match_vis:
                        _ensure_dir(str(vis_dir_bin))
                    # match を分類する（GT 再投影ベース）
                    # 注意: comparable が False でも分類自体は db frame で計算されるが、意味を持たない可能性がある。
                    #       そのため分類をスキップして全件赤扱いにする。
                    if comparable:
                        correct_mask, err_px, valid3d = classify_matches_by_gt_reproj(
                            mkpts0=mk0,
                            mkpts1=mk1,
                            anchor_index=anchor_index,
                            anchor_Twc_xf=Twc0_xf,
                            anchor_K=K0,
                            query_K=K1,
                            Twc_gt_xf=Twc_gt_xf,
                            T_db2q=T_db2q,
                            min_depth=float(args.min_depth),
                            max_depth=float(args.max_depth),
                            gt_reproj_th_px=float(args.match_vis_gt_th),
                        )
                    else:
                        correct_mask = np.zeros((mk0.shape[0],), dtype=bool)
                        err_px = np.full((mk0.shape[0],), np.inf, dtype=np.float64)
                        valid3d = np.zeros((mk0.shape[0],), dtype=bool)

                    n_all = int(mk0.shape[0])
                    n_green = int(np.sum(correct_mask))
                    n_valid = int(np.sum(valid3d))
                    out["n_green"] = int(n_green)
                    out["n_valid3d"] = int(n_valid)
                    out["n_match_total"] = int(n_all)
                    out["green_ratio"] = float(n_green / n_all) if n_all > 0 else float("nan")

                    title_lines = [
                        f"pair_id={str(pair_id)}",
                        f"q_frame={q_frame_index} dist_xy_m={dist_xy_m:.3f} gt_type={args.gt_type}",
                        f"pnp_success={bool(pose.success)} inlier={int(pose.n_inlier)}/{int(pose.n_3d)} matches={int(pose.n_match_in)}",
                        f"comparable={bool(comparable)}  gt_th={float(args.match_vis_gt_th):.1f}px  green={n_green}/{n_all}  valid3d={n_valid}/{n_all}",
                    ]
                    if comparable and np.isfinite(out.get("t_err_xy", np.nan)):
                        title_lines.append(
                            f"t_err_xy={float(out.get('t_err_xy', np.nan)):.3f}m  rot_err={float(out.get('rot_err_deg', np.nan)):.3f}deg"
                        )
                    else:
                        title_lines.append("t_err_xy/rot_err: n/a")

                    # ファイル名
                    base_key = f"{q_seg}|{q_frame_index}|{db_seg}|{rdict.get('anchor_id','')}|{pair_id}|{dist_xy_m}"
                    h = hashlib.sha1(base_key.encode("utf-8")).hexdigest()[:10]
                    fname = f"q{q_frame_index:06d}_d{dist_xy_m:.2f}_{h}.png"
                    out_path_h = str((vis_dir_bin / fname).resolve())

                    _visualize_matches_gt(
                        img0_gray=img0,
                        img1_gray=img1,
                        mkpts0=mk0,
                        mkpts1=mk1,
                        correct_mask=correct_mask,
                        out_path=out_path_h,
                        pair_title_lines=title_lines,
                        max_draw=int(args.match_vis_max_draw),
                        seed_key=base_key,
                        line_thickness=int(args.match_vis_line_thickness),
                        point_radius=int(args.match_vis_point_radius),
                        draw_text=(not bool(args.match_vis_no_text)),
                        layout="horizontal",
                    )
                    if bool(args.match_vis_save_vertical):
                        p_h = Path(out_path_h)
                        out_path_v = str((p_h.parent / f"{p_h.stem}_v.png").resolve())
                        _visualize_matches_gt(
                            img0_gray=img0,
                            img1_gray=img1,
                            mkpts0=mk0,
                            mkpts1=mk1,
                            correct_mask=correct_mask,
                            out_path=out_path_v,
                            pair_title_lines=title_lines,
                            max_draw=int(args.match_vis_max_draw),
                            seed_key=base_key,
                            line_thickness=int(args.match_vis_line_thickness),
                            point_radius=int(args.match_vis_point_radius),
                            draw_text=(not bool(args.match_vis_no_text)),
                            layout="vertical",
                        )

            per_rows.append(out)
            pbar.set_postfix(pnp=str(out["pnp_success"]), inl=int(out["n_inlier"]), xy=f"{out.get('t_err_xy', float('nan')):.2f}")

        except Exception as e:
            if not args.skip_error_rows:
                raise
            per_rows.append({
                "pair_id": pair_id,
                "q_seg": q_seg,
                "db_seg": db_seg,
                "q_frame_index": q_frame_index,
                "anchor_id": rdict.get("anchor_id", ""),
                "dist_xy_m": dist_xy_m,
                "gt_type": args.gt_type,
                "match_source": args.match_source,
                "pnp_success": False,
                "error": str(e),
            })

    per_pair_df = pd.DataFrame(per_rows)

    # --- 任意の z alignment（cross-segment 評価で t_err_xyz を意味ある値にするため） ---
    z_align_mode = str(args.z_align).strip().lower()
    if z_align_mode not in ["none", "segpair_median_nearest"]:
        raise ValueError(f"Invalid --z-align: {args.z_align}")
    per_pair_df, z_align_info = _apply_zalign(per_pair_df, mode=z_align_mode)

    # --- strict フラグ / distance bin ごとの pair 単位サマリ ---
    pair_summary_by_dist: Dict[str, Any] = {}
    if "is_strict_gt" in per_pair_df.columns:
        strict_df = per_pair_df[per_pair_df["is_strict_gt"].astype(bool)].copy()
        pair_summary_by_dist["strict"] = _summarize_pair_errors(strict_df)

    dist_bins_summary = _parse_dist_bins(args.dist_bins)
    if dist_bins_summary and "dist_xy_m" in per_pair_df.columns:
        dist_vals = pd.to_numeric(per_pair_df["dist_xy_m"], errors="coerce").to_numpy(dtype=np.float64)
        for i in range(len(dist_bins_summary) - 1):
            lo = dist_bins_summary[i]
            hi = dist_bins_summary[i + 1]
            label = f"dist_{_fmt_bin(lo)}_{_fmt_bin(hi)}"
            mask = (dist_vals >= lo) & (dist_vals < hi)
            pair_summary_by_dist[label] = _summarize_pair_errors(per_pair_df[mask])
        # 最終ビン以上
        last = dist_bins_summary[-1]
        mask = (dist_vals >= last)
        pair_summary_by_dist[f"dist_ge_{_fmt_bin(last)}"] = _summarize_pair_errors(per_pair_df[mask])

    per_pair_csv = out_dir / "per_pair.csv"
    per_pair_df.to_csv(per_pair_csv, index=False)

    per_query_df = _select_best_per_query(per_pair_df)
    per_query_csv = out_dir / "per_query.csv"
    per_query_df.to_csv(per_query_csv, index=False)

    radii = _parse_radii_arg(args.summary_radii)
    per_query_by_setting: Dict[str, pd.DataFrame] = {}
    per_query_by_setting["strict"] = _select_nearest_per_query(per_pair_df)

    cand_stats: Dict[str, Any] = {"strict": {"avg_candidates": 1.0, "median_candidates": 1.0, "added_vs_prev_avg": 0.0}}
    prev_avg = 1.0

    for r in radii:
        key = f"t{int(r)}m" if float(r).is_integer() else f"t{r}m"
        sub = per_pair_df[pd.to_numeric(per_pair_df["dist_xy_m"], errors="coerce") <= float(r)].copy()
        per_query_by_setting[key] = _select_best_per_query(sub)

        counts = sub.groupby(["q_seg", "q_frame_index"]).size()
        avg_c = float(counts.mean()) if len(counts) > 0 else 0.0
        med_c = float(counts.median()) if len(counts) > 0 else 0.0
        added_avg = float(avg_c - prev_avg)
        prev_avg = avg_c

        cand_stats[key] = {
            "avg_candidates": avg_c,
            "median_candidates": med_c,
            "added_vs_prev_avg": added_avg,
        }

    query_summary_by_setting: Dict[str, Any] = {}
    for k, dfq in per_query_by_setting.items():
        query_summary_by_setting[k] = _summarize_loc(dfq)

    oracle_summary_by_setting: Dict[str, Any] = {}
    if args.write_oracle:
        for r in radii:
            key = f"oracle_t{int(r)}m" if float(r).is_integer() else f"oracle_t{r}m"
            sub = per_pair_df[pd.to_numeric(per_pair_df["dist_xy_m"], errors="coerce") <= float(r)].copy()
            oracle_df = _select_oracle_min_xy(sub)
            oracle_summary_by_setting[key] = _summarize_loc(oracle_df)
            if not args.no_write_per_radius:
                oracle_df.to_csv(out_dir / f"per_query_{key}.csv", index=False)

    if not args.no_write_per_radius:
        per_query_by_setting["strict"].to_csv(out_dir / "per_query_strict.csv", index=False)
        for r in radii:
            key = f"t{int(r)}m" if float(r).is_integer() else f"t{r}m"
            per_query_by_setting[key].to_csv(out_dir / f"per_query_{key}.csv", index=False)

        if z_align_mode != "none":
            dz_rows = []
            for row in z_align_info.get("dz_offset_by_segpair", []):
                dz_rows.append(row)
            if dz_rows:
                pd.DataFrame(dz_rows).to_csv(out_dir / "z_align_dz_offsets.csv", index=False)

    summary = {
        "args": vars(args),
        "n_pairs": int(per_pair_df.shape[0]),
        "pair_pnp_success_rate": float(per_pair_df["pnp_success"].astype(bool).mean()) if "pnp_success" in per_pair_df.columns else 0.0,
        "pair_summary_by_dist": pair_summary_by_dist,
        "dist_bins": _parse_dist_bins(args.dist_bins),
        "query_summary": _summarize_loc(per_query_df),
        "query_summary_by_setting": query_summary_by_setting,
        "oracle_summary_by_setting": oracle_summary_by_setting,
        "candidate_stats_by_setting": cand_stats,
        "z_align": z_align_info,
        "elapsed_sec": float(time.time() - t0),
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=False)

    print(f"[DONE] per_pair:  {per_pair_csv}")
    print(f"[DONE] per_query: {per_query_csv}")
    if not args.no_write_per_radius:
        print(f"[DONE] per_query_strict: {out_dir / 'per_query_strict.csv'}")
        for r in _parse_radii_arg(args.summary_radii):
            k = f"t{int(r)}m" if float(r).is_integer() else f"t{r}m"
            print(f"[DONE] per_query_{k}: {out_dir / f'per_query_{k}.csv'}")
        if args.write_oracle:
            for r in _parse_radii_arg(args.summary_radii):
                k = f"oracle_t{int(r)}m" if float(r).is_integer() else f"oracle_t{r}m"
                print(f"[DONE] per_query_{k}: {out_dir / f'per_query_{k}.csv'}")
        if str(args.z_align).strip().lower() != "none":
            zcsv = out_dir / "z_align_dz_offsets.csv"
            if zcsv.exists():
                print(f"[DONE] z_align offsets: {zcsv}")
    if args.save_match_vis:
        print(f"[DONE] match_vis_dir: {vis_dir}")
    print(f"[DONE] summary:   {summary_path}")


if __name__ == "__main__":
    main()
