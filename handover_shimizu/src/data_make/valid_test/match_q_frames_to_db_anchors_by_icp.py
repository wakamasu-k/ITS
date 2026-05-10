#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 使い方:
#   python match_q_frames_to_db_anchors_by_icp.py --help
#   必要な入力パスと出力先を引数で指定して実行する。

"""
ICP 変換（DB -> Q）を使って、Q カメラフレームを DB 仮想アンカーへ対応付ける。

方針:
- 各 Q フレームについて、ICP の DB->Q 変換を適用したあとで最も近い DB アンカーを探す。
- distance_xy <= max_match_dist_m の場合だけ採用する。
- 複数の Q フレームが同じアンカーに対応してもよい。
- 近いアンカーがない Q フレームは捨てる。

入力:
- pairs_txt（triplet 形式）: lane_tag, db_segment_name, q_segment_name
- icp_csv: align_startend_staticmaps_icp.py の出力（final_yaw_deg, final_tx, final_ty を含む必要あり）
- db_submaps_root: DB セグメントごとの anchors.csv を含むルート
- q_cam_root: 書き出し済み Q 画像とフレームごとの json を含むルート

出力:
- out_csv: 対応付けられた行
- （任意）out_dropped_csv: 捨てられた行

前提:
- CSV 内の ICP 変換は DB->Q である（これまでの確認内容どおり）。
- 距離は XY 平面上で計算する。

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PAIR_LINE_RE = re.compile(r"^(segment-[0-9]+_.+?_with_camera_labels)$")


def _normalize_wsl_path(p: str) -> str:
    """WSL と Windows が混在するパス表記を読める形に正規化する。"""
    p = (p or "").strip()
    if not p:
        return p
    # Windows ドライブパスを WSL マウント形式へ変換する
    if len(p) >= 3 and p[1] == ":" and (p[2] == "\\" or p[2] == "/"):
        drive = p[0].lower()
        rest = p[2:].lstrip("\\/").replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return p


def _as_bool(s: str) -> bool:
    """CLI などで受け取った真偽値表現を bool に正規化する。"""
    t = (s or "").strip().lower()
    return t in ("1", "true", "yes", "y", "on")


def parse_pairs_triplets(pairs_path: Path) -> List[Tuple[str, str, str]]:
    """セグメント対や三つ組の一覧テキストを解釈する。"""
    lines = [ln.strip() for ln in pairs_path.read_text().splitlines() if ln.strip()]
    if len(lines) % 3 != 0:
        raise ValueError(f"pairs file must be multiples of 3 lines: {pairs_path} (got {len(lines)})")
    pairs = []
    for i in range(0, len(lines), 3):
        lane_tag = lines[i]
        db_seg = lines[i + 1]
        q_seg = lines[i + 2]
        # 厳密にはせず、ここでは落とさない
        if not PAIR_LINE_RE.match(db_seg):
            pass
        if not PAIR_LINE_RE.match(q_seg):
            pass
        pairs.append((lane_tag, db_seg, q_seg))
    return pairs


def short_id6(seg: str) -> str:
    """長い ID から比較や表示に使う短縮 6 文字 ID を作る。"""
    m = re.match(r"segment-([0-9]{6})", seg)
    return m.group(1) if m else seg[:6]


def pair_id_from_segments(db_seg: str, q_seg: str) -> str:
    """query と database のセグメント名から安定したペア ID を作る。"""
    return f"segpair_{short_id6(db_seg)}__{short_id6(q_seg)}"


def find_segment_dir(root: Path, segment: str) -> Tuple[str, Path]:
    """
    Find segment directory under root/<subset>/<segment>.
    Returns (subset, path).
    """
    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")
    # まず一般的な subset を優先する
    for subset in ["training", "validation", "testing"]:
        p = root / subset / segment
        if p.exists():
            return subset, p
    # そのほかの subset ディレクトリも探す
    for psub in root.iterdir():
        if not psub.is_dir():
            continue
        p = psub / segment
        if p.exists():
            return psub.name, p
    raise FileNotFoundError(f"segment dir not found: root={root} segment={segment}")


def load_icp_index(icp_csv: Path) -> Dict[Tuple[str, str], dict]:
    """
    Index ICP records by (db_seg, q_seg). Fallback by (pair_id) if needed.
    Expected columns (at least):
      - db_seg, q_seg OR pair_id
      - final_yaw_deg, final_tx, final_ty
    """
    idx: Dict[Tuple[str, str], dict] = {}
    idx_pairid: Dict[str, dict] = {}

    with icp_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            db_seg = (row.get("db_seg") or row.get("db_segment") or "").strip()
            q_seg = (row.get("q_seg") or row.get("q_segment") or "").strip()
            pid = (row.get("pair_id") or "").strip()

            # まず生の row を保持し、解釈は後で行う
            if db_seg and q_seg:
                idx[(db_seg, q_seg)] = row
            if pid:
                idx_pairid[pid] = row

    # pairid 用 index を隠しフィールドとして付ける
    idx[("__PAIRID__", "__PAIRID__")] = {"_pairid_index": idx_pairid}  # type: ignore
    return idx


def get_icp_transform_xy(
    icp_idx: Dict[Tuple[str, str], dict],
    db_seg: str,
    q_seg: str,
    invert: bool,
) -> Tuple[float, float, float]:
    """
    Return (yaw_rad, tx, ty) for DB->Q in XY.
    If invert=True, return the inverse (Q->DB) converted back to DB->Q style effectively reversed.
    """
    row = icp_idx.get((db_seg, q_seg))
    if row is None:
        pid = pair_id_from_segments(db_seg, q_seg)
        pairid_index = icp_idx.get(("__PAIRID__", "__PAIRID__"), {}).get("_pairid_index", {})
        row = pairid_index.get(pid)
    if row is None:
        raise KeyError(f"ICP row not found for pair: {db_seg} <-> {q_seg}")

    def _f(key: str) -> float:
        """辞書から数値項目を安全に取り出して float へ変換する。"""
        v = row.get(key)
        if v is None or str(v).strip() == "":
            raise KeyError(f"missing ICP column {key}")
        return float(v)

    yaw_deg = _f("final_yaw_deg")
    tx = _f("final_tx")
    ty = _f("final_ty")

    yaw = math.radians(yaw_deg)

    if not invert:
        return yaw, tx, ty

    # SE(2) の逆変換: p' = R p + t
    # 逆変換は p = R^T (p' - t) = R^T p' + (-R^T t)
    c = math.cos(yaw)
    s = math.sin(yaw)
    # R^T は [[c, s], [-s, c]]
    inv_yaw = -yaw
    inv_tx = -(c * tx + s * ty)
    inv_ty = -(-s * tx + c * ty)
    return inv_yaw, inv_tx, inv_ty


def load_db_anchors_xy(anchors_csv: Path) -> Tuple[List[int], List[float], List[float]]:
    """DB 側アンカーの xy 座標を読み込む。"""
    ids: List[int] = []
    xs: List[float] = []
    ys: List[float] = []
    with anchors_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        # これらの列が必要
        for row in reader:
            aid = int(row["anchor_id"])
            x = float(row["x"])
            y = float(row["y"])
            ids.append(aid)
            xs.append(x)
            ys.append(y)
    if not ids:
        raise RuntimeError(f"anchors.csv is empty: {anchors_csv}")
    return ids, xs, ys


def apply_se2_xy(xs: List[float], ys: List[float], yaw: float, tx: float, ty: float) -> Tuple[List[float], List[float]]:
    """2D の SE(2) 変換を xy 座標列へ適用する。"""
    c = math.cos(yaw)
    s = math.sin(yaw)
    outx: List[float] = []
    outy: List[float] = []
    for x, y in zip(xs, ys):
        x2 = c * x - s * y + tx
        y2 = s * x + c * y + ty
        outx.append(x2)
        outy.append(y2)
    return outx, outy


def list_q_frame_meta(q_cam_dir: Path) -> List[Path]:
    # フレームごとの json は 00010.json のような名前
    """query フレームごとのメタ JSON を列挙する。"""
    metas = sorted([p for p in q_cam_dir.glob("*.json") if p.name != "export_meta.json"])
    return metas


def read_q_frame_xy(meta_json: Path) -> Tuple[int, float, float]:
    """query フレームの位置をメタ情報から読み取る。"""
    d = json.loads(meta_json.read_text())
    # ファイル名は frame index を表すが、json があればそちらを優先する
    frame_index = d.get("frame_index")
    if frame_index is None:
        m = re.match(r"^([0-9]+)\.json$", meta_json.name)
        if not m:
            raise ValueError(f"cannot infer frame_index from filename: {meta_json}")
        frame_index = int(m.group(1))
    else:
        frame_index = int(frame_index)

    fp = d.get("frame_pose")
    if fp is None or not isinstance(fp, list) or len(fp) != 4:
        raise ValueError(f"frame_pose missing or invalid: {meta_json}")

    # 4x4 行列で、並進は [0][3], [1][3] に入る
    x = float(fp[0][3])
    y = float(fp[1][3])
    return frame_index, x, y


def main() -> None:
    """CLI 引数を解釈し、入力の読み込みから結果保存までの一連の処理を実行する。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_txt", required=True, help="pairs file (triplets) e.g. /mnt/d/.../pairs_val_2.txt")
    ap.add_argument("--icp_csv", required=True, help="ICP CSV (DB->Q) e.g. /mnt/d/.../val2_icp_startend_staticmaps.csv")
    ap.add_argument("--db_submaps_root", required=True, help="DB submaps root (contains anchors.csv)")
    ap.add_argument("--q_cam_root", required=True, help="Q cam_gray root (exported images + json)")
    ap.add_argument("--cam", default="FRONT", help="camera name (folder under q segment), e.g. FRONT")
    ap.add_argument("--max_match_dist_m", type=float, default=0.5, help="max XY distance (m) to accept match")
    ap.add_argument("--invert_icp", type=str, default="false", help="if true, use inverse transform (debug)")
    ap.add_argument("--out_csv", required=True, help="output CSV for matched rows")
    ap.add_argument("--write_dropped_csv", type=str, default="true", help="write dropped rows csv (true/false)")
    ap.add_argument("--out_dropped_csv", default="", help="output CSV for dropped rows (if enabled)")
    args = ap.parse_args()

    pairs_path = Path(_normalize_wsl_path(args.pairs_txt))
    icp_path = Path(_normalize_wsl_path(args.icp_csv))
    db_submaps_root = Path(_normalize_wsl_path(args.db_submaps_root))
    q_cam_root = Path(_normalize_wsl_path(args.q_cam_root))
    out_csv = Path(_normalize_wsl_path(args.out_csv))
    write_dropped = _as_bool(args.write_dropped_csv)
    out_dropped_csv = Path(_normalize_wsl_path(args.out_dropped_csv)) if args.out_dropped_csv else None
    invert = _as_bool(args.invert_icp)

    if not pairs_path.exists():
        raise FileNotFoundError(pairs_path)
    if not icp_path.exists():
        raise FileNotFoundError(icp_path)
    if not db_submaps_root.exists():
        raise FileNotFoundError(db_submaps_root)
    if not q_cam_root.exists():
        raise FileNotFoundError(q_cam_root)

    pairs = parse_pairs_triplets(pairs_path)
    print(f"[INFO] pairs={len(pairs)} from {pairs_path}")
    print(f"[INFO] icp_csv={icp_path}")
    print(f"[INFO] db_submaps_root={db_submaps_root}")
    print(f"[INFO] q_cam_root={q_cam_root} cam={args.cam}")
    print(f"[INFO] max_match_dist_m={args.max_match_dist_m} invert_icp={invert}")

    icp_idx = load_icp_index(icp_path)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if write_dropped and out_dropped_csv is not None:
        out_dropped_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "pair_id",
        "lane_tag",
        "db_seg",
        "q_seg",
        "db_subset",
        "q_subset",
        "q_frame_index",
        "q_meta_json",
        "q_image_png",
        "q_x",
        "q_y",
        "anchor_id",
        "anchor_x_q",
        "anchor_y_q",
        "dist_xy_m",
        "anchor_png_expected",
    ]

    matched_rows = 0
    dropped_rows = 0

    with out_csv.open("w", newline="", encoding="utf-8") as f_out:
        w = csv.DictWriter(f_out, fieldnames=fields)
        w.writeheader()

        w_drop = None
        f_drop = None
        if write_dropped and out_dropped_csv is not None:
            f_drop = out_dropped_csv.open("w", newline="", encoding="utf-8")
            w_drop = csv.DictWriter(f_drop, fieldnames=fields)
            w_drop.writeheader()

        try:
            for i, (lane_tag, db_seg, q_seg) in enumerate(pairs, 1):
                pid = pair_id_from_segments(db_seg, q_seg)

                # ディレクトリを探す（subset 自動判定）
                db_subset, db_seg_dir = find_segment_dir(db_submaps_root, db_seg)
                q_subset, q_seg_dir = find_segment_dir(q_cam_root, q_seg)

                q_cam_dir = q_seg_dir / args.cam
                if not q_cam_dir.exists():
                    raise FileNotFoundError(f"Q cam dir not found: {q_cam_dir}")

                anchors_csv = db_seg_dir / "anchors.csv"
                if not anchors_csv.exists():
                    raise FileNotFoundError(f"anchors.csv not found: {anchors_csv}")

                # ICP 変換（DB->Q）
                yaw, tx, ty = get_icp_transform_xy(icp_idx, db_seg, q_seg, invert=invert)

                # Q 座標系に変換したアンカー
                a_ids, a_xs, a_ys = load_db_anchors_xy(anchors_csv)
                a_xq, a_yq = apply_se2_xy(a_xs, a_ys, yaw=yaw, tx=tx, ty=ty)

                # Q フレームを読む
                meta_files = list_q_frame_meta(q_cam_dir)

                n_total = len(meta_files)
                n_keep = 0
                n_drop = 0

                for mj in meta_files:
                    q_idx, qx, qy = read_q_frame_xy(mj)

                    # 総当たり最近傍探索（~200x200 程度なら十分速い）
                    best_j = -1
                    best_d2 = 1e30
                    for j in range(len(a_ids)):
                        dx = a_xq[j] - qx
                        dy = a_yq[j] - qy
                        d2 = dx * dx + dy * dy
                        if d2 < best_d2:
                            best_d2 = d2
                            best_j = j

                    dist = math.sqrt(best_d2) if best_j >= 0 else 1e30
                    anchor_id = a_ids[best_j] if best_j >= 0 else -1
                    anchor_png_expected = str(db_seg_dir / str(anchor_id) / "anchor.png") if anchor_id >= 0 else ""

                    q_png = q_cam_dir / f"{q_idx:05d}.png"
                    row = dict(
                        pair_id=pid,
                        lane_tag=lane_tag,
                        db_seg=db_seg,
                        q_seg=q_seg,
                        db_subset=db_subset,
                        q_subset=q_subset,
                        q_frame_index=q_idx,
                        q_meta_json=str(mj),
                        q_image_png=str(q_png),
                        q_x=f"{qx:.6f}",
                        q_y=f"{qy:.6f}",
                        anchor_id=anchor_id,
                        anchor_x_q=f"{a_xq[best_j]:.6f}" if best_j >= 0 else "",
                        anchor_y_q=f"{a_yq[best_j]:.6f}" if best_j >= 0 else "",
                        dist_xy_m=f"{dist:.6f}" if math.isfinite(dist) else "",
                        anchor_png_expected=anchor_png_expected,
                    )

                    if dist <= args.max_match_dist_m:
                        w.writerow(row)
                        matched_rows += 1
                        n_keep += 1
                    else:
                        if w_drop is not None:
                            w_drop.writerow(row)
                        dropped_rows += 1
                        n_drop += 1

                print(f"[PAIR {i}/{len(pairs)}] {pid} | Q_frames={n_total} keep={n_keep} drop={n_drop} | db_subset={db_subset} q_subset={q_subset}")

        finally:
            if write_dropped and out_dropped_csv is not None and f_drop is not None:
                f_drop.close()

    print(f"[OK] wrote matched: {out_csv} rows={matched_rows}")
    if write_dropped and out_dropped_csv is not None:
        print(f"[OK] wrote dropped: {out_dropped_csv} rows={dropped_rows}")
    print("[DONE]")


if __name__ == "__main__":
    main()
