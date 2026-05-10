#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 使い方:
#   python expand_q_to_anchor_softgt.py --help
#   必要な入力パスと出力先を引数で指定して実行する。

"""
expand_q_to_anchor_softgt.py

目的
----
q_to_anchor_all_mergedpaths.csv のような「1クエリ=1アンカー（最近傍/strict GT）」の対応表を、
ICP（db->q 変換）とDBアンカーの座標から「1クエリ=複数アンカー（soft GT）」に拡張して書き出す。

このスクリプトは「評価に使っているペア（val4/test4 など）だけ」を対象にできるように、
--query-manifests（val4_all_pairs.parquet 等）でクエリ集合を指定してフィルタできる。

出力
----
1) <out_csv> : 行= (query, anchor候補) のCSV
   - 各クエリに対して、同一pair内のDBアンカーをICPでq座標系へ変換し、距離<=tmaxのアンカーを全て列挙
   - dist_xy_m を持つので、soft@1m/2m/5m/10m などは dist_xy_m<=閾値 で後から切れる
   - is_gt_t{...}m も出せる（閾値ごとにbool列を作る）

2) <out_csv>.summary.csv : ペア別/全体の「1クエリあたり候補アンカー数」サマリ

想定入力（最低限）
------------------
- q_map_csv: q_to_anchor_all_mergedpaths.csv
  必須列: pair_id, q_frame_index, q_image_png(or cam_png), q_x, q_y
  （他の列はそのまま引き継ぐ）

- db_manifest: val4_all_db.parquet など（複数指定可）
  必須列: pair_id, anchor_png, anchor_x_db, anchor_y_db, anchor_z_db
  代替: anchor_pose_rowmajor があれば anchor_x_db/y_db/z_db を復元

- icp_csv: icp_startend_staticmaps.csv
  必須列: pair_id（無い場合は db_seg/q_seg から segpair_******__****** を作れる形式）,
          final_yaw_deg, final_tx, final_ty, final_tz

注意（重要）
-----------
- 出力行数は「クエリ数 × (tmax内のアンカー数)」になる。tmaxを大きくしすぎると肥大化する。
- ICP精度が悪いペアでは dist_xy_m が系統的にズレる。soft@1mのような厳しい閾値は特に影響を受ける。

"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _str2bool(v: str | bool) -> bool:
    """CLI などで受け取った真偽値表現を bool に正規化する。"""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "on"}


def _normalize_wsl_path(p: str) -> str:
    """WSL と Windows が混在するパス表記を読める形に正規化する。"""
    p = (p or "").strip()
    if not p:
        return p
    if len(p) >= 3 and p[1] == ":" and (p[2] == "\\" or p[2] == "/"):
        drive = p[0].lower()
        rest = p[2:].lstrip("\\/").replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return p


def _read_table(path: Path) -> pd.DataFrame:
    """CSV や Parquet などの表形式ファイルを DataFrame として読み込む。"""
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _ensure_columns(df: pd.DataFrame, required: List[str], where: str) -> None:
    """必要な列が揃っているか確認し、不足時は例外を投げる。"""
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"{where}: missing required columns: {miss}. available={list(df.columns)}")


def _pose_rowmajor_to_xyz(rowmajor: str) -> Optional[Tuple[float, float, float]]:
    """行優先の 4x4 姿勢行列から xyz 平行移動を抽出する。"""
    if not isinstance(rowmajor, str):
        return None
    parts = [p.strip() for p in rowmajor.split(",") if p.strip()]
    if len(parts) != 16:
        return None
    try:
        m = [float(p) for p in parts]
        x = m[3]
        y = m[7]
        z = m[11]
        return x, y, z
    except Exception:
        return None


def _yaw_tx_ty_tz_to_T(yaw_deg: float, tx: float, ty: float, tz: float) -> np.ndarray:
    """yaw と並進成分から 4x4 変換行列を組み立てる。"""
    yaw = math.radians(float(yaw_deg))
    c = math.cos(yaw)
    s = math.sin(yaw)
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    T[0, 3] = float(tx)
    T[1, 3] = float(ty)
    T[2, 3] = float(tz)
    return T


def _load_icp_T_map(icp_csv: Path) -> Dict[str, np.ndarray]:
    """ICP 結果をセグメント対ごとの変換テーブルとして読み込む。"""
    import csv

    if not icp_csv.exists():
        raise FileNotFoundError(f"icp_csv not found: {icp_csv}")

    out: Dict[str, np.ndarray] = {}
    with icp_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            pair_id = (row.get("pair_id") or "").strip()
            if not pair_id:
                db_seg = (row.get("db_seg") or row.get("db_segment") or "").strip()
                q_seg = (row.get("q_seg") or row.get("q_segment") or "").strip()
                if db_seg and q_seg:
                    m1 = re.match(r"segment-([0-9]{6})", db_seg)
                    m2 = re.match(r"segment-([0-9]{6})", q_seg)
                    if m1 and m2:
                        pair_id = f"segpair_{m1.group(1)}__{m2.group(1)}"
            if not pair_id:
                continue

            try:
                yaw = float(row.get("final_yaw_deg"))
                tx = float(row.get("final_tx"))
                ty = float(row.get("final_ty"))
                tz = float(row.get("final_tz"))
            except Exception:
                continue

            out[pair_id] = _yaw_tx_ty_tz_to_T(yaw, tx, ty, tz)

    if not out:
        raise RuntimeError(f"failed to read any ICP rows from: {icp_csv}")
    return out


def _parse_thresholds(s: str) -> List[float]:
    """距離しきい値の文字列を数値列へ変換する。"""
    out: List[float] = []
    for x in str(s).split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    return sorted(set(out))


def main() -> None:
    """CLI 引数を解釈し、入力の読み込みから結果保存までの一連の処理を実行する。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--q-map-csv", type=str, required=True)
    ap.add_argument("--query-manifests", type=str, nargs="*", default=[])
    ap.add_argument("--db-manifests", type=str, nargs="+", required=True)
    ap.add_argument("--icp-csv", type=str, required=True)
    ap.add_argument("--invert-icp", type=str, default="false")

    ap.add_argument("--tmax-m", type=float, default=10.0)
    ap.add_argument("--thresholds-m", type=str, default="1,2,5,10")
    ap.add_argument("--max-anchors-per-query", type=int, default=0)

    ap.add_argument("--out-csv", type=str, required=True)
    args = ap.parse_args()

    q_map_path = Path(_normalize_wsl_path(args.q_map_csv))
    icp_path = Path(_normalize_wsl_path(args.icp_csv))
    out_csv = Path(_normalize_wsl_path(args.out_csv))
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    invert_icp = _str2bool(args.invert_icp)
    tmax = float(args.tmax_m)
    thresholds = _parse_thresholds(args.thresholds_m)

    df_q = pd.read_csv(q_map_path)
    if "q_image_png" not in df_q.columns and "cam_png" in df_q.columns:
        df_q = df_q.rename(columns={"cam_png": "q_image_png"})
    _ensure_columns(df_q, ["pair_id", "q_frame_index", "q_image_png", "q_x", "q_y"], where="q-map-csv")
    df_q = df_q.copy()
    df_q["q_image_png"] = df_q["q_image_png"].astype(str).apply(_normalize_wsl_path)

    if args.query_manifests:
        keep_paths: set[str] = set()
        for p in args.query_manifests:
            mp = Path(_normalize_wsl_path(p))
            mf = _read_table(mp)
            if "cam_png" in mf.columns:
                col = "cam_png"
            elif "q_cam_png" in mf.columns:
                col = "q_cam_png"
            elif "q_image_png" in mf.columns:
                col = "q_image_png"
            else:
                raise ValueError(f"query manifest {mp} must have cam_png / q_cam_png / q_image_png column")
            keep_paths.update(mf[col].astype(str).apply(_normalize_wsl_path).tolist())

        before = len(df_q)
        df_q = df_q[df_q["q_image_png"].isin(keep_paths)].copy()
        after = len(df_q)
        print(f"[INFO] filtered queries by query-manifests: {before} -> {after}")

    if len(df_q) == 0:
        raise RuntimeError("No queries left after filtering. Check paths / manifests.")

    db_frames: List[pd.DataFrame] = []
    for p in args.db_manifests:
        dp = Path(_normalize_wsl_path(p))
        d = _read_table(dp).copy()
        if "anchor_png" not in d.columns and "gt_anchor_png" in d.columns:
            d["anchor_png"] = d["gt_anchor_png"].astype(str)
        _ensure_columns(d, ["pair_id", "anchor_png"], where=f"db-manifest:{dp.name}")

        d["anchor_png"] = d["anchor_png"].astype(str).apply(_normalize_wsl_path)

        if not all(c in d.columns for c in ["anchor_x_db", "anchor_y_db", "anchor_z_db"]):
            if "anchor_pose_rowmajor" in d.columns:
                xyz = d["anchor_pose_rowmajor"].apply(_pose_rowmajor_to_xyz)
                d["anchor_x_db"] = xyz.apply(lambda t: t[0] if t else np.nan)
                d["anchor_y_db"] = xyz.apply(lambda t: t[1] if t else np.nan)
                d["anchor_z_db"] = xyz.apply(lambda t: t[2] if t else np.nan)
            else:
                raise ValueError(
                    f"db-manifest:{dp.name} needs anchor_x_db/anchor_y_db/anchor_z_db or anchor_pose_rowmajor"
                )

        if "anchor_id" not in d.columns:
            def _parse_anchor_id(png: str, fallback: int) -> int:
                """アンカー ID 文字列を構成要素へ分解する。"""
                stem = Path(png).stem
                if stem.isdigit():
                    return int(stem)
                m = re.search(r"(\d+)$", stem)
                if m:
                    return int(m.group(1))
                return int(fallback)

            d = d.reset_index(drop=True)
            d["anchor_id"] = [_parse_anchor_id(png, i) for i, png in enumerate(d["anchor_png"].astype(str).tolist())]

        db_frames.append(d)

    df_db = pd.concat(db_frames, axis=0, ignore_index=True)
    _ensure_columns(df_db, ["pair_id", "anchor_png", "anchor_x_db", "anchor_y_db", "anchor_z_db", "anchor_id"], where="db_all")
    df_db = df_db.reset_index(drop=True)

    used_pairs = set(df_q["pair_id"].astype(str).unique().tolist())
    before_db = len(df_db)
    df_db = df_db[df_db["pair_id"].astype(str).isin(used_pairs)].copy()
    after_db = len(df_db)
    print(f"[INFO] filtered DB by used pairs: {before_db} -> {after_db} (pairs={len(used_pairs)})")

    if len(df_db) == 0:
        raise RuntimeError("DB manifest has no rows after filtering by used pairs. Check pair_id consistency.")

    icp_map = _load_icp_T_map(icp_path)
    missing_icp_pairs = sorted(list(used_pairs - set(icp_map.keys())))
    if missing_icp_pairs:
        print(f"[WARN] ICP missing for {len(missing_icp_pairs)} used pairs. They will be skipped.")
        print("       examples:", missing_icp_pairs[:10])

    db_by_pair: Dict[str, Dict[str, np.ndarray]] = {}
    for pid, g in df_db.groupby("pair_id"):
        pid = str(pid)
        T = icp_map.get(pid)
        if T is None:
            continue
        if invert_icp:
            T = np.linalg.inv(T)

        xyz_db = g[["anchor_x_db", "anchor_y_db", "anchor_z_db"]].to_numpy(dtype=np.float64)
        ones = np.ones((xyz_db.shape[0], 1), dtype=np.float64)
        pts = np.concatenate([xyz_db, ones], axis=1)
        pts_q = (T @ pts.T).T
        xy_q = pts_q[:, :2].astype(np.float64)

        db_by_pair[pid] = {
            "anchor_png": g["anchor_png"].astype(str).to_numpy(),
            "anchor_id": g["anchor_id"].to_numpy(dtype=np.int64),
            "anchor_x_q": xy_q[:, 0],
            "anchor_y_q": xy_q[:, 1],
        }

    carry_cols = [c for c in df_q.columns if c not in {"anchor_id", "anchor_x_q", "anchor_y_q", "dist_xy_m", "anchor_png_expected", "anchor_png"}]

    rows_out: List[Dict[str, object]] = []
    counts_per_query: List[int] = []

    df_q = df_q.reset_index(drop=True)
    strict_anchor_col = None
    if "anchor_png_expected" in df_q.columns:
        strict_anchor_col = "anchor_png_expected"
    elif "anchor_png" in df_q.columns:
        strict_anchor_col = "anchor_png"

    for qi in range(len(df_q)):
        pid = str(df_q.at[qi, "pair_id"])
        if pid not in db_by_pair:
            counts_per_query.append(0)
            continue

        qx = float(df_q.at[qi, "q_x"])
        qy = float(df_q.at[qi, "q_y"])

        cache = db_by_pair[pid]
        ax = cache["anchor_x_q"]
        ay = cache["anchor_y_q"]
        d = np.hypot(ax - qx, ay - qy)

        mask = d <= tmax
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            counts_per_query.append(0)
            continue

        order = np.argsort(d[idxs])
        idxs = idxs[order]

        if args.max_anchors_per_query and args.max_anchors_per_query > 0:
            idxs = idxs[: int(args.max_anchors_per_query)]

        counts_per_query.append(int(idxs.size))

        strict_anchor = None
        if strict_anchor_col is not None:
            strict_anchor = str(df_q.at[qi, strict_anchor_col])
            strict_anchor = _normalize_wsl_path(strict_anchor)

        base: Dict[str, object] = {c: df_q.at[qi, c] for c in carry_cols}

        for j in idxs:
            anchor_png = str(cache["anchor_png"][j])
            dist_m = float(d[j])

            rec = dict(base)
            rec["anchor_png"] = anchor_png
            rec["anchor_id"] = int(cache["anchor_id"][j])
            rec["anchor_x_q"] = float(cache["anchor_x_q"][j])
            rec["anchor_y_q"] = float(cache["anchor_y_q"][j])
            rec["dist_xy_m"] = dist_m
            rec["is_strict_gt"] = (bool(anchor_png == strict_anchor) if strict_anchor is not None else None)

            for t in thresholds:
                tag = str(t).rstrip("0").rstrip(".").replace(".", "p")
                rec[f"is_gt_t{tag}m"] = bool(dist_m <= float(t))

            rows_out.append(rec)

    df_out = pd.DataFrame(rows_out)
    df_out.to_csv(out_csv, index=False)
    print(f"[OK] wrote: {out_csv} rows={len(df_out)}")

    counts = np.asarray(counts_per_query, dtype=np.int64)
    summary_rows: List[Dict[str, object]] = [{
        "scope": "ALL",
        "num_queries": int(len(counts)),
        "num_queries_with_candidates": int(np.sum(counts > 0)),
        "min_candidates": int(counts.min()) if len(counts) else 0,
        "mean_candidates": float(counts.mean()) if len(counts) else float("nan"),
        "p50_candidates": float(np.percentile(counts, 50)) if len(counts) else float("nan"),
        "p90_candidates": float(np.percentile(counts, 90)) if len(counts) else float("nan"),
        "max_candidates": int(counts.max()) if len(counts) else 0,
        "tmax_m": float(tmax),
        "thresholds_m": args.thresholds_m,
    }]

    if len(df_out) > 0 and "pair_id" in df_out.columns:
        for pid, g in df_out.groupby("pair_id"):
            if "q_frame_index" in g.columns:
                qkey = g.groupby(["pair_id", "q_frame_index"]).size().to_numpy(dtype=np.int64)
            else:
                qkey = g.groupby(["pair_id", "q_image_png"]).size().to_numpy(dtype=np.int64)

            summary_rows.append({
                "scope": f"pair:{pid}",
                "num_queries": int(len(qkey)),
                "num_queries_with_candidates": int(np.sum(qkey > 0)),
                "min_candidates": int(qkey.min()) if len(qkey) else 0,
                "mean_candidates": float(qkey.mean()) if len(qkey) else float("nan"),
                "p50_candidates": float(np.percentile(qkey, 50)) if len(qkey) else float("nan"),
                "p90_candidates": float(np.percentile(qkey, 90)) if len(qkey) else float("nan"),
                "max_candidates": int(qkey.max()) if len(qkey) else 0,
                "tmax_m": float(tmax),
                "thresholds_m": args.thresholds_m,
            })

    summary_path = out_csv.with_suffix(out_csv.suffix + ".summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"[OK] wrote: {summary_path}")

    if len(df_out) > 0 and "is_strict_gt" in df_out.columns:
        gk = ["pair_id", "q_frame_index"] if "q_frame_index" in df_out.columns else ["pair_id", "q_image_png"]
        s = df_out.groupby(gk)["is_strict_gt"].any()
        print(f"[INFO] strict GT contained in expanded candidates: {float(s.mean())*100:.2f}%")

    if len(counts) > 0:
        print("[INFO] expanded candidates per query (overall):",
              f"min={int(counts.min())} mean={float(counts.mean()):.2f} "
              f"p50={float(np.percentile(counts,50)):.1f} p90={float(np.percentile(counts,90)):.1f} max={int(counts.max())}")


if __name__ == "__main__":
    main()
