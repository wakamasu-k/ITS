#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 使い方:
#   python build_cross_manifests_from_q_to_anchor_v4_all_and_filtered.py --help
#   必要な入力パスと出力先を引数で指定して実行する。

"""クロス検索用 manifest を組み立てる。

クロスモーダル検索用 manifest を組み立てるスクリプト。
対象は Waymo Cross の DB 側 RI アンカーと Q 側カメラ gray の対応。

このスクリプトは `match_q_frames_to_db_anchors_by_icp.py` が出力した
matched CSV をもとに manifest を生成する。

✅ v4 の挙動（今回の要件）
---------------------------
まず必ず **ALL-IN** manifest（除外なし）を出力する:
  - {name}_all_pairs.(csv|parquet)
  - {name}_all_q.(csv|parquet)
  - {name}_all_db.(csv|parquet)

そのうえで、何らかの除外指定（segment-pair と sample/frame のどちらでも）がある場合は、
要求された除外を **すべて一度に** 反映した filtered 版を **1 セットだけ** 追加で出力する:
  - {name}_excl_<slug>_pairs.(csv|parquet)   （slug は指定された pair_id 由来）
  - {name}_excl_<slug>_q.(csv|parquet)
  - {name}_excl_<slug>_db.(csv|parquet)

除外の意味
----------
- 除外した segment-pair（pair_id）は pairs / q / db から削除する。
- 除外した sample は Q フレーム（cam_png）と、その sample に対応する (pair_id, anchor_id) の関連付けを
  pairs / q から削除する。
- `--exclude_samples_drop_db_anchor true` の場合は、除外 sample に対応する DB anchor も db から削除する。
  その anchor を参照している残り sample も、孤立 GT を避けるため削除する。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd


# -----------------------------
# 補助関数
# -----------------------------

def _normalize_wsl_path(p: str) -> str:
    """Windows ドライブパスを WSL 形式（/mnt/<drive>/...）へ正規化する。"""
    p = (p or "").strip()
    if not p:
        return p
    if len(p) >= 3 and p[1] == ":" and p[2] in ["\\", "/"]:
        drive = p[0].lower()
        rest = p[2:].lstrip("\\/").replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return p


def _read_txt_lines(p: Path) -> List[str]:
    """空行やコメントを除いてテキスト行を読み込む。"""
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]


PAIR_LINE_RE = re.compile(r"^(segment-[0-9]+_.+?_with_camera_labels)$")


def short_id6(seg: str) -> str:
    """長い ID から比較や表示に使う短縮 6 文字 ID を作る。"""
    m = re.match(r"segment-([0-9]{6})", seg)
    return m.group(1) if m else seg[:6]


@dataclass(frozen=True)
class PairSpec:
    """query-anchor ペア生成時に使う 1 件分の条件を保持する。"""
    pair_id: str
    lane_tag: str
    db_seg: str
    q_seg: str


def parse_pairs_triplets(pairs_path: Path) -> List[PairSpec]:
    """triplets file を解析する。

    lane_tag\n
    db_segment\n
    q_segment\n

    repeated.
    """
    lines = _read_txt_lines(pairs_path)
    if len(lines) % 3 != 0:
        raise ValueError(f"pairs file must be multiples of 3 lines: {pairs_path} (got {len(lines)})")

    out: List[PairSpec] = []
    for i in range(0, len(lines), 3):
        lane_tag = lines[i]
        db_seg = lines[i + 1]
        q_seg = lines[i + 2]
        # 厳しく縛らず、軽めに検証する
        if not PAIR_LINE_RE.match(db_seg):
            pass
        if not PAIR_LINE_RE.match(q_seg):
            pass
        pair_id = f"segpair_{short_id6(db_seg)}__{short_id6(q_seg)}"
        out.append(PairSpec(pair_id=pair_id, lane_tag=lane_tag, db_seg=db_seg, q_seg=q_seg))
    return out


def canon_int_str(x) -> str:
    """整数らしい値の表記を正規化する。例: '42', '00042' -> '42'。"""
    if x is None:
        return ""
    try:
        s = str(x).strip()
        if s == "":
            return ""
        # '00010.png' のような名前も扱う
        s2 = s
        if s2.lower().endswith(".png"):
            s2 = os.path.splitext(os.path.basename(s2))[0]
        return str(int(float(s2)))
    except Exception:
        return str(x).strip()


# -----------------------------
# Waymo TFRecord のメタ情報
# -----------------------------


@dataclass
class SegmentMeta:
    """各セグメントの属性や補助情報を保持する。"""
    segment: str
    subset: str
    tfrecord_path: str
    weather: str
    time_of_day: str
    n_frames: int


def _normalize_weather(s: str) -> str:
    """天候ラベルの表記ゆれを吸収して正規化する。"""
    if not s:
        return "unknown"
    t = s.strip()
    tl = t.lower()
    if "sun" in tl:
        return "Sunny"
    if "rain" in tl:
        return "Rain"
    return t


def _normalize_time_of_day(s: str) -> str:
    """時間帯ラベルの表記ゆれを吸収して正規化する。"""
    if not s:
        return "unknown"
    t = s.strip()
    tl = t.lower()
    if tl in ["day", "daytime"]:
        return "Day"
    if "dawn" in tl or "dusk" in tl:
        return "Dawn/Dusk"
    if "night" in tl:
        return "Night"
    return t


def find_tfrecord_for_segment(tfrecord_root: Path, segment: str, subset_hint: Optional[str] = None) -> Tuple[str, str]:
    """segment に対応する TFRecord パスと subset フォルダを探す。"""
    cand_subsets: List[str] = []
    if subset_hint:
        cand_subsets.append(subset_hint)
    cand_subsets += ["training", "validation", "testing", "domain_adaptation", "testing_3d_camera_only_detection"]
    # そのほかの直下ディレクトリも候補に加える
    if tfrecord_root.exists():
        for p in tfrecord_root.iterdir():
            if p.is_dir() and p.name not in cand_subsets:
                cand_subsets.append(p.name)

    hits: List[Tuple[str, str]] = []
    for subset in cand_subsets:
        fp = tfrecord_root / subset / f"{segment}.tfrecord"
        if fp.exists():
            hits.append((subset, str(fp)))
    if len(hits) == 1:
        return hits[0]
    if len(hits) == 0:
        raise FileNotFoundError(f"TFRecord not found for segment={segment} under {tfrecord_root}")
    priority = {name: i for i, name in enumerate(["training", "validation", "testing"])}
    hits.sort(key=lambda x: priority.get(x[0], 999))
    return hits[0]


def read_segment_meta(
    tfrecord_root: Path,
    segment: str,
    subset_hint: Optional[str],
    cache: Dict[str, dict],
) -> SegmentMeta:
    """セグメント単位の補助メタ情報を読み込む。"""
    if segment in cache:
        d = cache[segment]
        return SegmentMeta(
            segment=segment,
            subset=d.get("subset", "unknown"),
            tfrecord_path=d.get("tfrecord_path", ""),
            weather=d.get("weather", "unknown"),
            time_of_day=d.get("time_of_day", "unknown"),
            n_frames=int(d.get("n_frames", -1)),
        )

    subset, tf_path = find_tfrecord_for_segment(tfrecord_root, segment, subset_hint=subset_hint)

    try:
        import tensorflow as tf  # type: ignore
        from waymo_open_dataset import dataset_pb2  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Failed to import tensorflow / waymo_open_dataset. Run this script inside the Waymo env."
        ) from e

    ds = tf.data.TFRecordDataset([tf_path], compression_type="")
    n_frames = 0
    weather = "unknown"
    tod = "unknown"
    first = True
    for raw in ds:
        if first:
            fr = dataset_pb2.Frame()
            fr.ParseFromString(bytes(raw.numpy()))
            try:
                weather = _normalize_weather(getattr(fr.context.stats, "weather", "") or "")
            except Exception:
                weather = "unknown"
            try:
                tod = _normalize_time_of_day(getattr(fr.context.stats, "time_of_day", "") or "")
            except Exception:
                tod = "unknown"
            first = False
        n_frames += 1

    cache[segment] = {
        "subset": subset,
        "tfrecord_path": tf_path,
        "weather": weather,
        "time_of_day": tod,
        "n_frames": n_frames,
    }

    return SegmentMeta(
        segment=segment,
        subset=subset,
        tfrecord_path=tf_path,
        weather=weather,
        time_of_day=tod,
        n_frames=n_frames,
    )


# -----------------------------
# 除外指定の解析
# -----------------------------

def parse_exclude_pair_ids(exclude_pair_id_flags: Sequence[str], exclude_pair_ids_txt: str) -> List[str]:
    """除外対象ペア ID の一覧を読み込む。"""
    ids: List[str] = []

    # フラグは複数回指定でき、1 つの引数にカンマ区切りで複数値を入れることもできる。
    for x in exclude_pair_id_flags or []:
        for t in str(x).replace(" ", "").split(","):
            if t:
                ids.append(t)

    if exclude_pair_ids_txt:
        p = Path(_normalize_wsl_path(exclude_pair_ids_txt))
        if p.exists():
            for ln in _read_txt_lines(p):
                for t in ln.replace(" ", "").split(","):
                    if t:
                        ids.append(t)

    # 重複を除きつつ順序は維持する
    seen = set()
    out = []
    for t in ids:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _parse_exclude_sample_inline(s: str) -> Tuple[str, str, Optional[str]]:
    """--exclude_sample 文字列を解析する。

    Supported:
      - "pair_id,q_frame_index" (anchor is auto-resolved from matched_csv)
      - "pair_id,q_frame_index,anchor_id" (explicit)
      - "pair_id,cam_png" (if it looks like a path ending with .png)
    """
    raw = (s or "").strip()
    if not raw:
        raise ValueError("empty exclude_sample")

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) < 2:
        raise ValueError(f"exclude_sample needs at least 2 fields: {raw}")

    pair_id = parts[0]
    b = parts[1]

    # 2 番目の項目が png パスなら cam_png とみなす。
    if b.lower().endswith(".png") or "/" in b or "\\" in b:
        cam_png = _normalize_wsl_path(b)
        return pair_id, cam_png, None

    q_frame = canon_int_str(b)
    anchor_id = canon_int_str(parts[2]) if len(parts) >= 3 else None
    return pair_id, q_frame, anchor_id


def parse_excluded_samples(
    df_all: pd.DataFrame,
    exclude_sample_flags: Sequence[str],
    exclude_samples_csv: str,
) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]], Set[str]]:
    """(excluded_qkeys, excluded_anchor_keys, excluded_cam_paths) を返す。

    - excluded_qkeys: (pair_id, q_frame_str)
    - excluded_anchor_keys: (pair_id, anchor_id_str)
    - excluded_cam_paths: cam_png path strings

    If anchor_id is not given, it is auto-resolved from df_all using pair_id+q_frame.
    """
    excluded_qkeys: Set[Tuple[str, str]] = set()
    excluded_anchor_keys: Set[Tuple[str, str]] = set()
    excluded_cam_paths: Set[str] = set()

    # 1) フラグから取得
    for s in exclude_sample_flags or []:
        pid, b, aid = _parse_exclude_sample_inline(s)
        if b.lower().endswith(".png") or "/" in b:
            excluded_cam_paths.add(_normalize_wsl_path(b))
            continue
        qf = canon_int_str(b)
        excluded_qkeys.add((pid, qf))
        if aid is not None and aid != "":
            excluded_anchor_keys.add((pid, canon_int_str(aid)))

    # 2) CSV から取得
    if exclude_samples_csv:
        p = Path(_normalize_wsl_path(exclude_samples_csv))
        if not p.exists():
            raise FileNotFoundError(f"exclude_samples_csv not found: {p}")
        with p.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = (row.get("pair_id") or "").strip()
                if not pid:
                    continue
                cam_png = (row.get("cam_png") or row.get("q_image_png") or "").strip()
                if cam_png:
                    excluded_cam_paths.add(_normalize_wsl_path(cam_png))
                qf = row.get("q_frame_index")
                if qf is None:
                    qf = row.get("q_frame")
                if qf is not None and str(qf).strip() != "":
                    excluded_qkeys.add((pid, canon_int_str(qf)))
                aid = row.get("anchor_id")
                if aid is not None and str(aid).strip() != "":
                    excluded_anchor_keys.add((pid, canon_int_str(aid)))

    # 3) (pair_id, q_frame) から不足している anchor_id を自動解決する
    #    これにより (pair_id, q_frame_index) だけを指定しても対応する anchor を除外できる。
    if excluded_qkeys:
        # df から高速参照用の辞書を作る
        # q_frame_index があればそれを使い、なければファイル名から推定する
        pid_col = "pair_id" if "pair_id" in df_all.columns else None
        if pid_col is not None:
            # 照合用に q_frame_key 列を作る
            if "_q_frame_key" not in df_all.columns:
                if "q_frame_index" in df_all.columns:
                    df_all["_q_frame_key"] = df_all["q_frame_index"].apply(canon_int_str)
                elif "q_frame" in df_all.columns:
                    df_all["_q_frame_key"] = df_all["q_frame"].apply(canon_int_str)
                elif "cam_png" in df_all.columns:
                    df_all["_q_frame_key"] = df_all["cam_png"].apply(canon_int_str)
                elif "q_image_png" in df_all.columns:
                    df_all["_q_frame_key"] = df_all["q_image_png"].apply(canon_int_str)
                else:
                    df_all["_q_frame_key"] = ""

            if "anchor_id" not in df_all.columns:
                # 代替の列名も試す
                if "db_anchor_id" in df_all.columns:
                    df_all["anchor_id"] = df_all["db_anchor_id"]

            if "anchor_id" in df_all.columns:
                # (pid, qf) -> set(anchor_id) の対応を作る
                for pid, qf in list(excluded_qkeys):
                    m = (df_all["pair_id"] == pid) & (df_all["_q_frame_key"] == qf)
                    if not m.any():
                        continue
                    aids = df_all.loc[m, "anchor_id"].tolist()
                    for a in aids:
                        aa = canon_int_str(a)
                        if aa != "":
                            excluded_anchor_keys.add((pid, aa))

    return excluded_qkeys, excluded_anchor_keys, excluded_cam_paths


def _slugify_pair_ids(pair_ids: Sequence[str]) -> str:
    """ペア ID 群をファイル名向けの短い文字列へ整形する。"""
    if not pair_ids:
        return "filtered"
    s = "__".join(pair_ids)
    # ファイル名として扱いやすい形にする
    s = s.replace("/", "_").replace("\\", "_").replace(":", "_")
    s = s.replace(" ", "_")
    s = s.replace(".", "_")
    s = s.replace("-", "_")
    # 連続するアンダースコアも 1 つにまとめる
    while "__" in s:
        s = s.replace("__", "_")
    return s


# -----------------------------
# 中核の manifest 構築処理
# -----------------------------

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """列名を val2 manifest で使っている名前へ正規化する。"""
    df = df.copy()

    # cam_png 列
    if "cam_png" not in df.columns:
        if "q_image_png" in df.columns:
            df["cam_png"] = df["q_image_png"].astype(str)
        else:
            raise ValueError("matched_csv must have cam_png or q_image_png")

    # anchor_png 列
    if "anchor_png" not in df.columns:
        if "anchor_png_expected" in df.columns:
            df["anchor_png"] = df["anchor_png_expected"].astype(str)
        else:
            raise ValueError("matched_csv must have anchor_png or anchor_png_expected")

    # anchor_id 列
    if "anchor_id" not in df.columns:
        raise ValueError("matched_csv must have anchor_id")

    # pair_id 列
    if "pair_id" not in df.columns:
        raise ValueError("matched_csv must have pair_id")

    # db/q セグメント列
    if "db_seg" not in df.columns or "q_seg" not in df.columns:
        raise ValueError("matched_csv must have db_seg and q_seg")

    # subset 列
    if "db_subset" not in df.columns:
        df["db_subset"] = ""
    if "q_subset" not in df.columns:
        df["q_subset"] = ""

    # lane_tag 列
    if "lane_tag" not in df.columns:
        df["lane_tag"] = ""

    # q_frame_index 列（任意）
    if "q_frame_index" not in df.columns:
        # 可能ならファイル名からも解釈する
        df["q_frame_index"] = df["cam_png"].apply(lambda p: int(canon_int_str(p)) if canon_int_str(p) != "" else -1)

    return df


def _verify_files_for_df(df_pairs: pd.DataFrame, df_db: pd.DataFrame) -> None:
    """DataFrame が参照するファイル群の存在を検証する。"""
    missing = 0
    for p in df_pairs["cam_png"].astype(str).tolist():
        if not os.path.exists(p):
            missing += 1
            print(f"[MISS] cam_png: {p}")
    for p in df_pairs["anchor_png"].astype(str).tolist():
        if not os.path.exists(p):
            missing += 1
            print(f"[MISS] anchor_png: {p}")
    for p in df_db["anchor_png"].astype(str).tolist():
        if not os.path.exists(p):
            missing += 1
            print(f"[MISS] db_anchor_png: {p}")
    if missing:
        raise RuntimeError(f"verify_files failed: missing_files={missing}")


def _write_outputs(
    out_dir: Path,
    prefix: str,
    pairs_df: pd.DataFrame,
    q_df: pd.DataFrame,
    db_df: pd.DataFrame,
    write_parquet: bool,
) -> None:
    """生成した manifest と補助ファイルをまとめて書き出す。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs_csv = out_dir / f"{prefix}_pairs.csv"
    q_csv = out_dir / f"{prefix}_q.csv"
    db_csv = out_dir / f"{prefix}_db.csv"

    pairs_df.to_csv(pairs_csv, index=False)
    q_df.to_csv(q_csv, index=False)
    db_df.to_csv(db_csv, index=False)

    print(f"[OK] wrote: {pairs_csv} rows={len(pairs_df)}")
    print(f"[OK] wrote: {q_csv} rows={len(q_df)}")
    print(f"[OK] wrote: {db_csv} rows={len(db_df)}")

    if write_parquet:
        try:
            pairs_parq = out_dir / f"{prefix}_pairs.parquet"
            q_parq = out_dir / f"{prefix}_q.parquet"
            db_parq = out_dir / f"{prefix}_db.parquet"
            pairs_df.to_parquet(pairs_parq, index=False)
            q_df.to_parquet(q_parq, index=False)
            db_df.to_parquet(db_parq, index=False)
            print(f"[OK] wrote: {pairs_parq} rows={len(pairs_df)}")
            print(f"[OK] wrote: {q_parq} rows={len(q_df)}")
            print(f"[OK] wrote: {db_parq} rows={len(db_df)}")
        except Exception as e:
            raise RuntimeError("Failed to write parquet (install pyarrow).") from e


def build_manifests(
    *,
    pairs_txt: Path,
    matched_csv: Path,
    db_submaps_root: Path,
    q_cam_root: Path,
    tfrecord_root: Path,
    out_dir: Path,
    name: str,
    exclude_pair_ids: Sequence[str],
    exclude_sample_flags: Sequence[str],
    exclude_samples_csv: str,
    exclude_samples_drop_db_anchor: bool,
    write_parquet: bool,
    verify_files: bool,
) -> None:
    """query-anchor 組み合わせから学習・評価用 manifest を構築する。"""
    pairs = parse_pairs_triplets(pairs_txt)
    print(f"[INFO] pairs={len(pairs)} from {pairs_txt}")

    df_all = pd.read_csv(matched_csv)
    df_all = _ensure_columns(df_all)

    # 安全のため、pairs_txt に含まれる pair_id の行だけ残す
    pair_id_set = {p.pair_id for p in pairs}
    before = len(df_all)
    df_all = df_all[df_all["pair_id"].astype(str).isin(pair_id_set)].copy()
    after = len(df_all)
    if after != before:
        print(f"[WARN] matched_csv rows filtered by pairs_txt: {before} -> {after}")

    # db/q セグメント向けの補助メタ（weather/time_of_day）を作る
    segments_db = sorted({p.db_seg for p in pairs})
    segments_q = sorted({p.q_seg for p in pairs})
    segments_all = sorted(set(segments_db + segments_q))

    cache_path = out_dir / f"{name}_segment_meta_cache.json"
    cache: Dict[str, dict] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
            if not isinstance(cache, dict):
                cache = {}
        except Exception:
            cache = {}

    meta: Dict[str, SegmentMeta] = {}
    for seg in segments_all:
        # 可能なら df_all 側の subset ヒントを優先する
        hint = None
        if seg in segments_db:
            # DB 側 subset ヒント
            rows = df_all[df_all["db_seg"] == seg]
            if len(rows) > 0:
                h = str(rows.iloc[0].get("db_subset", "")).strip()
                hint = h if h else None
        else:
            rows = df_all[df_all["q_seg"] == seg]
            if len(rows) > 0:
                h = str(rows.iloc[0].get("q_subset", "")).strip()
                hint = h if h else None

        try:
            meta[seg] = read_segment_meta(tfrecord_root, seg, subset_hint=hint, cache=cache)
        except Exception as e:
            print(f"[WARN] failed to read meta for {seg}: {e}")
            meta[seg] = SegmentMeta(seg, hint or "unknown", "", "unknown", "unknown", -1)

    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    # メタ列を df_all に埋める
    df_all["db_weather"] = df_all["db_seg"].map(lambda s: meta.get(s).weather if meta.get(s) else "unknown")
    df_all["db_time_of_day"] = df_all["db_seg"].map(lambda s: meta.get(s).time_of_day if meta.get(s) else "unknown")
    df_all["q_weather"] = df_all["q_seg"].map(lambda s: meta.get(s).weather if meta.get(s) else "unknown")
    df_all["q_time_of_day"] = df_all["q_seg"].map(lambda s: meta.get(s).time_of_day if meta.get(s) else "unknown")

    # subset 列が空なら meta.subset から補う
    df_all.loc[df_all["db_subset"].astype(str).str.len() == 0, "db_subset"] = df_all["db_seg"].map(
        lambda s: meta.get(s).subset if meta.get(s) else "unknown"
    )
    df_all.loc[df_all["q_subset"].astype(str).str.len() == 0, "q_subset"] = df_all["q_seg"].map(
        lambda s: meta.get(s).subset if meta.get(s) else "unknown"
    )

    # ---- DB manifest（全件） ----
    db_rows: List[dict] = []
    for ps in pairs:
        db_subset = meta.get(ps.db_seg).subset if meta.get(ps.db_seg) else "unknown"
        seg_dir = db_submaps_root / db_subset / ps.db_seg
        anchors_csv = seg_dir / "anchors.csv"
        if not anchors_csv.exists():
            raise FileNotFoundError(f"anchors.csv not found: {anchors_csv}")
        a_df = pd.read_csv(anchors_csv)
        for _, r in a_df.iterrows():
            aid = int(r["anchor_id"]) if "anchor_id" in r else int(r["anchor"])
            db_rows.append(
                {
                    "pair_id": ps.pair_id,
                    "lane_tag": ps.lane_tag,
                    "db_subset": db_subset,
                    "db_seg": ps.db_seg,
                    "anchor_id": aid,
                    "anchor_png": str(seg_dir / str(aid) / "anchor.png"),
                    "is_virtual": int(r.get("is_virtual", 0)),
                    "db_weather": meta.get(ps.db_seg).weather if meta.get(ps.db_seg) else "unknown",
                    "db_time_of_day": meta.get(ps.db_seg).time_of_day if meta.get(ps.db_seg) else "unknown",
                    "anchor_T_wv_rowmajor": str(r.get("T_wv_rowmajor", "")),
                    "anchor_x_db": float(r.get("x", float("nan"))),
                    "anchor_y_db": float(r.get("y", float("nan"))),
                    "anchor_z_db": float(r.get("z", float("nan"))),
                }
            )

    db_df_all = pd.DataFrame(db_rows)

    # ---- PAIRS/Q manifest（全件） ----
    # 既存のスキーマ（val2_all_pairs.csv / val2_all_q.csv）を維持する
    pairs_df_all = pd.DataFrame(
        {
            "cam_png": df_all["cam_png"].astype(str).map(_normalize_wsl_path),
            "anchor_png": df_all["anchor_png"].astype(str).map(_normalize_wsl_path),
            "label": 1,
            "pair_id": df_all["pair_id"].astype(str),
            "lane_tag": df_all["lane_tag"],
            "db_subset": df_all["db_subset"].astype(str),
            "q_subset": df_all["q_subset"].astype(str),
            "db_seg": df_all["db_seg"].astype(str),
            "q_seg": df_all["q_seg"].astype(str),
            "anchor_id": df_all["anchor_id"],
            "db_weather": df_all["db_weather"].astype(str),
            "db_time_of_day": df_all["db_time_of_day"].astype(str),
            "q_weather": df_all["q_weather"].astype(str),
            "q_time_of_day": df_all["q_time_of_day"].astype(str),
        }
    )

    # query ごとに 1 行なので、現状では q_df は同一内容になる。
    q_df_all = pairs_df_all.copy()

    # 任意のファイル存在確認（ALL-IN）
    if verify_files:
        print("[INFO] verify_files=true (ALL-IN)")
        _verify_files_for_df(pairs_df_all, db_df_all)

    # ALL-IN 出力を書き出す
    _write_outputs(out_dir, f"{name}_all", pairs_df_all, q_df_all, db_df_all, write_parquet)

    # ---------------------------------------------------------
    # 指定された除外をすべて反映した filtered 版を 1 セットだけ作る
    # ---------------------------------------------------------
    excluded_qkeys, excluded_anchor_keys, excluded_cam_paths = parse_excluded_samples(
        df_all=df_all,
        exclude_sample_flags=exclude_sample_flags,
        exclude_samples_csv=exclude_samples_csv,
    )

    has_any_excl = bool(exclude_pair_ids) or bool(excluded_qkeys) or bool(excluded_anchor_keys) or bool(excluded_cam_paths)
    if not has_any_excl:
        print("[INFO] no exclusions specified -> filtered manifests are not generated")
        return

    print(
        f"[INFO] exclusions: exclude_pair_ids={len(exclude_pair_ids)}, exclude_qkeys={len(excluded_qkeys)}, "
        f"exclude_anchor_keys={len(excluded_anchor_keys)}, exclude_cam_paths={len(excluded_cam_paths)}"
    )

    # 1) まず pair_id 除外で DB を絞る
    db_df_f = db_df_all.copy()
    if exclude_pair_ids:
        db_df_f = db_df_f[~db_df_f["pair_id"].astype(str).isin(set(exclude_pair_ids))].copy()

    # 2) 指定されていれば、対応する anchor も DB から落とす
    if exclude_samples_drop_db_anchor and excluded_anchor_keys:
        # 高速フィルタ用のキー列を作る
        db_df_f["_k"] = db_df_f.apply(lambda r: (str(r["pair_id"]), canon_int_str(r["anchor_id"])), axis=1)
        keep_mask = ~db_df_f["_k"].isin(excluded_anchor_keys)
        dropped_n = int((~keep_mask).sum())
        if dropped_n:
            print(f"[INFO] dropping DB anchors by excluded samples: {dropped_n}")
        db_df_f = db_df_f[keep_mask].copy()
        db_df_f.drop(columns=["_k"], inplace=True, errors="ignore")

    # 3) pair_id と sample 除外で df_all（pairs/q 行）を絞る
    df_f = df_all.copy()

    # pair_id 列 exclusion
    if exclude_pair_ids:
        df_f = df_f[~df_f["pair_id"].astype(str).isin(set(exclude_pair_ids))].copy()

    # sample 除外マスク
    if excluded_qkeys or excluded_anchor_keys or excluded_cam_paths:
        def is_excluded_row(row) -> bool:
            """行が除外条件に該当するかどうかを判定する。"""
            pid = str(row.get("pair_id", ""))
            cam_png = _normalize_wsl_path(str(row.get("cam_png", "")))
            if cam_png and cam_png in excluded_cam_paths:
                return True
            qf = canon_int_str(row.get("q_frame_index", None) if "q_frame_index" in row else row.get("q_frame", None))
            if qf and (pid, qf) in excluded_qkeys:
                return True
            aid = canon_int_str(row.get("anchor_id", None))
            if aid and (pid, aid) in excluded_anchor_keys:
                return True
            return False

        m = df_f.apply(is_excluded_row, axis=1)
        dropped = int(m.sum())
        if dropped:
            print(f"[INFO] dropping samples by explicit exclusions: {dropped}")
        df_f = df_f[~m].copy()

    # 4) 一貫性を保つため、GT anchor が DB に存在しない残り sample を除去する
    #    （sample 除外により DB anchor を削除したときに起こりうる）
    db_keys: Set[Tuple[str, str]] = set(
        (str(r.pair_id), canon_int_str(r.anchor_id)) for r in db_df_f.itertuples(index=False)
    )

    df_f["_k"] = df_f.apply(lambda r: (str(r["pair_id"]), canon_int_str(r["anchor_id"])), axis=1)
    keep_mask = df_f["_k"].isin(db_keys)
    orphan_n = int((~keep_mask).sum())
    if orphan_n:
        print(f"[INFO] dropping orphan samples (GT anchor not in filtered DB): {orphan_n}")
    df_orphans = df_f[~keep_mask].copy()
    df_f = df_f[keep_mask].copy()
    df_f.drop(columns=["_k"], inplace=True, errors="ignore")

    # フィルタ済みの pairs/q を構築する
    pairs_df_f = pd.DataFrame(
        {
            "cam_png": df_f["cam_png"].astype(str).map(_normalize_wsl_path),
            "anchor_png": df_f["anchor_png"].astype(str).map(_normalize_wsl_path),
            "label": 1,
            "pair_id": df_f["pair_id"].astype(str),
            "lane_tag": df_f["lane_tag"],
            "db_subset": df_f["db_subset"].astype(str),
            "q_subset": df_f["q_subset"].astype(str),
            "db_seg": df_f["db_seg"].astype(str),
            "q_seg": df_f["q_seg"].astype(str),
            "anchor_id": df_f["anchor_id"],
            "db_weather": df_f["db_weather"].astype(str),
            "db_time_of_day": df_f["db_time_of_day"].astype(str),
            "q_weather": df_f["q_weather"].astype(str),
            "q_time_of_day": df_f["q_time_of_day"].astype(str),
        }
    )
    q_df_f = pairs_df_f.copy()

    # 任意の検証（FILTERED）
    if verify_files:
        print("[INFO] verify_files=true (FILTERED)")
        _verify_files_for_df(pairs_df_f, db_df_f)

    # FILTERED 出力タグを決める
    slug = _slugify_pair_ids(exclude_pair_ids) if exclude_pair_ids else "filtered"
    prefix = f"{name}_excl_{slug}"

    _write_outputs(out_dir, prefix, pairs_df_f, q_df_f, db_df_f, write_parquet)

    # 除外された項目を後で調べやすいよう、解析用 dump を書き出す
    if len(df_orphans) > 0:
        p = out_dir / f"{prefix}_dropped_orphans.csv"
        df_orphans.to_csv(p, index=False)
        print(f"[OK] wrote dropped-orphans: {p} rows={len(df_orphans)}")


# -----------------------------
# CLI 引数
# -----------------------------

def main() -> None:
    """CLI 引数を解釈し、入力の読み込みから結果保存までの一連の処理を実行する。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_txt", required=True, help="triplets pairs file (lane_tag/db/q repeating)")
    ap.add_argument("--matched_csv", required=True, help="matched q->anchor csv (e.g., val2_q_to_anchor.csv)")
    ap.add_argument("--db_submaps_root", required=True, help="DB submaps root (contains anchors.csv and anchor.png dirs)")
    ap.add_argument("--q_cam_root", required=True, help="Q cam gray root (not strictly needed; kept for compatibility)")
    ap.add_argument("--tfrecord_root", required=True, help="Waymo individual_files root")
    ap.add_argument("--out_dir", required=True, help="output directory")
    ap.add_argument("--name", required=True, help="name prefix for outputs")

    # pair 単位の除外設定
    ap.add_argument(
        "--exclude_pair_id",
        action="append",
        default=[],
        help="pair_id to exclude. Can be repeated. Comma-separated values also supported.",
    )
    ap.add_argument(
        "--exclude_pair_ids_txt",
        type=str,
        default="",
        help="text file listing pair_id to exclude (one per line or comma-separated)",
    )

    # sample 単位の除外設定
    ap.add_argument(
        "--exclude_sample",
        action="append",
        default=[],
        help=(
            "Exclude a sample. Formats: "
            "'pair_id,q_frame_index' or 'pair_id,q_frame_index,anchor_id' or 'pair_id,/path/to/cam.png'. "
            "Can be repeated."
        ),
    )
    ap.add_argument(
        "--exclude_samples_csv",
        type=str,
        default="",
        help="CSV with columns like pair_id,q_frame_index,anchor_id,cam_png to exclude.",
    )
    ap.add_argument(
        "--exclude_samples_drop_db_anchor",
        type=str,
        default="true",
        help="If true, excluded samples also remove the corresponding DB anchor (and dependent samples).",
    )

    ap.add_argument("--write_parquet", type=str, default="true")
    ap.add_argument("--verify_files", type=str, default="false")

    args = ap.parse_args()

    pairs_txt = Path(_normalize_wsl_path(args.pairs_txt))
    matched_csv = Path(_normalize_wsl_path(args.matched_csv))
    db_submaps_root = Path(_normalize_wsl_path(args.db_submaps_root))
    q_cam_root = Path(_normalize_wsl_path(args.q_cam_root))
    tfrecord_root = Path(_normalize_wsl_path(args.tfrecord_root))
    out_dir = Path(_normalize_wsl_path(args.out_dir))

    exclude_pair_ids = parse_exclude_pair_ids(args.exclude_pair_id, args.exclude_pair_ids_txt)

    def _parse_bool(x: str) -> bool:
        """CLI などで受け取った真偽値表現を bool に正規化する。"""
        return str(x).strip().lower() in ["1", "true", "yes", "y", "on"]

    build_manifests(
        pairs_txt=pairs_txt,
        matched_csv=matched_csv,
        db_submaps_root=db_submaps_root,
        q_cam_root=q_cam_root,
        tfrecord_root=tfrecord_root,
        out_dir=out_dir,
        name=str(args.name),
        exclude_pair_ids=exclude_pair_ids,
        exclude_sample_flags=args.exclude_sample,
        exclude_samples_csv=str(args.exclude_samples_csv),
        exclude_samples_drop_db_anchor=_parse_bool(args.exclude_samples_drop_db_anchor),
        write_parquet=_parse_bool(args.write_parquet),
        verify_files=_parse_bool(args.verify_files),
    )

    print("[DONE]")


if __name__ == "__main__":
    main()
