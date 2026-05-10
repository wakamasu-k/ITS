#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 使い方:
#   python render_anchor_intensity_from_segments_auto_subset.py --help
#   必要な入力パスと出力先を引数で指定して実行する。

"""セグメント一覧から anchor submap 用 RI（反射強度）画像を描画する。

このスクリプトは、既存の `render_anchor_intensity_txt.py` を呼び出すための薄いラッパー。

------------------------
`render_anchor_intensity_txt.py` は `--subset`（training/validation/testing）の指定が必要。
一方で DB セグメント一覧（例: val2_db.txt, test2_db.txt）には、`individual_files` 配下で
training と validation の両方にまたがって存在するセグメントが含まれることがある。

そのため、このスクリプトでは次を行う:
  1) `--segments_txt` からセグメント一覧を読む
  2) `<tfrecord_root>/<subset>/<segment>.tfrecord` を確認して各セグメントの subset を自動判定する
  3) subset ごとの list file を書き出す
  4) subset ごとに `render_anchor_intensity_txt.py` を 1 回ずつ呼ぶ

出力
----
実際の出力は `render_anchor_intensity_txt.py` 側がそのまま書き出す。
つまり保存先は `--submaps_root/<subset>/<segment>/<sub_id>/...` になる。
このラッパー自身が作るのは、生成した list file を置く小さな `_lists/` ディレクトリだけ。


"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# -------------------------
# ユーティリティ
# -------------------------

def str2bool(v: str) -> bool:
    """CLI などで受け取った真偽値表現を bool に正規化する。"""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {v}")


def _normalize_wsl_path(p: str) -> str:
    """D:\\... のような Windows パスを WSL の /mnt/d/... 形式へ正規化する。"""
    p = (p or "").strip()
    if not p:
        return p
    if len(p) >= 3 and p[1] == ":" and (p[2] == "\\" or p[2] == "/"):
        drive = p[0].lower()
        rest = p[2:].lstrip("\\/").replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return p


def read_segments(txt: Path) -> List[str]:
    """テキストファイルからセグメント名（stem）の一覧を読む。"""
    segs: List[str] = []
    for raw in txt.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        # 誤って '.tfrecord' が付いていても受け付ける
        if line.endswith(".tfrecord"):
            line = Path(line).stem
        segs.append(line)

    # 重複を除去しつつ順序は維持する
    seen = set()
    uniq: List[str] = []
    for s in segs:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def detect_subset(tfrecord_root: Path, segment: str, subsets: List[str]) -> Tuple[str, Path]:
    """指定 segment が存在する (subset, tfrecord_path) を返す。"""
    hits: List[Tuple[str, Path]] = []
    for subset in subsets:
        p = tfrecord_root / subset / f"{segment}.tfrecord"
        if p.exists():
            hits.append((subset, p))

    if not hits:
        raise FileNotFoundError(f"TFRecord not found for segment={segment} under {tfrecord_root}")

    # 複数見つかった場合は training -> validation -> testing -> その他の順で優先する
    priority = {name: i for i, name in enumerate(["training", "validation", "testing"]) }
    hits.sort(key=lambda x: priority.get(x[0], 999))
    return hits[0][0], hits[0][1]


def run_cmd(cmd: List[str]) -> int:
    """外部コマンドを実行し、失敗時はその場で検出する。"""
    print("[RUN] " + " ".join(cmd), flush=True)
    return subprocess.call(cmd)


# -------------------------
# メイン処理
# -------------------------

def main() -> None:
    """CLI 引数を解釈し、入力の読み込みから結果保存までの一連の処理を実行する。"""
    ap = argparse.ArgumentParser()

    ap.add_argument("--segments_txt", "--segments-txt", type=str, required=True, help="text file of segment stems (one per line)")
    ap.add_argument("--tfrecord_root", "--tfrecord-root", type=str, required=True, help="Waymo individual_files root")
    ap.add_argument("--maps_root", "--maps-root", type=str, required=True, help="maps-root passed to render_anchor_intensity_txt.py")
    ap.add_argument("--submaps_root", "--submaps-root", type=str, required=True, help="submaps-root passed to render_anchor_intensity_txt.py")
    ap.add_argument("--render_py", "--render-py", type=str, required=True, help="path to render_anchor_intensity_txt.py")

    ap.add_argument("--python", type=str, default=sys.executable, help="python interpreter to run render script")
    ap.add_argument(
        "--subsets",
        type=str,
        default="training,validation,testing",
        help="subset search order (comma-separated)",
    )
    ap.add_argument(
        "--lists_dir",
        type=str,
        default="",
        help="where to write generated subset list files (default: <submaps_root>/_lists)",
    )

    # --- 描画オプション（そのまま転送） ---
    ap.add_argument("--cam", type=str, default="FRONT")
    ap.add_argument("--int-mode", type=str, default="global_map")
    ap.add_argument("--int-percentile", type=float, default=99.5)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--range-corr-power", type=float, default=0.0)
    ap.add_argument("--crop-mode", type=str, default="none")
    ap.add_argument("--point-size", type=int, default=2)

    ap.add_argument("--write-index", type=str2bool, default=True)
    ap.add_argument("--post-hist-eq", type=str2bool, default=True)
    ap.add_argument("--post-clahe", type=str2bool, default=True)
    ap.add_argument("--clahe-clip-limit", type=float, default=2.0)
    ap.add_argument("--clahe-tile-grid", type=int, default=8)
    ap.add_argument("--post-mask-covered-only", type=str2bool, default=True)
    ap.add_argument("--overwrite", type=str2bool, default=False)

    args = ap.parse_args()

    segments_txt = Path(_normalize_wsl_path(args.segments_txt))
    tfrecord_root = Path(_normalize_wsl_path(args.tfrecord_root))
    maps_root = Path(_normalize_wsl_path(args.maps_root))
    submaps_root = Path(_normalize_wsl_path(args.submaps_root))
    render_py = Path(_normalize_wsl_path(args.render_py))

    if not segments_txt.exists():
        raise FileNotFoundError(f"segments_txt not found: {segments_txt}")
    if not tfrecord_root.exists():
        raise FileNotFoundError(f"tfrecord_root not found: {tfrecord_root}")
    if not maps_root.exists():
        raise FileNotFoundError(f"maps_root not found: {maps_root}")
    if not submaps_root.exists():
        raise FileNotFoundError(f"submaps_root not found: {submaps_root}")
    if not render_py.exists():
        raise FileNotFoundError(f"render_py not found: {render_py}")

    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    if not subsets:
        raise ValueError("--subsets is empty")

    lists_dir = Path(_normalize_wsl_path(args.lists_dir)) if args.lists_dir else (submaps_root / "_lists")
    lists_dir.mkdir(parents=True, exist_ok=True)

    segments = read_segments(segments_txt)
    print(f"[INFO] segments={len(segments)} from {segments_txt}")
    print(f"[INFO] tfrecord_root={tfrecord_root}")
    print(f"[INFO] maps_root={maps_root}")
    print(f"[INFO] submaps_root={submaps_root}")
    print(f"[INFO] lists_dir={lists_dir}")

    by_subset: Dict[str, List[str]] = {s: [] for s in subsets}
    unknown: List[str] = []

    for seg in segments:
        try:
            subset, _ = detect_subset(tfrecord_root, seg, subsets=subsets)
        except Exception as e:
            print(f"[WARN] subset detect failed: {seg} ({e})")
            unknown.append(seg)
            continue

        # 任意: この seg/subset に対応する submaps が存在するか確認する
        seg_submaps_dir = submaps_root / subset / seg
        if not seg_submaps_dir.exists():
            # make_submaps_* 系は submaps_root/<subset>/<segment> 以下へ書き出す。
            # 無ければ render スクリプトのクラッシュを避けるため skip する。
            print(f"[WARN] submaps dir missing, skip: {seg_submaps_dir}")
            continue

        by_subset[subset].append(seg)

    for subset, segs in list(by_subset.items()):
        if not segs:
            by_subset.pop(subset, None)

    if unknown:
        print(f"[WARN] segments not found in any subset: {len(unknown)}")

    if not by_subset:
        raise RuntimeError("No segments to process (after subset detection / submaps existence check)")

    # subset ごとに list file を書き出し、render スクリプトを呼ぶ。
    ok = 0
    fail = 0
    for subset, segs in by_subset.items():
        list_path = lists_dir / f"segments_{subset}.txt"
        list_path.write_text("\n".join(segs) + "\n", encoding="utf-8")
        print(f"[INFO] subset={subset} segments={len(segs)} list={list_path}")

        cmd = [
            args.python,
            str(render_py),
            "--tfrecord",
            str(list_path),
            "--subset",
            subset,
            "--tfrecord-root",
            str(tfrecord_root),
            "--maps-root",
            str(maps_root),
            "--submaps-root",
            str(submaps_root),
            "--cam",
            args.cam,
            "--int-mode",
            args.int_mode,
            "--int-percentile",
            str(args.int_percentile),
            "--gamma",
            str(args.gamma),
            "--range-corr-power",
            str(args.range_corr_power),
            "--crop-mode",
            args.crop_mode,
            "--point-size",
            str(args.point_size),
            "--write-index",
            str(args.write_index).lower(),
            "--post-hist-eq",
            str(args.post_hist_eq).lower(),
            "--post-clahe",
            str(args.post_clahe).lower(),
            "--clahe-clip-limit",
            str(args.clahe_clip_limit),
            "--clahe-tile-grid",
            str(args.clahe_tile_grid),
            "--post-mask-covered-only",
            str(args.post_mask_covered_only).lower(),
            "--overwrite",
            str(args.overwrite).lower(),
        ]

        rc = run_cmd(cmd)
        if rc == 0:
            ok += 1
        else:
            fail += 1
            print(f"[ERROR] render_anchor_intensity_txt.py failed (subset={subset}) rc={rc}")
            break

    print(f"[DONE] ok={ok} fail={fail}")


if __name__ == "__main__":
    main()
