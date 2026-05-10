#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 使い方:
#   python make_retrieval_pairs_parquet.py --help
#   必要な入力パスと出力先を引数で指定して実行する。


"""
make_retrieval_pairs_parquet.py

make_retrieval_shifted_ri.py の出力（/mnt/e/waymo/data/retrieval_pairs）を走査して、
diff_train_contrastive_vlad_one.py が読める形式の pairs.parquet を作成する。

重要:
- InfoNCE(in-batch negatives) なので、「同じ cam_png を複数行に増やす」と
  同一バッチ内重複が起きやすくなり、矛盾シグナル（破綻）の原因になる。
- したがって本スクリプトは基本的に「1 cam_png につき 1行」になるように作る。
  （重複があれば drop_duplicates で削除）

入力ディレクトリ構造（例）:
  /mnt/e/waymo/data/retrieval_pairs/training/<segment>/<submap_id>/
    cam.json              (cam_image_path を含む)
    ri_000.png, ri_001.png, ...

出力 Parquet（列）:
  cam_png, anchor_png, label, segment_id, submap_id, anchor_frame_index, anchor_choice
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd


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


def read_segment_list(path: str) -> Optional[set]:
    """セグメント一覧テキストを読み込む。"""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    segs = set()
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                segs.add(s)
    return segs


def load_json(path: Path) -> Dict[str, Any]:
    """JSON ファイルを読み込んで Python オブジェクトとして返す。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_anchor_png(anchor_dir: Path, ri_pngs: List[Path], mode: str, rng: random.Random) -> Optional[Path]:
    """候補の中から利用するアンカー画像パスを選ぶ。"""
    if not ri_pngs:
        return None

    ri000 = anchor_dir / "ri_000.png"
    if mode == "ri000":
        if ri000.exists():
            return ri000
        return ri_pngs[0]

    if mode == "random":
        return rng.choice(ri_pngs)

    if mode == "random_no_000":
        cand = [p for p in ri_pngs if p.name != "ri_000.png"]
        if cand:
            return rng.choice(cand)
        return rng.choice(ri_pngs)

    raise ValueError(f"unknown mode: {mode}")


def main():
    """CLI 引数を解釈し、入力の読み込みから結果保存までの一連の処理を実行する。"""
    ap = argparse.ArgumentParser(description="retrieval_pairs -> pairs.parquet generator")

    ap.add_argument("--pairs-root", type=str, default="/mnt/e/waymo/data/retrieval_pairs",
                    help="make_retrieval_shifted_ri.py の出力ルート")
    ap.add_argument("--subset", type=str, required=True, choices=["training", "validation", "testing"],
                    help="対象subset")
    ap.add_argument("--out-parquet", type=str, required=True, help="出力parquetパス")
    ap.add_argument("--out-summary-json", type=str, default="", help="要約JSONの出力先（任意）")

    ap.add_argument("--segment-list", type=str, default="",
                    help="対象segment名のテキスト（任意、1行=segmentディレクトリ名）")

    ap.add_argument("--anchor-select", type=str, default="random",
                    choices=["ri000", "random", "random_no_000"],
                    help="anchor_png に使う ri の選び方")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--min-ri", type=int, default=1, help="各アンカーdirに必要なri_*.png最低枚数")
    ap.add_argument("--require-cam-exists", type=str2bool, default=True, help="cam画像が無い行は捨てる")
    ap.add_argument("--require-ri000", type=str2bool, default=False, help="ri_000.png が無いアンカーdirは捨てる")

    ap.add_argument("--drop-dup-cam", type=str2bool, default=True, help="同一cam_pngが複数行なら重複削除")
    ap.add_argument("--max-rows", type=int, default=-1, help="デバッグ用に最大行数で打ち切り（-1で無制限）")

    args = ap.parse_args()

    rng = random.Random(args.seed)
    allow = read_segment_list(args.segment_list)

    root = Path(args.pairs_root) / args.subset
    if not root.exists():
        raise FileNotFoundError(str(root))

    rows = []
    n_anchor_dirs = 0
    n_skipped_no_cam = 0
    n_skipped_no_ri = 0
    n_skipped_no_ri000 = 0

    seg_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    for seg_dir in seg_dirs:
        segment_id = seg_dir.name
        if allow is not None and segment_id not in allow:
            continue

        anchor_dirs = sorted([p for p in seg_dir.iterdir() if p.is_dir()])
        for anchor_dir in anchor_dirs:
            n_anchor_dirs += 1
            submap_id = anchor_dir.name

            cam_json = anchor_dir / "cam.json"
            if not cam_json.exists():
                continue
            cam_meta = load_json(cam_json)

            cam_png = cam_meta.get("cam_image_path", "")
            if not cam_png:
                n_skipped_no_cam += 1
                continue
            cam_png_p = Path(cam_png)
            if args.require_cam_exists and (not cam_png_p.exists()):
                n_skipped_no_cam += 1
                continue

            ri_pngs = sorted(anchor_dir.glob("ri_*.png"))
            if len(ri_pngs) < args.min_ri:
                n_skipped_no_ri += 1
                continue

            if args.require_ri000 and (not (anchor_dir / "ri_000.png").exists()):
                n_skipped_no_ri000 += 1
                continue

            anchor_png_p = pick_anchor_png(anchor_dir, ri_pngs, args.anchor_select, rng)
            if anchor_png_p is None or (not anchor_png_p.exists()):
                n_skipped_no_ri += 1
                continue

            row = {
                "cam_png": str(cam_png_p),
                "anchor_png": str(anchor_png_p),
                "label": 1,
                "segment_id": segment_id,
                "submap_id": submap_id,
                "anchor_frame_index": int(cam_meta.get("anchor_frame_index", -1)),
                "anchor_choice": args.anchor_select,
            }
            rows.append(row)

            if args.max_rows > 0 and len(rows) >= args.max_rows:
                break

        if args.max_rows > 0 and len(rows) >= args.max_rows:
            break

    if not rows:
        raise RuntimeError("no rows collected. check paths / subset / segment-list.")

    df = pd.DataFrame(rows)

    dup_before = int(df.duplicated(subset=["cam_png"]).sum())
    if args.drop_dup_cam:
        df = df.drop_duplicates(subset=["cam_png"], keep="first").reset_index(drop=True)

    out_parquet = Path(args.out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)

    summary = {
        "subset": args.subset,
        "pairs_root": str(root),
        "out_parquet": str(out_parquet),
        "num_rows": int(len(df)),
        "num_anchor_dirs_seen": int(n_anchor_dirs),
        "skipped_no_cam": int(n_skipped_no_cam),
        "skipped_no_ri": int(n_skipped_no_ri),
        "skipped_no_ri000": int(n_skipped_no_ri000),
        "dup_cam_before_drop": int(dup_before),
        "anchor_select": args.anchor_select,
        "seed": int(args.seed),
    }

    print("[OK] wrote parquet:", out_parquet)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.out_summary_json:
        out_sum = Path(args.out_summary_json)
        out_sum.parent.mkdir(parents=True, exist_ok=True)
        with open(out_sum, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("[OK] wrote summary:", out_sum)


if __name__ == "__main__":
    main()


