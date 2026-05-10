#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 使い方:
#   python eval_cross_match_to_pose_zalign_multi_visall_v2_loftr.py --help
#   必要な入力パスと出力先を引数で指定して実行する。

"""
通常の LoFTR チェックポイントを使って `eval_cross_match_to_pose_zalign_multi_visall_v2.py` を実行する。

このラッパーは元の evaluator 本体を変更しない。
single-backbone の LoFTR チェックポイントを dual-backbone 互換
（SFPPR 形式の backbone0/backbone1）へ変換してから、
`tools/eval_cross_match_to_pose_zalign_multi_visall_v2.py` を起動する。


"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch


def _str2bool(v: str | bool) -> bool:
    """CLI などで受け取った真偽値表現を bool に正規化する。"""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {v}")


def _looks_like_dual_backbone_keys(keys: Iterable[str]) -> bool:
    """重みキーが dual-backbone 形式かどうかを判定する。"""
    has0 = any(k.startswith("backbone0.") or ".backbone0." in k for k in keys)
    has1 = any(k.startswith("backbone1.") or ".backbone1." in k for k in keys)
    return bool(has0 and has1)


def _load_state_dict_any(ckpt_path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """各種 ckpt 形式から state_dict を取り出す。"""
    ckpt_obj = torch.load(str(ckpt_path), map_location="cpu")

    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
        sd = ckpt_obj["state_dict"]
    elif isinstance(ckpt_obj, dict) and "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        sd = ckpt_obj["model"]
    elif isinstance(ckpt_obj, dict):
        tensor_like = {k: v for k, v in ckpt_obj.items() if torch.is_tensor(v)}
        if not tensor_like:
            raise ValueError(f"Unsupported checkpoint format (dict without tensor-like entries): {ckpt_path}")
        sd = tensor_like
    else:
        raise ValueError(f"Unsupported checkpoint format type: {type(ckpt_obj)}")

    out = OrderedDict()
    for k, v in sd.items():
        kk = str(k)
        if kk.startswith("matcher."):
            kk = kk[len("matcher."):]
        out[kk] = v

    meta: Dict[str, Any] = {}
    if isinstance(ckpt_obj, dict):
        if "config" in ckpt_obj:
            meta["config"] = ckpt_obj["config"]
        if "cfg" in ckpt_obj and "config" not in meta:
            meta["config"] = ckpt_obj["cfg"]
    return out, meta


def _convert_single_to_dual_backbone(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """single-backbone 重みを dual-backbone 形式へ変換する。"""
    out: Dict[str, torch.Tensor] = OrderedDict()
    for k, v in sd.items():
        if k.startswith("backbone0.") or k.startswith("backbone1.") or ".backbone0." in k or ".backbone1." in k:
            out[k] = v
            continue
        if k.startswith("backbone."):
            suffix = k[len("backbone."):]
            out[f"backbone0.{suffix}"] = v
            out[f"backbone1.{suffix}"] = v
            continue
        if ".backbone." in k:
            out[k.replace(".backbone.", ".backbone0.")] = v
            out[k.replace(".backbone.", ".backbone1.")] = v
            continue
        out[k] = v
    return out


def _prepare_eval_ckpt(
    loftr_ckpt: Path,
    out_ckpt: Path,
    force_reconvert: bool,
) -> Path:
    """評価前に ckpt 形式を整えて読み込める状態にする。"""
    if out_ckpt.exists() and not force_reconvert:
        print(f"[INFO] reuse converted ckpt: {out_ckpt}")
        return out_ckpt

    sd, meta = _load_state_dict_any(loftr_ckpt)
    is_dual = _looks_like_dual_backbone_keys(sd.keys())
    sd_out = sd if is_dual else _convert_single_to_dual_backbone(sd)

    out_obj: Dict[str, Any] = {
        "state_dict": sd_out,
        "note": (
            "auto-converted for eval_cross_match_to_pose_zalign_multi_visall_v2.py "
            "(single backbone duplicated into backbone0/backbone1 when needed)"
        ),
        "source_loftr_ckpt": str(loftr_ckpt),
        "source_was_dual": bool(is_dual),
    }
    if "config" in meta:
        out_obj["config"] = meta["config"]

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_obj, str(out_ckpt))
    print(f"[OK] converted ckpt written: {out_ckpt}")
    return out_ckpt


def _with_repo_pythonpath(env: Dict[str, str], loftr_root: Path) -> Dict[str, str]:
    """外部リポジトリを import できるよう PYTHONPATH を拡張する。"""
    src_dir = str((loftr_root / "src").resolve())
    cur = env.get("PYTHONPATH", "").strip()
    if not cur:
        env["PYTHONPATH"] = src_dir
        return env
    parts = [p for p in cur.split(":") if p]
    if src_dir not in parts:
        env["PYTHONPATH"] = src_dir + ":" + cur
    return env


def main() -> None:
    """CLI 引数を解釈し、入力の読み込みから結果保存までの一連の処理を実行する。"""
    this_py = Path(__file__).resolve()
    loftr_root = this_py.parents[1]
    default_loftr_ckpt = loftr_root / "weights" / "mega_depth" / "outdoor_ds.ckpt"
    default_out_ckpt = loftr_root / "weights" / ".converted_for_eval" / "outdoor_ds__dual_for_eval.pth"
    default_eval_script = loftr_root / "tools" / "eval_cross_match_to_pose_zalign_multi_visall_v2.py"

    ap = argparse.ArgumentParser(
        description="Wrapper: use vanilla LoFTR ckpt for eval_cross_match_to_pose_zalign_multi_visall_v2.py"
    )
    ap.add_argument("--loftr-ckpt", type=str, default=str(default_loftr_ckpt), help="Input LoFTR ckpt (single or dual)")
    ap.add_argument("--converted-ckpt", type=str, default=str(default_out_ckpt), help="Path to write/read converted ckpt")
    ap.add_argument("--eval-script", type=str, default=str(default_eval_script), help="Evaluator script path")
    ap.add_argument("--python", type=str, default=sys.executable, help="Python executable for launching evaluator")
    ap.add_argument("--force-reconvert", type=_str2bool, default=False, help="Rebuild converted ckpt even if exists")
    ap.add_argument("--dry-run", type=_str2bool, default=False, help="Print command and exit")
    args, eval_args = ap.parse_known_args()
    # ユーザーが `--` で下流引数を区切ると、argparse はそれを unknown に残す。
    # 区切りトークンだけを除去し、実際の evaluator 引数は保持する。
    eval_args = [x for x in eval_args if x != "--"]

    if any(x == "--ckpt" for x in eval_args):
        raise ValueError("Do not pass --ckpt to this wrapper. Use --loftr-ckpt instead.")

    loftr_ckpt = Path(args.loftr_ckpt).expanduser().resolve()
    converted_ckpt = Path(args.converted_ckpt).expanduser().resolve()
    eval_script = Path(args.eval_script).expanduser().resolve()

    if not loftr_ckpt.exists():
        raise FileNotFoundError(f"loftr ckpt not found: {loftr_ckpt}")
    if not eval_script.exists():
        raise FileNotFoundError(f"eval script not found: {eval_script}")

    ckpt_for_eval = _prepare_eval_ckpt(
        loftr_ckpt=loftr_ckpt,
        out_ckpt=converted_ckpt,
        force_reconvert=bool(args.force_reconvert),
    )

    cmd = [args.python, str(eval_script), "--ckpt", str(ckpt_for_eval)] + eval_args
    print("[RUN] " + " ".join(shlex.quote(c) for c in cmd))

    if bool(args.dry_run):
        return

    env = _with_repo_pythonpath(dict(os.environ), loftr_root=loftr_root)
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
