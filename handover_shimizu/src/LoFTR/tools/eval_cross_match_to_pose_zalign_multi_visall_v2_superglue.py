#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 使い方:
#   python eval_cross_match_to_pose_zalign_multi_visall_v2_superglue.py --help
#   必要な入力パスと出力先を引数で指定して実行する。

"""
SuperGlue matcher を使って `eval_cross_match_to_pose_zalign_multi_visall_v2.py` を実行する。

このスクリプトは元の評価パイプライン（PnP / z-align / summary / CSV 出力）はそのまま使い、
差し替えるのは次の 2 つだけ:
  - load_matcher()
  - infer_matches()



"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
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


def _normalize_wsl_path(p: str) -> str:
    """WSL と Windows が混在するパス表記を読める形に正規化する。"""
    p = (p or "").strip()
    if len(p) >= 3 and p[1] == ":" and p[2] in ["\\", "/"]:
        drive = p[0].lower()
        rest = p[2:].lstrip("\\/").replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return p


def _load_module_from_path(module_name: str, py_path: Path):
    """任意パスの Python モジュールを動的に読み込む。"""
    spec = importlib.util.spec_from_file_location(module_name, str(py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve_superglue_repo(path_str: str) -> Path:
    """SuperGlue リポジトリの場所を解決する。"""
    p = Path(_normalize_wsl_path(path_str)).expanduser()
    if p.exists() and (p / "models" / "matching.py").exists():
        return p.resolve()

    if p.exists() and p.is_dir():
        cands = []
        for ch in p.iterdir():
            if ch.is_dir() and (ch / "models" / "matching.py").exists():
                cands.append(ch)
        if len(cands) == 1:
            return cands[0].resolve()
        if len(cands) > 1:
            names = ", ".join(str(x) for x in cands[:8])
            raise FileNotFoundError(
                f"multiple SuperGlue-like repos under {p}. please pass exact repo path. candidates={names}"
            )

    raise FileNotFoundError(
        f"SuperGlue repo not found. expected <repo>/models/matching.py under: {p}"
    )


class _SuperGlueWrapper:
    """SuperGlue 推論器を既存 evaluator から呼びやすく包む。"""
    def __init__(self, matching: Any):
        """インスタンス生成時に必要な設定値と内部状態を初期化する。"""
        self.matching = matching


def main() -> None:
    """CLI 引数を解釈し、入力の読み込みから結果保存までの一連の処理を実行する。"""
    this_py = Path(__file__).resolve()
    loftr_root = this_py.parents[1]
    default_eval_script = loftr_root / "tools" / "eval_cross_match_to_pose_zalign_multi_visall_v2.py"
    loftr_src = (loftr_root / "src").resolve()
    if str(loftr_src) not in sys.path:
        sys.path.insert(0, str(loftr_src))

    ap = argparse.ArgumentParser(
        description="Wrapper: run cross pose eval with SuperGlue instead of LoFTR/SFPPR matcher."
    )
    ap.add_argument(
        "--superglue-repo",
        type=str,
        required=True,
        help="Path to SuperGluePretrainedNetwork repo root (or parent dir that contains it).",
    )
    ap.add_argument("--eval-script", type=str, default=str(default_eval_script))

    # SuperPoint/SuperGlue の設定
    ap.add_argument("--superpoint-nms-radius", type=int, default=4)
    ap.add_argument("--superpoint-keypoint-threshold", type=float, default=0.005)
    ap.add_argument("--superpoint-max-keypoints", type=int, default=1024)
    ap.add_argument("--superglue-weights", type=str, default="outdoor", choices=["outdoor", "indoor"])
    ap.add_argument("--superglue-sinkhorn-iterations", type=int, default=20)
    ap.add_argument("--superglue-match-threshold", type=float, default=0.2)

    ap.add_argument("--dry-run", type=_str2bool, default=False)

    args, eval_args = ap.parse_known_args()
    eval_args = [x for x in eval_args if x != "--"]

    eval_script = Path(_normalize_wsl_path(args.eval_script)).expanduser().resolve()
    if not eval_script.exists():
        raise FileNotFoundError(f"eval script not found: {eval_script}")

    sg_repo = _resolve_superglue_repo(args.superglue_repo)
    sys.path.insert(0, str(sg_repo))
    from models.matching import Matching  # type: ignore

    if "--ckpt" not in eval_args:
        eval_args = ["--ckpt", "superglue://builtin"] + eval_args

    # evaluator モジュールを読み込み、matcher フックを差し替える。
    mod = _load_module_from_path("eval_cross_pose_base_superglue", eval_script)

    sg_cfg: Dict[str, Any] = {
        "superpoint": {
            "nms_radius": int(args.superpoint_nms_radius),
            "keypoint_threshold": float(args.superpoint_keypoint_threshold),
            "max_keypoints": int(args.superpoint_max_keypoints),
        },
        "superglue": {
            "weights": str(args.superglue_weights),
            "sinkhorn_iterations": int(args.superglue_sinkhorn_iterations),
            "match_threshold": float(args.superglue_match_threshold),
        },
    }

    def _patched_load_matcher(
        ckpt_path: str,
        device: torch.device,
        enable_fine: bool,
        enable_repeatability: bool = False,
    ) -> _SuperGlueWrapper:
        """既存 evaluator の matcher 読み込み処理を差し替える。"""
        _ = ckpt_path
        _ = enable_fine
        _ = enable_repeatability
        matcher = Matching(sg_cfg).eval().to(device)
        print(
            "[INFO] SuperGlue matcher loaded "
            f"(weights={args.superglue_weights}, max_kpts={args.superpoint_max_keypoints}, "
            f"match_th={args.superglue_match_threshold})"
        )
        return _SuperGlueWrapper(matcher)

    def _patched_infer_matches(
        matcher: _SuperGlueWrapper,
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
        """既存 evaluator の対応点推論処理を差し替える。"""
        _ = stride
        _ = cell_point
        _ = amp_enabled
        _ = amp_dtype
        _ = match_source

        H0o, W0o = img0_u8.shape[:2]
        H1o, W1o = img1_u8.shape[:2]

        # base evaluator と同じ resize 方針を維持する。
        img0_r, sx0, sy0 = mod._resize_long_keep_aspect(img0_u8, resize_long, divisor=8)
        img1_r, sx1, sy1 = mod._resize_long_keep_aspect(img1_u8, resize_long, divisor=8)
        H0, W0 = img0_r.shape[:2]
        H1, W1 = img1_r.shape[:2]

        t0 = torch.from_numpy(img0_r).float().to(device) / 255.0
        t1 = torch.from_numpy(img1_r).float().to(device) / 255.0
        data = {
            "image0": t0[None, None],
            "image1": t1[None, None],
        }

        with torch.inference_mode():
            pred = matcher.matching(data)

        kpts0 = pred["keypoints0"][0].detach().cpu().numpy().astype(np.float32, copy=False)
        kpts1 = pred["keypoints1"][0].detach().cpu().numpy().astype(np.float32, copy=False)
        m0 = pred["matches0"][0].detach().cpu().numpy().astype(np.int64, copy=False)
        s0 = pred["matching_scores0"][0].detach().cpu().numpy().astype(np.float32, copy=False)

        valid = m0 >= 0
        if np.any(valid):
            idx0 = np.where(valid)[0]
            idx1 = m0[valid]
            mk0 = kpts0[idx0]
            mk1 = kpts1[idx1]
            conf = s0[valid]
        else:
            mk0 = np.zeros((0, 2), dtype=np.float32)
            mk1 = np.zeros((0, 2), dtype=np.float32)
            conf = np.zeros((0,), dtype=np.float32)

        if conf_th > 0.0 and conf.size == mk0.shape[0]:
            keep = conf >= float(conf_th)
            mk0 = mk0[keep]
            mk1 = mk1[keep]
            conf = conf[keep]

        if max_matches is not None and int(max_matches) > 0 and mk0.shape[0] > int(max_matches):
            k = int(max_matches)
            idx = np.argsort(-conf)[:k]
            mk0 = mk0[idx]
            mk1 = mk1[idx]
            conf = conf[idx]

        # 元の解像度へ戻す
        mk0[:, 0] /= float(sx0)
        mk0[:, 1] /= float(sy0)
        mk1[:, 0] /= float(sx1)
        mk1[:, 1] /= float(sy1)

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
            },
        }

    mod.load_matcher = _patched_load_matcher
    mod.infer_matches = _patched_infer_matches

    cmd_preview = "python " + str(eval_script) + " " + " ".join(eval_args)
    print("[INFO] superglue_repo:", sg_repo)
    print("[RUN] ", cmd_preview)

    if bool(args.dry_run):
        return

    # 引数を引き継いで元の main() を呼ぶ。
    old_argv = list(sys.argv)
    try:
        sys.argv = [str(eval_script)] + eval_args
        mod.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
