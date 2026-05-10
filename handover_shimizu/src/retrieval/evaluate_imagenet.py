#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 使い方:
#   python evaluate_imagenet.py --help
#   必要な入力パスと出力先を引数で指定して実行する。


"""
ImageNet 初期化済み backbone を使って評価する（学習済みチェックポイントは使わない）。

このラッパーは evaluate.py の評価パイプライン全体を再利用しつつ、
モデル構築だけを差し替えて以下の状態にする:
- ResNet は ImageNet 事前学習重みで初期化する。
- NetVLAD / projection MLP はランダム初期化のままにする。

"""

from __future__ import annotations

import argparse
import random
import sys
from typing import Optional

import numpy as np

_WRAPPER_SEED = 0


def _set_seed(seed: int) -> None:
    """乱数シードを固定して再現性を高める。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _build_resnet_imagenet(backbone: str):
    """ImageNet 初期化済み ResNet バックボーンを構築する。"""
    from torchvision import models

    bb = str(backbone).lower().strip()
    if bb not in {"resnet18", "resnet34", "resnet50"}:
        raise ValueError(f"unsupported backbone: {bb}")

    if bb == "resnet18":
        try:
            return models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            return models.resnet18(pretrained=True)
    if bb == "resnet34":
        try:
            return models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        except Exception:
            return models.resnet34(pretrained=True)
    try:
        return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    except Exception:
        return models.resnet50(pretrained=True)


def _load_model_imagenet(args):
    """評価用の ImageNet 初期化モデルを読み込む。"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _set_seed(_WRAPPER_SEED)

    class NetVLAD(nn.Module):
        """局所特徴を集約してグローバル記述子を作る NetVLAD 層。"""
        def __init__(
            self,
            num_clusters: int = 32,
            dim: int = 512,
            alpha: float = 1.0,
            normalize_input: bool = True,
            intra_norm: bool = True,
        ):
            """インスタンス生成時に必要な設定値と内部状態を初期化する。"""
            super().__init__()
            self.K = int(num_clusters)
            self.D = int(dim)
            self.alpha = float(alpha)
            self.normalize_input = bool(normalize_input)
            self.intra_norm = bool(intra_norm)

            self.conv = nn.Conv2d(self.D, self.K, kernel_size=(1, 1), bias=True)
            self.centroids = nn.Parameter(torch.rand(self.K, self.D))
            self._init_params()

        def _init_params(self) -> None:
            """層パラメータを初期化する。"""
            with torch.no_grad():
                self.conv.weight.copy_((2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1))
                self.conv.bias.copy_(-self.alpha * self.centroids.norm(dim=1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """順伝播を実行して必要な特徴量または損失を計算する。"""
            b, c, _, _ = x.shape
            if self.normalize_input:
                x = F.normalize(x, p=2, dim=1)

            soft = self.conv(x).view(b, self.K, -1)
            soft = F.softmax(soft, dim=1)
            x_flat = x.view(b, c, -1)

            vlad = torch.zeros(b, self.K, c, device=x.device, dtype=x.dtype)
            for k in range(self.K):
                a = soft[:, k, :].unsqueeze(1)
                residual = x_flat - self.centroids[k].view(1, c, 1)
                vlad[:, k, :] = (a * residual).sum(dim=2)

            if self.intra_norm:
                vlad = F.normalize(vlad, p=2, dim=2)
            vlad = vlad.view(b, -1)
            return F.normalize(vlad, p=2, dim=1)

    class ResNetVLAD(nn.Module):
        """ResNet バックボーンと NetVLAD を組み合わせた埋め込み抽出器。"""
        def __init__(self, backbone: str = "resnet18", out_dim: int = 256, clusters: int = 32, freeze_stages: int = 2):
            """インスタンス生成時に必要な設定値と内部状態を初期化する。"""
            super().__init__()
            net = _build_resnet_imagenet(backbone)

            feat_dim = 2048 if backbone == "resnet50" else 512
            self.conv1, self.bn1, self.relu, self.maxpool = net.conv1, net.bn1, net.relu, net.maxpool
            self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4

            if feat_dim != 512:
                self.reduce = nn.Conv2d(feat_dim, 512, kernel_size=1, bias=False)
                feat_dim = 512
            else:
                self.reduce = None

            self._freeze_stages_base_compatible(int(freeze_stages))

            self.vlad = NetVLAD(num_clusters=int(clusters), dim=int(feat_dim), alpha=1.0, normalize_input=True, intra_norm=True)
            self.proj = nn.Sequential(
                nn.Linear(int(clusters) * int(feat_dim), 1024),
                nn.BatchNorm1d(1024),
                nn.GELU(),
                nn.Linear(1024, int(out_dim)),
            )

        def _freeze_stages_base_compatible(self, k: int) -> None:
            """互換性を保ちながら backbone の一部ステージを凍結する。"""
            names_to_freeze = []
            if k >= 1:
                names_to_freeze += ["conv1", "bn1"]
            if k >= 2:
                names_to_freeze += ["layer1"]
            if k >= 3:
                names_to_freeze += ["layer2"]

            for n, p in self.named_parameters():
                if any(n.startswith(prefix) for prefix in names_to_freeze):
                    p.requires_grad = False

            for m in self.modules():
                cls = m.__class__.__name__.lower()
                if "batchnorm" in cls:
                    m.eval()

            if hasattr(self, "layer3"):
                self.layer3.train()
                for p in self.layer3.parameters():
                    p.requires_grad = True
            if hasattr(self, "layer4"):
                self.layer4.train()
                for p in self.layer4.parameters():
                    p.requires_grad = True

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """順伝播を実行して必要な特徴量または損失を計算する。"""
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            if self.reduce is not None:
                x = self.reduce(x)
            v = self.vlad(x)
            e = self.proj(v)
            return F.normalize(e, p=2, dim=1)

    class DualEncoder(nn.Module):
        """カメラとアンカーの 2 系統エンコーダを束ねるモデル。"""
        def __init__(self, backbone: str = "resnet18", out_dim: int = 256, clusters: int = 32, freeze_stages: int = 2):
            """インスタンス生成時に必要な設定値と内部状態を初期化する。"""
            super().__init__()
            self.cam = ResNetVLAD(backbone, out_dim, clusters, freeze_stages)
            self.int = ResNetVLAD(backbone, out_dim, clusters, freeze_stages)

        def forward(self, cam_img: torch.Tensor, anc_img: torch.Tensor):
            """順伝播を実行して必要な特徴量または損失を計算する。"""
            return self.cam(cam_img), self.int(anc_img)

    model = DualEncoder(
        backbone=str(args.backbone),
        out_dim=int(args.embed_dim),
        clusters=int(args.clusters),
        freeze_stages=int(args.freeze_stages),
    )
    model.eval()
    return model


def _extract_value(argv: list[str], key: str, default: Optional[str] = None) -> Optional[str]:
    """辞書やログ構造から目的の値を安全に取り出す。"""
    if key in argv:
        i = argv.index(key)
        if i + 1 < len(argv):
            return argv[i + 1]
    return default


def main() -> None:
    """CLI 引数を解釈し、入力の読み込みから結果保存までの一連の処理を実行する。"""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--seed", type=int, default=0, help="Seed for random init parts (NetVLAD/MLP).")
    p.add_argument("--help", action="store_true")
    known, rest = p.parse_known_args()

    if known.help:
        import evaluate as base_eval

        old_argv = sys.argv[:]
        try:
            sys.argv = [old_argv[0], "--help"]
            base_eval.main()
        finally:
            sys.argv = old_argv
        return

    if "--checkpoint" in rest:
        raise SystemExit("Do not pass --checkpoint to evaluate_imagenet.py (it is checkpoint-free).")

    global _WRAPPER_SEED
    _WRAPPER_SEED = int(known.seed)

    import evaluate as base_eval

    base_eval._load_model = _load_model_imagenet

    backbone = _extract_value(rest, "--backbone", "resnet34")
    print(
        f"[INFO] evaluate_imagenet: using ImageNet init backbone={backbone}, "
        f"random NetVLAD/MLP seed={_WRAPPER_SEED}"
    )

    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0], "--checkpoint", "__imagenet_init__", *rest]
        base_eval.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
