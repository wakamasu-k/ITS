# uncompyle6 version 3.9.3
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.10.12 (main, Mar  3 2026, 11:56:32) [GCC 11.4.0]
# Embedded file name: /home/ubuntu/waymo/datasets/LoFTR/tools/train_sfppr_coarse_rep_joint.py
# Compiled at: 2025-11-21 05:07:50
# Size of source mod 2**32: 9990 bytes
import argparse, os
from typing import Tuple
import torch
import torch.nn.functional as F
from tqdm import tqdm
from loftr import default_cfg
from loftr.loftr_sfppr import SFPPRLoFTR
from loftr.datasets.waymo_sfppr_dataset import WaymoSFPPRDataset

def build_coarse_gt_indices_from_sample(sample: dict, batch: dict) -> Tuple[(torch.Tensor, torch.Tensor)]:
    """
    mkpts0 / mkpts1（fine 座標）から coarse グリッド上の index を作る。
    - sample["mkpts0"], sample["mkpts1"] : [N, 2] (x, y) in resized image coordinates
    - batch["imsize0"], batch["imsize1"] : [1, 2] (H, W)
    - batch["mask0_coarse"]             : [1, Hc, Wc]  (coverage mask, 0/1)
    戻り値:
      idx0_flat, idx1_flat : [Ng]  (Ng は valid な GT の数)
    """
    device = batch["image0"].device
    mkpts0 = sample["mkpts0"].to(device)
    mkpts1 = sample["mkpts1"].to(device)
    imsize0 = batch["imsize0"][0]
    imsize1 = batch["imsize1"][0]
    H0, W0 = int(imsize0[0].item()), int(imsize0[1].item())
    H1, W1 = int(imsize1[0].item()), int(imsize1[1].item())
    mask0_coarse = batch["mask0_coarse"]
    Hc0, Wc0 = mask0_coarse.shape[-2], mask0_coarse.shape[-1]
    Hc1, Wc1 = Hc0, Wc0
    scale_y0 = Hc0 / float(H0)
    scale_x0 = Wc0 / float(W0)
    scale_y1 = Hc1 / float(H1)
    scale_x1 = Wc1 / float(W1)
    x0 = mkpts0[:, 0].clamp(0, W0 - 1)
    y0 = mkpts0[:, 1].clamp(0, H0 - 1)
    x1 = mkpts1[:, 0].clamp(0, W1 - 1)
    y1 = mkpts1[:, 1].clamp(0, H1 - 1)
    i0 = (y0 * scale_y0).long()
    j0 = (x0 * scale_x0).long()
    i1 = (y1 * scale_y1).long()
    j1 = (x1 * scale_x1).long()
    valid0 = (i0 >= 0) & (i0 < Hc0) & (j0 >= 0) & (j0 < Wc0)
    valid1 = (i1 >= 0) & (i1 < Hc1) & (j1 >= 0) & (j1 < Wc1)
    valid = valid0 & valid1
    if valid.sum() == 0:
        return (torch.empty(0, dtype=(torch.long), device=device),
         torch.empty(0, dtype=(torch.long), device=device))
    i0 = i0[valid]
    j0 = j0[valid]
    i1 = i1[valid]
    j1 = j1[valid]
    mask0_flat = mask0_coarse.view(-1)
    idx0_flat_all = i0 * Wc0 + j0
    inside_mask = mask0_flat[idx0_flat_all] > 0.5
    if inside_mask.sum() == 0:
        return (torch.empty(0, dtype=(torch.long), device=device),
         torch.empty(0, dtype=(torch.long), device=device))
    idx0_flat = idx0_flat_all[inside_mask]
    idx1_flat = (i1 * Wc1 + j1)[inside_mask]
    return (
     idx0_flat, idx1_flat)


def compute_coarse_loss(batch: dict, sample: dict) -> torch.Tensor:
    """
    conf_matrix（LoFTR coarse matching の softmax 後スコア）に対する
    SFPPR 風 coarse 損失を計算する。
    """
    device = batch["image0"].device
    conf = batch["conf_matrix"]
    B, N0, N1 = conf.shape
    assert B == 1, "現状 batch_size=1 前提で実装している"
    idx0_flat, idx1_flat = build_coarse_gt_indices_from_sample(sample, batch)
    if idx0_flat.numel() == 0:
        return torch.tensor(0.0, device=device)
    conf_gt = conf[(0, idx0_flat, idx1_flat)]
    eps = 1e-08
    loss = -torch.log(conf_gt + eps).mean()
    return loss


def compute_repeatability_loss(batch: dict) -> torch.Tensor:
    """
    repeatability0（logits）と coverage mask（0/1）から BCE 損失を計算する。
    """
    device = batch["image0"].device
    rep_logits = batch["repeatability0"]
    mask0_coarse = batch["mask0_coarse"]
    B, Hc, Wc = mask0_coarse.shape
    target = mask0_coarse.view(B, Hc * Wc).to(device)
    rep_probs = torch.sigmoid(rep_logits)
    loss = F.binary_cross_entropy(rep_probs, target)
    return loss


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


