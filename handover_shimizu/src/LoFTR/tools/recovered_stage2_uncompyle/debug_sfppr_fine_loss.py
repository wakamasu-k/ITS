# uncompyle6 version 3.9.3
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.10.12 (main, Mar  3 2026, 11:56:32) [GCC 11.4.0]
# Embedded file name: /home/ubuntu/waymo/datasets/LoFTR/tools/debug_sfppr_fine_loss.py
# Compiled at: 2025-11-26 02:16:07
# Size of source mod 2**32: 10845 bytes
import argparse
from typing import Dict, Any, Tuple
import torch
import torch.nn.functional as F
from loftr import default_cfg
from loftr.loftr_sfppr import SFPPRLoFTR
from loftr.datasets.waymo_sfppr_dataset import WaymoSFPPRDataset

def compute_fine_loss_from_batch(batch: Dict[(str, Any)], max_gt_points: int=4096, eps: float=1e-06) -> Tuple[(torch.Tensor, Dict[(str, Any)])]:
    """
    SFPPR 論文の Eq.(7) を模した fine-level 損失 L_f を計算する（近似版）。

      L_f ≈ 1/|M_f| Σ_{pred} ( 1 / τ^2 ) * || j_pred - j_gt ||^2

    ここでは:
      - mkpts0_f, mkpts1_f: LoFTR fine stage の予測 (pixel 座標)
      - expec_f[:, 2]: fine ヒートマップの std (正規化座標系) → τ とみなす
      - mkpts0_gt, mkpts1_gt: pseudo GT (pixel 座標)

    実装上の近似:
      - LiDAR 側 mkpts0_f から mkpts0_gt への最近傍をとり、その index の mkpts1_gt を
        カメラ側 GT j_gt とみなす。
      - coarse GT mask (gt_mask) がある場合は、それで valid な fine マッチを絞る。
        ただし、gt_mask が全て False の場合は「警告を出して全マッチを使用」する。
    """
    device = batch["image0"].device
    mkpts0_gt = batch["mkpts0"][0]
    mkpts1_gt = batch["mkpts1"][0]
    n_gt = mkpts0_gt.shape[0]
    if n_gt > max_gt_points:
        perm = torch.randperm(n_gt, device=device)[:max_gt_points]
        mkpts0_gt = mkpts0_gt[perm]
        mkpts1_gt = mkpts1_gt[perm]
        n_gt = mkpts0_gt.shape[0]
    else:
        mkpts0_f = batch["mkpts0_f"]
        mkpts1_f = batch["mkpts1_f"]
        expec_f = batch["expec_f"]
        M = mkpts0_f.shape[0]
        if M == 0 or n_gt == 0:
            loss = mkpts0_f.sum() * 0.0
            stats = {'num_pred':int(M), 
             'num_gt':int(n_gt), 
             'num_valid':0, 
             'mean_err':float("nan"), 
             'median_err':float("nan"), 
             'tau_mean':float("nan"), 
             'tau_min':float("nan"), 
             'tau_max':float("nan")}
            return (
             loss, stats)
        else:
            gt_mask = batch.get("gt_mask", None)
            if gt_mask is not None:
                valid_mask = gt_mask.to(device=device)
                if valid_mask.sum() == 0:
                    valid_mask = torch.ones(M, dtype=(torch.bool), device=device)
            else:
                valid_mask = torch.ones(M, dtype=(torch.bool), device=device)
    mkpts0_f_valid = mkpts0_f[valid_mask]
    mkpts1_f_valid = mkpts1_f[valid_mask]
    expec_f_valid = expec_f[valid_mask]
    M_valid = mkpts0_f_valid.shape[0]
    if M_valid == 0:
        loss = mkpts0_f.sum() * 0.0
        stats = {'num_pred':int(M), 
         'num_gt':int(n_gt), 
         'num_valid':0, 
         'mean_err':float("nan"), 
         'median_err':float("nan"), 
         'tau_mean':float("nan"), 
         'tau_min':float("nan"), 
         'tau_max':float("nan")}
        return (
         loss, stats)
    tau = expec_f_valid[:, 2]
    tau = torch.clamp(tau, min=0.001)
    dists = torch.cdist(mkpts0_f_valid, mkpts0_gt)
    nn_idx = dists.argmin(dim=1)
    mkpts1_gt_nn = mkpts1_gt[nn_idx]
    diff = mkpts1_f_valid - mkpts1_gt_nn
    err2 = (diff ** 2).sum(dim=1)
    weights = 1.0 / (tau ** 2 + eps)
    loss = (weights * err2).mean()
    with torch.no_grad():
        err = torch.sqrt(err2)
        mean_err = float(err.mean().item())
        median_err = float(err.median().item())
        tau_mean = float(tau.mean().item())
        tau_min = float(tau.min().item())
        tau_max = float(tau.max().item())
    stats = {'num_pred':int(M), 
     'num_gt':int(n_gt), 
     'num_valid':int(M_valid), 
     'mean_err':mean_err, 
     'median_err':median_err, 
     'tau_mean':tau_mean, 
     'tau_min':tau_min, 
     'tau_max':tau_max}
    return (
     loss, stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug SFPPR fine-level loss (Lf).")
    parser.add_argument("--manifest",
      type=str,
      required=True,
      help="Path to training manifest CSV (e.g. pf0_train_manifest_cov_pxmask.csv)")
    parser.add_argument("--index",
      type=int,
      default=0,
      help="Dataset index to debug.")
    parser.add_argument("--ckpt",
      type=str,
      default="",
      help="Path to SFPPRLoFTR checkpoint (joint coarse+rep など).")
    parser.add_argument("--resize-long",
      type=int,
      default=840,
      help="Resize long side of image while keeping aspect ratio.")
    parser.add_argument("--device",
      type=str,
      default="cuda",
      help="Device: 'cuda' or 'cpu'.")
    parser.add_argument("--max-gt-points",
      type=int,
      default=4096,
      help="Max number of GT points to use for NN search (for speed).")
    args = parser.parse_args()
    device = torch.device(args.device)
    print(f"[DEBUG] device = {device}")
    dataset = WaymoSFPPRDataset(manifest_csv=(args.manifest),
      sources=[
     "strict", "close_only"],
      min_n_matches=8,
      resize_long_side=(args.resize_long))
    print(f"[WaymoSFPPRDataset] len(dataset) = {len(dataset)}")
    idx = min(max(args.index, 0), len(dataset) - 1)
    sample = dataset[idx]
    print(f"[DEBUG] use dataset index = {idx}")
    print("[DEBUG] sample keys:", sample.keys())
    print("[DEBUG] image0 shape:", sample["image0"].shape)
    print("[DEBUG] image1 shape:", sample["image1"].shape)
    print("[DEBUG] mkpts0 shape:", sample["mkpts0"].shape)
    print("[DEBUG] mkpts1 shape:", sample["mkpts1"].shape)
    print("[DEBUG] mask0_coarse shape:", sample["mask0_coarse"].shape)
    print("[DEBUG] imsize0:", sample["imsize0"])
    print("[DEBUG] imsize1:", sample["imsize1"])
    cfg = default_cfg.copy()
    matcher = SFPPRLoFTR(config=cfg)
    matcher = matcher.to(device)
    if args.ckpt:
        print(f"[DEBUG] load ckpt from: {args.ckpt}")
        ckpt = torch.load((args.ckpt), map_location=device)
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        missing, unexpected = matcher.load_state_dict(state_dict, strict=False)
        print(f"[DEBUG] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print("  missing:", missing)
        if unexpected:
            print("  unexpected:", unexpected)
    matcher.eval()
    batch = {'image0':(sample["image0"].unsqueeze(0).to)(device), 
     'image1':(sample["image1"].unsqueeze(0).to)(device), 
     'imsize0':(sample["imsize0"].unsqueeze(0).to)(device), 
     'imsize1':(sample["imsize1"].unsqueeze(0).to)(device), 
     'mask0_coarse':(sample["mask0_coarse"].unsqueeze(0).to)(device), 
     'mkpts0':(sample["mkpts0"].unsqueeze(0).to)(device), 
     'mkpts1':(sample["mkpts1"].unsqueeze(0).to)(device)}
    matcher.zero_grad()
    matcher(batch)
    print("[DEBUG] after forward, batch keys:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
        else:
            print(f"  {k}: type={type(v)}")
    else:
        loss_f, stats = compute_fine_loss_from_batch(batch,
          max_gt_points=(args.max_gt_points))
        print("\n[DEBUG] fine loss L_f =", float(loss_f.item()))
        print("[DEBUG] fine loss stats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        else:
            loss_f.backward()
            print("\n[DEBUG] backward() 完了")
            print("\n[DEBUG] grad summary (mean |grad| per block):")
            blocks = [
             'backbone0', 
             'backbone1', 
             'coarse_matching', 
             'fine_preprocess', 
             'fine_matching', 
             'repeatability_head']
            for bname in blocks:
                grads = []
                for name, p in matcher.named_parameters():
                    if bname in name and p.grad is not None:
                        grads.append(p.grad.abs().mean().item())

                if grads:
                    mean_grad = sum(grads) / len(grads)
                    print(f"  {bname}: mean |grad| = {mean_grad:.3e} (n_params_with_grad={len(grads)})")
                else:
                    print(f"  {bname}: no grad")
            else:
                print("\n[DEBUG] example param grads:")
                for name, p in matcher.named_parameters():
                    if p.grad is None:
                        pass
                    elif any((s in name for s in ('fine', 'backbone0.conv1', 'backbone1.conv1'))):
                        gmean = p.grad.abs().mean().item()
                        print(f"  {name}: mean |grad| = {gmean:.3e}")


if __name__ == "__main__":
    main()
