# uncompyle6 version 3.9.3
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.10.12 (main, Mar  3 2026, 11:56:32) [GCC 11.4.0]
# Embedded file name: /home/ubuntu/waymo/datasets/LoFTR/tools/train_sfppr_all_joint.py
# Compiled at: 2025-11-25 03:13:01
# Size of source mod 2**32: 8720 bytes
import argparse, os, torch
from tqdm import tqdm
from loftr import default_cfg
from loftr.loftr_sfppr import SFPPRLoFTR
from loftr.datasets.waymo_sfppr_dataset import WaymoSFPPRDataset
from train_sfppr_coarse_rep_joint import compute_coarse_loss, compute_repeatability_loss, count_trainable_params
from debug_sfppr_fine_loss import compute_fine_loss_from_batch

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint training of SFPPRLoFTR with coarse, repeatability, and fine losses (L = L_c + lambda_rep * L_r + lambda_fine * L_f).")
    parser.add_argument("--manifest",
      type=str,
      required=True,
      help="Training manifest CSV (e.g., pf0_train_manifest_cov_pxmask.csv).")
    parser.add_argument("--sources",
      type=str,
      nargs="+",
      default=[
     "strict", "close_only"],
      help="List of source tags to use (e.g., strict close_only).")
    parser.add_argument("--epochs",
      type=int,
      default=1,
      help="Number of training epochs.")
    parser.add_argument("--lr",
      type=float,
      default=0.0001,
      help="Learning rate for Adam optimizer.")
    parser.add_argument("--device",
      type=str,
      default="cuda",
      help="Device to use (e.g., cuda or cpu).")
    parser.add_argument("--out-dir",
      type=str,
      required=True,
      help="Directory to save checkpoints.")
    parser.add_argument("--max-steps",
      type=int,
      default=(-1),
      help="Maximum training steps (global). If < 0, use all samples in each epoch.")
    parser.add_argument("--resize-long",
      type=int,
      default=840,
      help="Resize long side of images (WaymoSFPPRDataset).")
    parser.add_argument("--init-ckpt",
      type=str,
      required=True,
      help="Initial checkpoint path (typically fine-only checkpoint).")
    parser.add_argument("--lambda-coarse",
      type=float,
      default=1.0,
      help="Weight for coarse loss L_c.")
    parser.add_argument("--lambda-rep",
      type=float,
      default=0.5,
      help="Weight for repeatability loss L_r.")
    parser.add_argument("--lambda-fine",
      type=float,
      default=1.0,
      help="Weight for fine loss L_f.")
    parser.add_argument("--max-gt-points",
      type=int,
      default=4096,
      help="Maximum number of GT points used in fine loss (sub-sampling).")
    return parser.parse_args()


def build_batch_from_sample(sample: dict, device: torch.device) -> dict:
    """
    WaymoSFPPRDataset の 1 サンプル (CPU Tensor) から、
    SFPPRLoFTR にそのまま渡せる 1 バッチ分の dict を組み立てる。
    """
    batch = {'image0':sample["image0"].unsqueeze(0).to(device, non_blocking=True), 
     'image1':sample["image1"].unsqueeze(0).to(device, non_blocking=True), 
     'mkpts0':sample["mkpts0"].unsqueeze(0).to(device, non_blocking=True), 
     'mkpts1':sample["mkpts1"].unsqueeze(0).to(device, non_blocking=True), 
     'mask0_coarse':sample["mask0_coarse"].unsqueeze(0).to(device, non_blocking=True), 
     'imsize0':sample["imsize0"].unsqueeze(0).to(device, non_blocking=True), 
     'imsize1':sample["imsize1"].unsqueeze(0).to(device, non_blocking=True)}
    return batch


def main() -> None:
    args = parse_args()
    os.makedirs((args.out_dir), exist_ok=True)
    if args.device == "cuda":
        torch.cuda.is_available() or print("[WARN] CUDA is not available, falling back to CPU.")
        device = torch.device(args.device)
    else:
        device = torch.device(args.device)
    print(f"[INFO] device = {device}")
    dataset = WaymoSFPPRDataset(manifest_csv=(args.manifest),
      sources=(args.sources),
      min_n_matches=8,
      resize_to=None,
      resize_long_side=(args.resize_long))
    num_samples = len(dataset)
    print(f"[INFO] num_samples = {num_samples}")
    print(f"[INFO] sources filter: {dataset.sources}")
    print(f"[INFO] resize_long_side: {args.resize_long}")
    cfg = default_cfg.copy()
    matcher = SFPPRLoFTR(config=cfg).to(device)
    if args.init_ckpt:
        print(f"[INFO] load init ckpt from {args.init_ckpt}")
        ckpt = torch.load((args.init_ckpt), map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        missing, unexpected = matcher.load_state_dict(state_dict, strict=False)
        print(f"[INFO] load_state_dict done: missing={len(missing)}, unexpected={len(unexpected)}")
    matcher.eval()
    for p in matcher.parameters():
        p.requires_grad = True
    else:
        num_trainable = count_trainable_params(matcher)
        print(f"[INFO] num_trainable_params (all joint) = {num_trainable}")
        optimizer = torch.optim.Adam([p for p in matcher.parameters() if p.requires_grad],
          lr=(args.lr))
        global_step = 0
        for epoch in range(1, args.epochs + 1):
            epoch_loss = 0.0
            epoch_coarse = 0.0
            epoch_rep = 0.0
            epoch_fine = 0.0
            steps_in_epoch = 0
            indices = torch.randperm(num_samples)
            pbar = tqdm(indices, desc=f"Epoch {epoch}/{args.epochs}", ncols=120)

        for idx in pbar:
            idx_int = int(idx)
            sample = dataset[idx_int]
            batch = build_batch_from_sample(sample, device=device)
            optimizer.zero_grad()
            matcher(batch)
            loss_c = compute_coarse_loss(batch, sample)
            loss_r = compute_repeatability_loss(batch)
            loss_f, _fine_stats = compute_fine_loss_from_batch(batch,
              max_gt_points=(args.max_gt_points))
            total_loss = args.lambda_coarse * loss_c + args.lambda_rep * loss_r + args.lambda_fine * loss_f
            total_loss.backward()
            optimizer.step()
            global_step += 1
            steps_in_epoch += 1
            epoch_loss += float(total_loss.item())
            epoch_coarse += float(loss_c.item())
            epoch_rep += float(loss_r.item())
            epoch_fine += float(loss_f.item())
            avg_loss = epoch_loss / steps_in_epoch
            avg_c = epoch_coarse / steps_in_epoch
            avg_r = epoch_rep / steps_in_epoch
            avg_f = epoch_fine / steps_in_epoch
            pbar.set_postfix(loss=(f"{avg_loss:.3f}"),
              Lc=(f"{avg_c:.3f}"),
              Lr=(f"{avg_r:.3f}"),
              Lf=(f"{avg_f:.3f}"),
              step=global_step)
            if args.max_steps > 0:
                if global_step >= args.max_steps:
                    break
                print(f"[INFO] epoch {epoch} avg_loss = {avg_loss:.6f}, avg_coarse = {avg_c:.6f}, avg_rep = {avg_r:.6f}, avg_fine = {avg_f:.6f}")
                ckpt_dict = {'state_dict':(matcher.state_dict)(), 
                 'epoch':epoch, 
                 'global_step':global_step, 
                 'lambda_coarse':args.lambda_coarse, 
                 'lambda_rep':args.lambda_rep, 
                 'lambda_fine':args.lambda_fine}
                ckpt_path = os.path.join(args.out_dir, f"epoch_{epoch:03d}.pth")
                latest_path = os.path.join(args.out_dir, "latest_model.pth")
                torch.save(ckpt_dict, ckpt_path)
                torch.save(ckpt_dict, latest_path)
                print(f"[INFO] saved checkpoint to {ckpt_path}")
                print(f"[INFO] saved latest model weights to {latest_path}")
                if args.max_steps > 0 and global_step >= args.max_steps:
                    print("[INFO] reached max_steps; stopping training.")
                    break


if __name__ == "__main__":
    main()
