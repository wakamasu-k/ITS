from pathlib import Path
import argparse
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from einops.einops import rearrange
from tqdm import tqdm

from loftr import default_cfg
from loftr.loftr_sfppr import SFPPRLoFTR
from loftr.datasets.waymo_step2a_manifest_fine_dataset import WaymoStep2AFineManifestDataset


def load_state(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("[load] missing:", len(missing))
    print("[load] unexpected:", len(unexpected))


def save_ckpt(model, out_dir, epoch, global_step, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        path,
    )
    print("[save]", path)


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_fine(model):
    # Stage-2: fine_preprocess + loftr_fine のみ学習
    for m in [model.fine_preprocess, model.loftr_fine]:
        for p in m.parameters():
            p.requires_grad = True


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def move_to_device(batch, device):
    data = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            data[k] = v.to(device, non_blocking=True)
        else:
            data[k] = v

    # DataLoaderを通さずに使った場合への保険
    if data["image0"].dim() == 3:
        data["image0"] = data["image0"].unsqueeze(0)
    if data["image1"].dim() == 3:
        data["image1"] = data["image1"].unsqueeze(0)

    return data


def maybe_subsample_gt(data, max_gt_points):
    i_ids = data["i_ids_gt"].view(-1).long()
    j_ids = data["j_ids_gt"].view(-1).long()
    expec_f_gt = data["expec_f_gt"].view(-1, 2)

    n = i_ids.numel()
    if max_gt_points > 0 and n > max_gt_points:
        perm = torch.randperm(n, device=i_ids.device)[:max_gt_points]
        i_ids = i_ids[perm]
        j_ids = j_ids[perm]
        expec_f_gt = expec_f_gt[perm]

    data["i_ids_gt"] = i_ids
    data["j_ids_gt"] = j_ids
    data["expec_f_gt"] = expec_f_gt
    return data


def coarse_ids_to_xy(ids, hw_c, stride, device):
    Hc, Wc = int(hw_c[0]), int(hw_c[1])
    ids = ids.long()
    y = torch.div(ids, Wc, rounding_mode="floor").float()
    x = (ids % Wc).float()
    xy = torch.stack([(x + 0.5) * stride, (y + 0.5) * stride], dim=1)
    return xy.to(device)


def forward_fine_with_gt(matcher, data):
    image0 = data["image0"]
    image1 = data["image1"]

    data.update({
        "bs": image0.size(0),
        "hw0_i": image0.shape[2:],
        "hw1_i": image1.shape[2:],
    })

    feat_c0, feat_f0 = matcher.backbone0(image0)
    feat_c1, feat_f1 = matcher.backbone1(image1)

    data.update({
        "hw0_c": feat_c0.shape[2:],
        "hw1_c": feat_c1.shape[2:],
        "hw0_f": feat_f0.shape[2:],
        "hw1_f": feat_f1.shape[2:],
    })

    feat_c0 = rearrange(matcher.pos_encoding(feat_c0), "b c h w -> b (h w) c")
    feat_c1 = rearrange(matcher.pos_encoding(feat_c1), "b c h w -> b (h w) c")

    # GT coarse対応をfine stageに入れる
    i_ids = data["i_ids_gt"].view(-1).long()
    j_ids = data["j_ids_gt"].view(-1).long()
    b_ids = torch.zeros_like(i_ids)

    data["b_ids"] = b_ids
    data["i_ids"] = i_ids
    data["j_ids"] = j_ids

    stride = data.get("stride", 8)
    if torch.is_tensor(stride):
        stride = int(stride.view(-1)[0].item())
    else:
        stride = int(stride)

    data["mkpts0_c"] = coarse_ids_to_xy(i_ids, data["hw0_c"], stride, image0.device)
    data["mkpts1_c"] = coarse_ids_to_xy(j_ids, data["hw1_c"], stride, image0.device)
    data["mconf"] = torch.ones_like(i_ids, dtype=torch.float32, device=image0.device)

    feat_f0_unfold, feat_f1_unfold = matcher.fine_preprocess(
        feat_f0, feat_f1, feat_c0, feat_c1, data
    )

    if feat_f0_unfold.size(0) != 0:
        feat_f0_unfold, feat_f1_unfold = matcher.loftr_fine(
            feat_f0_unfold, feat_f1_unfold
        )

    matcher.fine_matching(feat_f0_unfold, feat_f1_unfold, data)
    return data


def fine_loss(data, eps=1e-6):
    pred = data["expec_f"][:, :2]
    tau = data["expec_f"][:, 2].clamp(min=1e-3)
    gt = data["expec_f_gt"].view(-1, 2).to(pred.device)

    n = min(pred.shape[0], gt.shape[0])
    pred = pred[:n]
    tau = tau[:n]
    gt = gt[:n]

    if n == 0:
        loss = pred.sum() * 0.0
        return loss, 0.0, 0

    err2 = ((pred - gt) ** 2).sum(dim=1)
    loss = (err2 / (tau ** 2 + eps)).mean()
    mean_err = err2.sqrt().mean().item()
    return loss, mean_err, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--init-ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--max-gt-points", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("[INFO] device =", device)

    ds = WaymoStep2AFineManifestDataset(
        manifest_csv=args.manifest,
        min_matches=8,
        min_fine_ok=200,
        skip_error_rows=True,
        verify_files=False,
        resize_long_side=840,
        coarse_stride=8,
        fine_stride=2,
        fine_window_size=5,
        cell_point="topleft",
        only_fine_ok=True,
        verbose=True,
    )

    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    cfg = deepcopy(default_cfg)
    cfg["match_coarse"]["match_type"] = "dual_softmax"

    matcher = SFPPRLoFTR(config=cfg, enable_fine=True, enable_repeatability=False).to(device)
    load_state(matcher, args.init_ckpt)

    freeze_all(matcher)
    unfreeze_fine(matcher)
    matcher.train()

    print("[INFO] trainable params:", count_trainable(matcher))

    optimizer = torch.optim.AdamW(
        [p for p in matcher.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.0,
    )

    global_step = 0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{args.epochs}")
        loss_sum = 0.0
        err_sum = 0.0
        cnt = 0

        for batch in pbar:
            data = move_to_device(batch, device)
            data = maybe_subsample_gt(data, args.max_gt_points)

            optimizer.zero_grad(set_to_none=True)

            out = forward_fine_with_gt(matcher, data)
            loss, mean_err, n = fine_loss(out)

            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in matcher.parameters() if p.requires_grad],
                    args.grad_clip,
                )

            optimizer.step()

            global_step += 1
            loss_sum += float(loss.item())
            err_sum += float(mean_err)
            cnt += 1

            pbar.set_postfix({
                "loss": f"{loss_sum / cnt:.4f}",
                "err": f"{err_sum / cnt:.4f}",
                "n": n,
                "gstep": global_step,
            })

            if args.save_every > 0 and global_step % args.save_every == 0:
                save_ckpt(matcher, out_dir, epoch, global_step, f"step_{global_step:07d}.pth")
                save_ckpt(matcher, out_dir, epoch, global_step, "latest_model.pth")

            if args.max_steps > 0 and global_step >= args.max_steps:
                save_ckpt(matcher, out_dir, epoch, global_step, "latest_model.pth")
                print("[INFO] reached max_steps:", args.max_steps)
                return

        save_ckpt(matcher, out_dir, epoch, global_step, f"epoch_{epoch:03d}.pth")
        save_ckpt(matcher, out_dir, epoch, global_step, "latest_model.pth")

    print("[INFO] training finished.")


if __name__ == "__main__":
    main()
