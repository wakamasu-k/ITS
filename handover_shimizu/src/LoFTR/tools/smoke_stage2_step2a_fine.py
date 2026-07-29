from pathlib import Path
import torch
import torch.nn.functional as F
from einops.einops import rearrange

from loftr.loftr_sfppr import SFPPRLoFTR
from loftr.datasets.waymo_step2a_manifest_fine_dataset import WaymoStep2AFineManifestDataset
from copy import deepcopy
from loftr import default_cfg


def load_state(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("[load] missing:", len(missing))
    print("[load] unexpected:", len(unexpected))


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_fine(model):
    for m in [model.fine_preprocess, model.loftr_fine]:
        for p in m.parameters():
            p.requires_grad = True


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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

    # Datasetから来たGT coarse対応をfine stageに使う
    i_ids = data["i_ids_gt"].view(-1).long()
    j_ids = data["j_ids_gt"].view(-1).long()
    b_ids = torch.zeros_like(i_ids)

    data["b_ids"] = b_ids
    data["i_ids"] = i_ids
    data["j_ids"] = j_ids

    stride = int(data.get("stride", 8))
    if not isinstance(stride, int):
        stride = int(stride.view(-1)[0].item())

    # fine_matching.get_fine_match が必要とする値を用意
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

    err2 = ((pred - gt) ** 2).sum(dim=1)
    loss = (err2 / (tau ** 2 + eps)).mean()
    return loss, err2.sqrt().mean().item(), n


def main():
    manifest = "/mnt/w/32line/datasets/loftr_pairs_shifted/training/manifest_matches_step2a_paper_final_wdrive.csv"
    ckpt = str(Path.home() / "ITS/waymo_outputs/old32_dense/loftr_retrain_stage1_paper_wdrive/stage1_final_gstep715958.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device =", device)

    ds = WaymoStep2AFineManifestDataset(
        manifest_csv=manifest,
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

    sample = ds[0]

    data = {}
    for k, v in sample.items():
        if torch.is_tensor(v):
            if k in ["image0", "image1"]:
                data[k] = v.unsqueeze(0).to(device)
            else:
                data[k] = v.to(device)
        else:
            data[k] = v

    cfg = deepcopy(default_cfg)
    cfg["match_coarse"]["match_type"] = "dual_softmax"

    matcher = SFPPRLoFTR(config=cfg, enable_fine=True, enable_repeatability=False).to(device)
    load_state(matcher, ckpt)

    freeze_all(matcher)
    unfreeze_fine(matcher)
    matcher.train()

    print("[INFO] trainable params:", count_trainable(matcher))

    out = forward_fine_with_gt(matcher, data)
    loss, mean_err, n = fine_loss(out)

    print("[OK] forward fine completed")
    print("expec_f:", out["expec_f"].shape)
    print("expec_f_gt:", out["expec_f_gt"].shape)
    print("n:", n)
    print("loss:", float(loss.item()))
    print("mean_err:", mean_err)

    loss.backward()
    print("[OK] backward completed")


if __name__ == "__main__":
    main()
