# 使い方:
#   python train_stage1_step2a_backbone0_coarse_fast_bf16v2.py --help
#   必要な入力パスと出力先を引数で指定して実行する。


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 目的: Stage1（coarse）学習を行う。

"""
Stage1（Step2A v6）で backbone0 と coarse のみを学習する。BF16 で安全に動かす。
- max-steps は総ステップ数ではなく、各 epoch ごとの上限。
- 定期的に checkpoint を保存し、クラッシュ後に resume できるようにする。
- --resume では out_dir/latest_model.pth から、--resume-ckpt では明示パスから再開する。


"""

import argparse
import copy
import os
import time
from collections import OrderedDict
from typing import Dict, Any, List, Optional

import torch
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from loftr import default_cfg
from loftr.loftr_sfppr import SFPPRLoFTR
from loftr.datasets.waymo_step2a_manifest_dataset import WaymoStep2AManifestDataset
from loftr.datasets.waymo_step2a_manifest_fine_dataset import WaymoStep2AFineManifestDataset


def parse_args() -> argparse.Namespace:
    """CLI 引数を定義して解析結果を返す。"""
    p = argparse.ArgumentParser(description="Stage1 (Step2A v6): backbone0 + coarse only. BF16-safe + resumable.")
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--epochs", type=int, default=1)

    # 重要: これは各 epoch ごとの上限。-1 でフル epoch にする。
    p.add_argument("--max-steps", type=int, default=2000, help="Max steps PER EPOCH (-1 = full epoch).")

    # 任意: 全 epoch をまたいだ総上限（「合計 N step 学習」に便利）。
    p.add_argument("--max-steps-total", type=int, default=-1, help="Stop after N total steps (-1 = no limit).")

    # 初期化 / 再開
    p.add_argument("--init-ckpt", type=str, required=True, help="Init ckpt (used only when NOT resuming).")
    p.add_argument("--resume", action="store_true", help="Resume from <out-dir>/latest_model.pth if exists.")
    p.add_argument("--resume-ckpt", type=str, default=None, help="Resume from an explicit ckpt path.")

    # 最適化設定
    p.add_argument("--lr-backbone0", type=float, default=1e-5)
    p.add_argument("--lr-coarse", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)

    # データセットのフィルタ条件
    p.add_argument("--min-matches", type=int, default=8)
    p.add_argument("--skip-error-rows", action="store_true")
    p.add_argument("--no-verify-files", action="store_true")
    p.add_argument(
        "--dataset-mode",
        type=str,
        default="fine_remap",
        choices=["fine_remap", "legacy"],
        help=(
            "fine_remap: use WaymoStep2AFineManifestDataset (Stage2-like resize remap + 1-to-1 dedup). "
            "legacy: original WaymoStep2AManifestDataset behavior."
        ),
    )
    p.add_argument(
        "--min-fine-ok",
        type=int,
        default=0,
        help="Only used in --dataset-mode fine_remap. Keep rows with n_fine >= this value.",
    )
    p.add_argument(
        "--only-fine-ok",
        action="store_true",
        help="Only used in --dataset-mode fine_remap. Keep only fine-window-valid GT points.",
    )

    # 速度 / 安定性に関する設定
    p.add_argument("--resize-long", type=int, default=840)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--train-bn0", action="store_true")

    # AMP の dtype
    p.add_argument("--amp-dtype", type=str, default="bf16", choices=["none", "fp16", "bf16"])

    # checkpoint 保存設定
    p.add_argument("--save-every", type=int, default=1000, help="Save latest_model.pth every N steps (0=disable).")

    # 再現性設定（sampler 用）
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args()


def count_params(params: List[torch.nn.Parameter]) -> int:
    """学習対象となるパラメータ数を数える。"""
    return int(sum(p.numel() for p in params))


def freeze_module(m: torch.nn.Module) -> None:
    """指定モジュールの勾配更新を止める。"""
    for p in m.parameters():
        p.requires_grad = False


def unfreeze_module(m: torch.nn.Module) -> None:
    """指定モジュールの勾配更新を再開する。"""
    for p in m.parameters():
        p.requires_grad = True


def _strip_matcher_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """`_strip_matcher_prefix` に対応する内部補助処理をまとめる。"""
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("matcher."):
            out[k[len("matcher."):]] = v
        else:
            out[k] = v
    return out


def _looks_like_dual_backbone(sd: Dict[str, torch.Tensor]) -> bool:
    """モデル重みが dual-backbone 形式かどうかを判定する。"""
    return any(k.startswith("backbone0.") or k.startswith("backbone1.") for k in sd.keys())


def _looks_like_single_backbone(sd: Dict[str, torch.Tensor]) -> bool:
    """モデル重みが single-backbone 形式かどうかを判定する。"""
    return any(k.startswith("backbone.") for k in sd.keys())


def merge_single_backbone_to_dual(sd_single: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """single-backbone 重みを dual-backbone 初期値へ展開する。"""
    sd_single = _strip_matcher_prefix(sd_single)
    out = OrderedDict()
    for k, v in sd_single.items():
        if k.startswith("backbone."):
            suf = k[len("backbone."): ]
            out["backbone0." + suf] = v
            out["backbone1." + suf] = v
        else:
            out[k] = v
    return out


def load_init_ckpt_flexible(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    """複数形式に対応して初期重みを読み込む。"""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[ERROR] init ckpt not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd_raw = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd_raw = ckpt
    else:
        raise ValueError(f"[ERROR] unexpected ckpt format: {type(ckpt)}")

    sd_raw = _strip_matcher_prefix(sd_raw)

    if _looks_like_dual_backbone(sd_raw):
        sd = sd_raw
        print("[INFO] init_ckpt looks like dual-backbone; loading directly (strict=False).", flush=True)
    elif _looks_like_single_backbone(sd_raw):
        sd = merge_single_backbone_to_dual(sd_raw)
        print("[INFO] init_ckpt looks like single-backbone; duplicating into backbone0/backbone1 then loading.", flush=True)
    else:
        sd = sd_raw
        print("[WARN] init_ckpt has no obvious backbone prefix; loading as-is (strict=False).", flush=True)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[INFO] load_state_dict done: missing={len(missing)}, unexpected={len(unexpected)}", flush=True)


def coarse_loss_nll_from_ids(batch: Dict[str, Any], i_ids_gt: torch.Tensor, j_ids_gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """coarse stage 用の負対数尤度損失を計算する。"""
    conf = batch["conf_matrix"]  # (1,HW0,HW1)
    if conf.dim() != 3 or conf.size(0) != 1:
        raise ValueError(f"batch_size=1 only. conf_matrix.shape={tuple(conf.shape)}")

    conf0 = conf[0]  # (HW0,HW1)
    HW0, HW1 = int(conf0.size(0)), int(conf0.size(1))

    i = i_ids_gt.to(conf0.device).long().view(-1)
    j = j_ids_gt.to(conf0.device).long().view(-1)

    valid = (i >= 0) & (i < HW0) & (j >= 0) & (j < HW1)
    if int(valid.sum().item()) == 0:
        return conf0.new_tensor(0.0)

    conf_gt = conf0[i[valid], j[valid]].clamp(min=eps)
    return -torch.log(conf_gt).mean()


def atomic_torch_save(obj: Dict[str, Any], path: str) -> None:
    """torch オブジェクトを一時ファイル経由で安全に保存する。"""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


class MutableListSampler(Sampler[int]):
    """epoch 間で index 一覧を差し替えられる sampler。persistent workers に対応する。"""
    def __init__(self, indices: Optional[List[int]] = None):
        """インスタンス生成時に必要な設定値と内部状態を初期化する。"""
        self.indices: List[int] = list(indices) if indices is not None else []

    def set_indices(self, indices: List[int]) -> None:
        """イテレータやデータセットが使うインデックス列を更新する。"""
        self.indices = list(indices)

    def __iter__(self):
        """現在の設定に対応する要素列を順に返す。"""
        return iter(self.indices)

    def __len__(self) -> int:
        """保持しているサンプル数を返す。"""
        return len(self.indices)


def make_epoch_indices(n: int, seed: int, epoch: int, max_steps: int) -> List[int]:
    """1 エポック分のサンプル順序を生成する。"""
    g = torch.Generator()
    g.manual_seed(int(seed) + int(epoch))
    perm = torch.randperm(n, generator=g).tolist()
    if max_steps is None or int(max_steps) < 0:
        return perm
    return perm[: min(int(max_steps), n)]


def main() -> None:
    """CLI 引数を解釈し、入力の読み込みから結果保存までの一連の処理を実行する。"""
    t0 = time.perf_counter()
    print("[BOOT] start", flush=True)

    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] device = {device}", flush=True)

    # 速度設定
    if device.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    torch.backends.cudnn.benchmark = True

    # dataset 設定
    print("[INFO] build dataset...", flush=True)
    if str(args.dataset_mode) == "fine_remap":
        ds = WaymoStep2AFineManifestDataset(
            manifest_csv=args.manifest,
            min_matches=args.min_matches,
            min_fine_ok=max(0, int(args.min_fine_ok)),
            skip_error_rows=args.skip_error_rows,
            verify_files=(not args.no_verify_files),
            resize_long_side=args.resize_long,
            coarse_stride=8,
            fine_stride=2,
            fine_window_size=5,
            cell_point="topleft",
            only_fine_ok=args.only_fine_ok,
            verbose=True,
        )
        print("[INFO] dataset-mode = fine_remap (Stage2-like remap + 1-to-1 dedup)", flush=True)
    else:
        ds = WaymoStep2AManifestDataset(
            manifest_csv=args.manifest,
            min_matches=args.min_matches,
            skip_error_rows=args.skip_error_rows,
            verify_files=(not args.no_verify_files),
            resize_long_side=args.resize_long,
            cell_point="topleft",
        )
        print("[INFO] dataset-mode = legacy", flush=True)
    print(f"[TIME] dataset ready: {time.perf_counter()-t0:.2f}s (len={len(ds)})", flush=True)

    # sampler と dataloader（shuffle を決定的に制御する）
    sampler = MutableListSampler([])
    nw = int(args.num_workers)
    pf = int(args.prefetch_factor)

    loader_kwargs = dict(
        dataset=ds,
        batch_size=1,
        sampler=sampler,
        pin_memory=True,
        drop_last=False,
    )
    if nw <= 0:
        loader = DataLoader(**loader_kwargs, num_workers=0)
    else:
        loader = DataLoader(
            **loader_kwargs,
            num_workers=nw,
            persistent_workers=True,
            prefetch_factor=max(1, pf),
        )
    print(f"[TIME] dataloader ready: {time.perf_counter()-t0:.2f}s", flush=True)

    # モデル
    cfg = copy.deepcopy(default_cfg)
    matcher = SFPPRLoFTR(config=cfg, enable_fine=False, enable_repeatability=False).to(device)

    # 最適化器
    params_backbone0 = [p for p in matcher.backbone0.parameters()]
    params_coarse = [p for p in matcher.loftr_coarse.parameters()]
    optim = torch.optim.AdamW(
        [{"params": params_backbone0, "lr": args.lr_backbone0},
         {"params": params_coarse, "lr": args.lr_coarse}],
        weight_decay=args.weight_decay,
    )

    # AMP 設定
    amp_mode = str(args.amp_dtype).lower()
    use_amp = (amp_mode != "none") and (device.type == "cuda")
    if amp_mode == "bf16":
        amp_dtype = torch.bfloat16
        use_scaler = False
        if device.type == "cuda" and (not torch.cuda.is_bf16_supported()):
            raise RuntimeError("bf16 requested but not supported on this GPU.")
    elif amp_mode == "fp16":
        amp_dtype = torch.float16
        use_scaler = True
    else:
        amp_dtype = torch.float32
        use_scaler = False

    scaler = torch.amp.GradScaler("cuda", enabled=(use_scaler and device.type == "cuda"))

    # モジュール凍結（Stage1 方針）
    freeze_module(matcher.backbone1)
    freeze_module(matcher.fine_preprocess)
    freeze_module(matcher.loftr_fine)
    freeze_module(matcher.fine_matching)
    freeze_module(matcher.repeatability_head)

    unfreeze_module(matcher.backbone0)
    unfreeze_module(matcher.loftr_coarse)

    # モード設定
    matcher.train()
    matcher.coarse_matching.eval()
    matcher.backbone1.eval()
    if args.train_bn0:
        matcher.backbone0.train()
        print("[INFO] backbone0: train() (BN update ON)", flush=True)
    else:
        matcher.backbone0.eval()
        print("[INFO] backbone0: eval() (BN frozen)", flush=True)
    matcher.loftr_coarse.train()

    print(f"[INFO] amp: enabled={use_amp}, dtype={amp_mode}, scaler={scaler.is_enabled()}", flush=True)

    # resume するか init するか
    latest_path = os.path.join(args.out_dir, "latest_model.pth")
    resume_path = args.resume_ckpt if args.resume_ckpt is not None else latest_path

    start_epoch = 1
    start_step_in_epoch = 0
    global_step = 0
    base_seed = int(args.seed)

    if args.resume or (args.resume_ckpt is not None):
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"[ERROR] resume ckpt not found: {resume_path}")
        print(f"[INFO] RESUME from: {resume_path}", flush=True)
        ckpt = torch.load(resume_path, map_location=device)

        sd = ckpt.get("state_dict", ckpt)
        missing, unexpected = matcher.load_state_dict(sd, strict=False)
        print(f"[INFO] resume load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}", flush=True)

        if isinstance(ckpt, dict) and "optimizer" in ckpt:
            try:
                optim.load_state_dict(ckpt["optimizer"])
                print("[INFO] optimizer state loaded", flush=True)
            except Exception as e:
                print(f"[WARN] optimizer load failed: {repr(e)}", flush=True)

        if scaler.is_enabled() and isinstance(ckpt, dict) and "scaler" in ckpt:
            try:
                scaler.load_state_dict(ckpt["scaler"])
                print("[INFO] scaler state loaded", flush=True)
            except Exception as e:
                print(f"[WARN] scaler load failed: {repr(e)}", flush=True)

        start_epoch = int(ckpt.get("epoch", 1))
        start_step_in_epoch = int(ckpt.get("step_in_epoch", 0))
        global_step = int(ckpt.get("global_step", 0))

        ckpt_seed = ckpt.get("seed", base_seed)
        if int(ckpt_seed) != int(base_seed):
            print(f"[WARN] seed differs: args.seed={base_seed} ckpt.seed={ckpt_seed}. Use ckpt.seed for deterministic resume.", flush=True)
            base_seed = int(ckpt_seed)

        print(f"[INFO] resume position: epoch={start_epoch}, step_in_epoch={start_step_in_epoch}, global_step={global_step}", flush=True)

    else:
        print(f"[INFO] load init_ckpt: {args.init_ckpt}", flush=True)
        load_init_ckpt_flexible(matcher, args.init_ckpt, device=device)

    # 情報出力
    trainable = [p for p in matcher.parameters() if p.requires_grad]
    frozen = [p for p in matcher.parameters() if not p.requires_grad]
    print(f"[INFO] num_total_params      = {count_params(list(matcher.parameters()))}", flush=True)
    print(f"[INFO] num_trainable_params  = {count_params(trainable)}", flush=True)
    print(f"[INFO] num_frozen_params     = {count_params(frozen)}", flush=True)
    print(f"[TIME] start training loop: {time.perf_counter()-t0:.2f}s", flush=True)

    def save_ckpt(epoch: int, step_in_epoch: int, global_step_now: int, tag: str) -> None:
        """学習状態をチェックポイントとして保存する。"""
        obj: Dict[str, Any] = {
            "epoch": int(epoch),
            "step_in_epoch": int(step_in_epoch),
            "global_step": int(global_step_now),
            "state_dict": matcher.state_dict(),
            "optimizer": optim.state_dict(),
            "args": vars(args),
            "config": cfg,
            "seed": int(base_seed),
            "amp_dtype": str(amp_mode),
        }
        if scaler.is_enabled():
            obj["scaler"] = scaler.state_dict()

        # latest は毎回更新する
        atomic_torch_save(obj, latest_path)
        if tag:
            # epoch checkpoint は必要分だけ残す（ディスク肥大化を避ける）
            ep_path = os.path.join(args.out_dir, f"{tag}.pth")
            atomic_torch_save(obj, ep_path)
            print(f"[INFO] saved: {ep_path}", flush=True)
        print(f"[INFO] saved: {latest_path}", flush=True)

    # 学習ループ
    stop_all = False
    for ep in range(start_epoch, int(args.epochs) + 1):
        # 各 epoch の index 列（決定的）
        epoch_indices = make_epoch_indices(len(ds), base_seed, ep, args.max_steps)
        epoch_target = len(epoch_indices)

        # epoch 途中から resume した場合
        step0 = 0
        if ep == start_epoch and start_step_in_epoch > 0:
            step0 = min(start_step_in_epoch, epoch_target)
            epoch_indices = epoch_indices[step0:]
        sampler.set_indices(epoch_indices)

        loss_sum = 0.0
        loss_cnt = 0
        seen_cnt = step0  # how many samples already consumed in this epoch

        pbar = tqdm(loader, desc=f"Epoch {ep}/{args.epochs}", ncols=120, total=len(epoch_indices))
        for b in pbar:
            # 全体停止条件
            if int(args.max_steps_total) > 0 and global_step >= int(args.max_steps_total):
                stop_all = True
                break

            image0 = b["image0"].to(device, non_blocking=True)
            image1 = b["image1"].to(device, non_blocking=True)
            i_ids = b["i_ids_gt"][0]
            j_ids = b["j_ids_gt"][0]

            model_in = {"image0": image0, "image1": image1}
            optim.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                    matcher(model_in)
                    loss = coarse_loss_nll_from_ids(model_in, i_ids, j_ids)
            else:
                matcher(model_in)
                loss = coarse_loss_nll_from_ids(model_in, i_ids, j_ids)

            global_step += 1
            seen_cnt += 1

            if (not torch.isfinite(loss)) or float(loss.detach().cpu().item()) == 0.0:
                # resume 位置が進むよう、定期保存は継続する
                if int(args.save_every) > 0 and (global_step % int(args.save_every) == 0):
                    save_ckpt(ep, seen_cnt, global_step, tag="")
                continue

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()

            lv = float(loss.detach().cpu().item())
            loss_sum += lv
            loss_cnt += 1

            avg = loss_sum / max(loss_cnt, 1)
            pbar.set_postfix(loss=f"{lv:.3f}", avg=f"{avg:.3f}", gstep=global_step, cnt=loss_cnt, seen=seen_cnt)

            # 定期保存（latest のみ）
            if int(args.save_every) > 0 and (global_step % int(args.save_every) == 0):
                save_ckpt(ep, seen_cnt, global_step, tag="")

        avg = loss_sum / max(loss_cnt, 1)
        print(f"[INFO] epoch {ep} avg_loss = {avg:.6f} (count={loss_cnt}, seen={seen_cnt}/{epoch_target})", flush=True)

        # epoch スナップショットと latest を保存する
        save_ckpt(ep, seen_cnt, global_step, tag=f"epoch_{ep:03d}")

        # 再開後の最初の epoch が終わったら resume offset をクリアする
        start_step_in_epoch = 0

        if stop_all:
            print("[INFO] reached --max-steps-total. stopping.", flush=True)
            break

    print("[INFO] training finished.", flush=True)


if __name__ == "__main__":
    main()
