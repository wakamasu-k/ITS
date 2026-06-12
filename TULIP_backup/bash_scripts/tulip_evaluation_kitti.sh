#!/bin/bash
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

args=(
    --eval

    # --mc_drop
    # --noise_threshold 0.03

    --model_select tulip_base
    --pixel_shuffle
    --circular_padding
    --patch_unmerging
    --log_transform

    --dataset_select kitti
    --data_path_low_res ./dataset/KITTI/
    --data_path_high_res ./dataset/KITTI/

    # 数値評価では一旦PLY保存しない
    --save_pcd

    --run_name tulip_base_intensity_eval
    --entity myentity
    --wandb_disabled
    --project_name kitti_evaluation

    --resume ./experiment/kitti/tulip_base_intensity/checkpoint-29.pth
    --output_dir ./experiment/kitti/tulip_base_intensity

    --img_size_low_res 16 1024
    --img_size_high_res 64 1024

    --window_size 2 8
    --patch_size 1 4
    --in_chans 2
)

torchrun --nproc_per_node=1 tulip/main_lidar_upsampling.py "${args[@]}"