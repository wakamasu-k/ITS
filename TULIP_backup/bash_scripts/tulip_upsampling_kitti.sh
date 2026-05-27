#!/bin/bash
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

args=(
    --batch_size 8
    --epochs 30
    --num_workers 2
    --lr 5e-4
    --weight_decay 0.01
    --warmup_epochs 5

    # Model parameters
    --model_select tulip_base
    --pixel_shuffle
    --circular_padding
    --log_transform
    --patch_unmerging

    # Dataset
    --dataset_select kitti
    --data_path_low_res ./dataset/KITTI/
    --data_path_high_res ./dataset/KITTI/

    # WandB Parameters
    --run_name tulip_base_intensity
    --entity myentity
    # --wandb_disabled
    --project_name experiment_kitti

    # Output
    --output_dir ./experiment/kitti/tulip_base_intensity

    # Image size
    --img_size_low_res 16 1024
    --img_size_high_res 64 1024

    # Model shape
    --window_size 2 8
    --patch_size 1 4
    --in_chans 2
)

torchrun --nproc_per_node=1 tulip/main_lidar_upsampling.py "${args[@]}"