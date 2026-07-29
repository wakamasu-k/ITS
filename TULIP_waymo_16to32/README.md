# Waymo TULIP 16-to-32 Line Upsampling

This directory is an isolated development area for Waymo Open Dataset
16-line to 32-line TULIP experiments.

Existing `TULIP_backup` and `handover_shimizu` code is intentionally not
modified by this scaffold.

## Planned pipeline

Waymo TFRecord -> native 64-line range image -> 32-line GT -> 16-line input
-> TULIP -> predicted 32-line output -> point cloud -> camera-plane
intensity image -> matching evaluation.

## Environment separation

- `waymo_preprocess/`: run in the Waymo/TensorFlow environment.
- `tulip_adapter/`: run in the TULIP/PyTorch environment.

The Python files are currently scaffolds. Implementation will be added in
stages after the one-frame range-image validation.
