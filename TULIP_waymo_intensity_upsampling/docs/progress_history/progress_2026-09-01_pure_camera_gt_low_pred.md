# Waymo pure GT-Low-Pred camera comparison

Date: 2026-09-01

## Purpose

Compare GT, Low, and Pred independently using the same camera calibration, projection convention, and z-buffer.

- GT: 32-line GT range and intensity
- Low: 16-line input range and intensity placed on even rows of the 32-line grid
- Pred: TULIP predicted range and intensity

This is not oracle mode (GT range plus predicted intensity).

## Implementation

Added `tools/visualize_waymo_16_to_32_camera_gt_low_pred.py`.

The TULIP model, TULIP_backup, and Phase 1-3 code were not modified.

Generated artifacts are camera_rgb.png, intensity projection and overlay PNGs, camera_comparison_raw.npz, and metadata.json. PNG files are visualization only; numeric inspection uses the raw NPZ.

## Real-data smoke test

Test split, sample 0, FRONT camera:

- camera image: `[1280, 1920, 3]`
- GT projected points: `9248`
- Low projected points: `4654`
- Pred projected points: `9248`
- GT valid LiDAR pixels: `74352`
- Low valid LiDAR pixels: `37100`
- Pred valid LiDAR pixels: `74352`
- GT projection rate: `0.1243813213`
- Low projection rate: `0.1254447439`
- Pred projection rate: `0.1243813213`

The same sample was also generated for the side cameras:

| camera | GT points | Low points | Pred points | GT rate | Low rate | Pred rate |
|---|---:|---:|---:|---:|---:|---:|
| FRONT | 9248 | 4654 | 9248 | 0.124381 | 0.125445 | 0.124381 |
| FRONT_LEFT | 10500 | 5352 | 10504 | 0.141220 | 0.144259 | 0.141274 |
| FRONT_RIGHT | 10923 | 5575 | 10925 | 0.146909 | 0.150270 | 0.146936 |

Output directories:

- `outputs/waymo_16_to_32_full_e300/pure_camera_test_sample0_front/`
- `outputs/waymo_16_to_32_full_e300/pure_camera_test_sample0_front_left/`
- `outputs/waymo_16_to_32_full_e300/pure_camera_test_sample0_front_right/`

All NPZ arrays reloaded successfully. Coordinates, depth, and intensity are float32; maps are float32; masks are bool; finite-value checks passed.

## Environment note

The TFRecord reader requires `.venv_waymo`. Since that environment lacks the TULIP import dependencies `timm` and `chamfer_distance`, the smoke test used transient runtime stubs for imports not called by this evaluation path. This is not a substitute for a unified production environment.

## Next checks

Run samples 10, 20, 30, and 39 for the three cameras, then inspect metadata.json and camera_comparison_raw.npz before visual comparison with RGB.

