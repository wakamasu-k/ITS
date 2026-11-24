# A/B/C 5-frame static maps (2026-09-29)

## Scope

- Source: Waymo TOP LiDAR first return
- Frames: 0--4
- A: measured native rings `0,4,...,60`
- B: measured even rows plus TULIP-generated odd rows
- C: measured GT native rings `0,2,...,62`
- Coordinate frame: Waymo world
- Common removal: vehicle/pedestrian/cyclist boxes and the existing ego-roof box
- Checkpoint: `outputs/waymo_16_to_32_baseline_to_moderate_s0_e20/checkpoint.pth`

## Command

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true \
PYTHONPATH=/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling \
/home/wakamatsu/ITS/.venv_tulip/bin/python \
tools/build_tulip_16_to_32_static_map.py \
  --tfrecord /mnt/w/waymo_tf/perception_v1.4.3/individual_files/training/segment-10485926982439064520_4980_000_5000_000_with_camera_labels.tfrecord \
  --subset training \
  --tulip-root /home/wakamatsu/ITS/TULIP_backup/tulip \
  --checkpoint outputs/waymo_16_to_32_baseline_to_moderate_s0_e20/checkpoint.pth \
  --output-root outputs/waymo_16_to_32_baseline_to_moderate_s0_e20/static_maps_ABC_5frames_20260929 \
  --range-scale-m 75 \
  --intensity-transform log1p \
  --device cuda \
  --max-frames 5 \
  --map-variants A B C
```

## Results

| Variant | Static points | Failed frames | Map-median max distance | Points over 100 m |
|---|---:|---:|---:|---:|
| A observed16 | 142,797 | 0 | 78.249 m | 0 |
| B TULIP32 | 288,460 | 0 | 78.320 m | 0 |
| C GT32 | 284,530 | 0 | 78.445 m | 0 |

B removed 41 generated pixels above 75 m and 26,166 generated pixels with
invalid TOP pixel pose. Values were removed rather than clipped. Observed rows
were not range-limited.

All xyz/intensity values are finite. Every exact A xyz/intensity row exists in
both B and C (`missing=0`). The B `map_static.npz` SHA-256 exactly matches the
previous safe five-frame B output:

```text
72ced81b90779b0a5882d38072d760f42d37d6659805d8a413e025eab02698ad
```

## Next step

Render A/B/C from one common FRONT anchor with identical crop, camera,
normalization, point size, z-buffer, and index/depth output. Compare projected
point count, pixel coverage, B-added coverage, and C-relative depth/intensity
error before scaling to a full segment.
