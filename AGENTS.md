# ITS Codex handoff

## LiDAR localization research

When a task concerns Waymo LiDAR, TULIP, Retrieval, LoFTR or PnP, start by reading:

- [Research history and Git commits](TULIP_waymo_intensity_upsampling/docs/progress_history/research_history.md)
- [Experiment conditions, results and resume commands](TULIP_waymo_intensity_upsampling/docs/progress_history/progress_2026-11-24_lidar_handoff.md)

Use the checked-out branch and saved outputs to establish the latest status. The research checkpoint branch is `checkpoint/kitti-tulip-camera-eval`.

## WSL execution

- For experiments, use the existing `ITS` checkout at `/home/wakamatsu/ITS` on WSL Ubuntu-22.04.
- Use `.venv_waymo/bin/python` for data preparation and Retrieval, and `.venv_tulip/bin/python` for the inherited LoFTR evaluator. Resolve those paths from the repository root.
- Check the required checkpoint, datasets, output path and CUDA availability before running an experiment. The handoff lists the local checkpoints, `/mnt/w` data and prior outputs. Use `handover_shimizu` and the Waymo input datasets as references and inputs.
- Preserve existing results. Resume the same output directory only when the saved protocol matches the intended parameters. Use a separate output directory for a changed condition.

## Current research question

Can selective TULIP completion improve localization while preserving observed 16-line pixels? Fill a subset of image holes with TULIP points supported by nearby measured 3D points. Compare observed 16-line A, all-hole filling that preserves A's observed image values, and 3D-plane-selected filling. Keep the 2D depth-boundary mask and the PnP-only correspondence filter distinct from this image-generation experiment.

Check the saved run before resuming. The last documented 3D-plane pilot produced nine anchors. The next evaluation covers 443 database anchors and 374 validation queries with the same fixed Retrieval model. Use the resulting Top-10 candidates and their corresponding depth and point index for a subsequent LoFTR/PnP comparison. Report Retrieval recall and XY localization accuracy separately.
