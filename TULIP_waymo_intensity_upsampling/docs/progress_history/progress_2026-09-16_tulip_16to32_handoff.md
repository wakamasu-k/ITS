# Waymo TULIP 16→32: progress and handoff (2026-09-16)

## Purpose of this document

This is a self-contained handoff record for another AI or researcher.  It
separates confirmed facts from open questions and gives the exact current
artifacts and the safe next order of work.

## Research objective

Starting with a Waymo TOP LiDAR 16-line observation, use the TULIP/Swin
Transformer backbone to generate the missing rows of a 32-line native Range
Image.  Each pixel has two channels:

- channel 0: range [m]
- channel 1: Waymo raw intensity

The near-term objective is Range Image and point-cloud completion.  LoFTR/PnP
camera matching is intentionally not the current optimization target.

## Non-negotiable boundaries

- `/home/wakamatsu/ITS/TULIP_backup` is not edited.
- `/home/wakamatsu/ITS/handover_shimizu` is not edited.
- `/mnt/w/32line` is treated as read-only.
- The TULIP Transformer blocks, encoder, decoder, attention, merging and skip
  connections are reused from TULIP; no changes are made inside TULIP_backup.
- The Waymo model is a newly trained 2-channel model.  A KITTI checkpoint is
  not silently converted to it.
- The raw dataset is retained.  `log1p` is a training/evaluation transform,
  not a destructive rewrite of raw intensity.

## Repositories and environment

| Item | Path / value |
|---|---|
| Current Waymo project | `/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling` |
| Unmodified TULIP source | `/home/wakamatsu/ITS/TULIP_backup/tulip` |
| Integrated Python environment | `/home/wakamatsu/ITS/.venv_tulip` |
| Waymo TFRecord environment | `/home/wakamatsu/ITS/.venv_waymo` |
| Raw Waymo input | `/mnt/w/waymo_tf/perception_v1.4.3/individual_files` |

`.venv_tulip` has PyTorch/CUDA, TensorFlow, `timm`, `chamfer_distance`, and
`kornia==0.4.1`.  It is the preferred environment for model inference and
integrated reconstruction.  When TensorFlow and PyTorch share the GPU, use
`TF_FORCE_GPU_ALLOW_GROWTH=true` to avoid TensorFlow reserving all GPU memory.

## Data pipeline confirmed so far

1. Read Waymo TOP LiDAR first return (`return_index=0`).
2. Native range image: `[64, 2650, 4]` float32.
3. Range is channel 0; intensity is channel 1.
4. A valid pixel requires finite range, finite intensity, and `range > 0`.
5. Invalid pixels are stored as range = 0 and intensity = 0.
6. For 64→32 GT, use native rings `0, 2, …, 62`, yielding `[32, 2650, 2]`.
7. For the current 16→32 task, observed target rows are the even 32-grid rows
   `0, 2, …, 30`; generated target rows are `1, 3, …, 31`.
8. The Dataset circularly pads width 2650→2688.  Metrics crop to native width
   and mask only valid generated odd rows.

The current full 16→32 split is 160 train / 40 validation / 40 test frames.
All controlled results below use this split and exactly 40 validation or test
frames.

## Waymo adapter: what differs from original TULIP

The adapter is outside TULIP_backup:

`tulip_adapter/waymo_16_to_32_tulip.py`

- input: `[B, 2, 16, 2688]`
- output: `[B, 2, 32, 2688]`, then crop to width 2650
- model factory: `patch_size=(1,2)`, `window_size=(2,8)`, `in_chans=2`
- Waymo patch embedding: Conv kernel `(1,8)`, stride `(1,2)`, circular width
  padding 3 pixels on each side; embedded tensor `[B,16,1344,96]`
- adapter replaces only the input patch embedding and final anisotropic
  expansion with vertical scale 2 and horizontal scale 2.
- the backbone `decoder_pred` remains a 2-channel output layer.

This is not a modification to source files in TULIP_backup, but it **is** an
external adapter change to the input stem/final expansion.  The Transformer
body itself is not unchanged only by assertion: its parameters and modules are
reused as-is.  This distinction must be stated accurately in any report.

## Fixed controlled evaluation protocol

Use these settings when comparing current 16→32 models:

| Setting | Fixed value |
|---|---|
| Evaluation pixels | valid generated odd rows only |
| Range scale | 75.0 m |
| Intensity transform used by model | `log1p` |
| Reported intensity domains | raw and log1p |
| Raw negative intensity for metrics | clamp to zero only for metric calculation |
| Baselines | nearest-row and linear-row interpolation in raw domain |
| Train/validation/test | 160 / 40 / 40, no test-driven model selection |

`raw intensity MAE` is dominated by rare extremely large Waymo values.
`log1p MAE` measures the ordinary visual/dynamic range more robustly.  Both
must be reported, together with intensity-band metrics.

## Models evaluated

| Short name | Checkpoint | Intent |
|---|---|---|
| baseline | `outputs/waymo_16_to_32_controlled_baseline_s0/checkpoint.pth` | no high-intensity reweighting |
| moderate | `outputs/waymo_16_to_32_baseline_to_moderate_s0_e20/checkpoint.pth` | moderate tail-aware fine-tuning |
| band-balanced | `outputs/waymo_16_to_32_moderate_to_band_balanced_s0_e10/checkpoint.pth` | stronger per-band balance after moderate |

All three are 2-channel TULIP models with the same adapter/data shape.  They
are not interchangeable checkpoints for original KITTI TULIP.

## Main numerical results: held-out test, generated rows only

All values below use 1,491,970 valid generated pixels.

| Model | Range MAE [m] | Raw intensity MAE | log1p intensity MAE | Raw intensity MAE for target 0–1 |
|---|---:|---:|---:|---:|
| baseline | 0.973 | 17.400 | 0.0658 | **0.0851** |
| moderate | **0.959** | 17.203 | **0.0653** | 0.1046 |
| band-balanced | 0.965 | **16.948** | 0.0668 | 0.1285 |
| nearest interpolation | 2.771 | 18.525 | 0.0873 | 2.4018 |
| linear interpolation | 2.418 | 18.218 | 0.0818 | 2.2226 |

Interpretation:

- Range completion is clearly better than the two simple interpolation
  baselines for all three TULIP conditions.
- Ordinary intensity (`0–1`) represents about 99.0% of valid generated test
  pixels.  TULIP is much better than interpolation there.
- The baseline is best for ordinary raw intensity.  Moderate is best for
  range and total log1p error.  No condition is dominant for every goal.
- Raw intensity MAE remains large because rare values in the thousands to
  tens of thousands contribute very large absolute errors.

## High-reflection detection results: held-out test

A target pixel is positive when its **raw GT intensity** is at least the
threshold.  Precision answers “when the model predicts high, how often is it
correct?”  Recall answers “how many real high-intensity pixels are found?”

### Threshold >=1000

There are 1,768 positive target pixels (about 0.1185% of generated valid test
pixels).

| Model | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 277 | 65 | 1,491 | 0.810 | 0.157 | 0.263 |
| moderate | 361 | 93 | 1,407 | 0.795 | 0.204 | 0.325 |
| band-balanced | 487 | 118 | 1,281 | 0.805 | 0.275 | 0.410 |
| nearest interpolation | 749 | 574 | 1,019 | 0.566 | 0.424 | 0.485 |
| linear interpolation | 1,033 | 667 | 735 | 0.608 | 0.584 | 0.596 |

### Threshold >=100 and >=20 (band-balanced)

| Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|
| >=20 | 0.787 | 0.472 | 0.590 |
| >=100 | 0.813 | 0.464 | 0.591 |
| >=1000 | 0.805 | 0.275 | 0.410 |

Interpretation:

- Fine-tuning from baseline→moderate→band-balanced improves high-intensity
  recall while retaining roughly 80% precision at >=1000.
- The band-balanced result means 487 of 605 predicted >=1000 pixels are
  correct, but it still misses 1,281 of 1,768 true >=1000 pixels (72.5%).
- Therefore high-reflection recovery is **partially improved, not solved**.
- Linear interpolation finds more high points but produces far more false
  positives.  It currently has a better >=1000 F1 than TULIP.

## Why high reflection is difficult

1. High values are extremely rare: >=1000 is only about 0.12% of valid test
   pixels.
2. Values are spatially thin/local; the adjacent observed ring often lacks a
   corresponding high response.
3. `log1p` deliberately compresses extreme magnitudes, which stabilizes the
   main task but weakens sensitivity to the numerical difference between very
   large values.
4. Very strong loss weights previously improved recall partly but harmed range
   and ordinary-intensity quality.  A weight of 20 is not a safe final choice.

## Point-cloud / camera checks already completed

With the scaled 16→32 model, conversion from range image back to Waymo
vehicle-frame point clouds and camera projection have been smoke-tested.
For 20 held-out localization frames using the band-balanced model, FRONT
projection gave:

| Metric | Value |
|---|---:|
| common LiDAR pixels | 769,553 |
| oracle projected points | 84,219 |
| predicted projected points | 84,218 |
| oracle projection rate | 10.942% |
| predicted projection rate | 10.942% |
| oracle/predicted pixel shift mean | 0.130 px |
| oracle/predicted pixel shift p95 | 0.421 px |

This verifies that predicted ranges can be reconstructed and projected using
Waymo calibration.  It does **not** establish localization accuracy or LoFTR
matching improvement.

## LoFTR/PnP status (not current priority)

- The existing Shimizu LoFTR/PnP pipeline contract was inspected.
- A 1-pair oracle 32-line strict LoFTR/PnP smoke test succeeded using the
  existing static-map anchor representation.
- The current TULIP camera projection NPZ alone is insufficient for PnP: it
  lacks the static-map point-index/depth anchor representation required by
  the Shimizu evaluator.
- Before matching comparison, create a new TULIP-derived static map,
  reconstruct/fuse even observed rows with odd predicted rows, apply the same
  static/dynamic filtering, render anchor images/index maps, and create a
  new manifest.  Do not overwrite `/mnt/w/32line` assets.

This work is deferred until range/intensity conditions are fixed.

## Existing important artifacts

| Artifact | Path |
|---|---|
| baseline validation | `outputs/waymo_16_to_32_controlled_baseline_s0/validation40/summary.json` |
| baseline new test comparison | `outputs/waymo_16_to_32_controlled_baseline_s0/test40_20260916/summary.json` |
| moderate validation | `outputs/waymo_16_to_32_baseline_to_moderate_s0_e20/validation40/summary.json` |
| moderate new test comparison | `outputs/waymo_16_to_32_baseline_to_moderate_s0_e20/test40_20260916/summary.json` |
| band-balanced validation | `outputs/waymo_16_to_32_moderate_to_band_balanced_s0_e10/validation40/summary.json` |
| band-balanced test | `outputs/waymo_16_to_32_moderate_to_band_balanced_s0_e10/test40_20260915/summary.json` |
| GT/Low/Pred visualization example | `outputs/waymo_16_to_32_full_e300/pure_test_sample0/` |
| high-intensity experimental history | `docs/20260905_high_intensity_progress.md`, `docs/20260908_controlled_high_intensity_experiment.md`, `docs/20260909_band_balanced_intensity_experiment.md` |

## Open issues and priority

### P0 — decide high-reflection operating point without using test for tuning

Run a prediction-threshold calibration on **validation only** for the selected
tail-aware checkpoint.  Sweep the threshold used to call a pixel “predicted
high” (for example 300, 500, 750, 1000, 1500 in raw-intensity units), then
choose one criterion such as maximum F1 or a minimum precision requirement.
Freeze it before evaluating once on test.  This changes no model structure.

### P1 — choose a main 16→32 checkpoint according to the actual objective

- For best usual 0–1 intensity: baseline.
- For best range / overall log1p intensity: moderate.
- For more high-reflection recall at a small usual-intensity cost:
  band-balanced.

Do not claim one universally best model.  Select the condition before
downstream experiments and record why.

### P2 — inspect high-reflection spatial context

For true >=1000 pixels, quantify whether their two observed neighboring rows
also contain high intensity and whether missed examples are isolated single
pixels.  This determines whether missing signal is fundamentally absent from
the 16-line input or learnable from local context.

### P3 — only if P0/P2 justify it, make a minimal tail-specific experiment

Before adding model heads, try:

1. moderate band weights capped at 2–3;
2. sampling more training frames/patches that contain high-intensity targets;
3. residual intensity prediction relative to linear interpolation.

Do not add a classification head, new CNN, or new attention path unless these
minimal experiments fail and the change is explicitly authorized.  A
classification-plus-regression head would be a separate model experiment,
not the original TULIP configuration.

### P4 — scale data after the condition is frozen

The present 160/40/40 split demonstrates the pipeline but is small.  Before
claims of general performance, increase the number of **segments**, preserve
segment-level split isolation, and keep test untouched.

### P5 — point-cloud and matching stage

After freezing the model condition, generate fused 32-line point clouds:

`even observed 16 rows (raw measurement) + odd predicted 16 rows`

Then perform static-map generation, anchor rendering, and LoFTR/PnP comparison
under one common camera/normalization/matching protocol.

## Recommended immediate next action

Do P0 first: validation-only threshold calibration for `band-balanced`.
It is read-only with respect to the model and tells us whether the present
prediction distribution can trade precision for recall without retraining.
After freezing that decision, run its single test evaluation and record it in
this document or a new dated progress file.

## Warnings that are currently non-fatal

- PyTorch warning: `Initializing zero-element tensors is a no-op` is emitted
  during current model creation.  Forward/training/evaluation have still
  completed with finite outputs; record it but do not mistake it for a failed
  run.
- TensorFlow logs about AVX, TensorRT absence, NUMA, and placeholder debug
  messages have been non-fatal during Waymo frame/projection processing.
- `timm` and `chamfer_distance` errors occur when accidentally using
  `.venv_waymo`; activate `.venv_tulip` for model work.

