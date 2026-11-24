# Retrieval reproduction checkpoint (2026-10-19)

This file records the current investigation so the A/B comparison can be resumed later. It is an investigation checkpoint, not proof that the thesis's final model has been recovered. Do not modify the inherited `handover_shimizu` code or source Waymo data during this audit.

## Current priority: thesis-versus-C reproduction

- Thesis retrieval setup (section 3.4): dual ResNet34 ImageNet initialization, NetVLAD with 64 clusters, 256-dimensional embeddings, 512×864 input, modality-specific training-set mean/std, symmetric contrastive loss with temperature 0.12, AdamW LR 3e-4 / WD 1e-4, cosine schedule, 80 epochs, and validation selection described as Soft R@10. Published evaluation: 507 queries, 754 database anchors, five scenario pairs, global search.
- The local 80-epoch run used the same architecture, image size, optimizer settings, and temperature, per `/home/wakamatsu/ITS/waymo_outputs/old32_retrieval/train_wdrive_paper80_b16/args.json`.
- That run's `modality_stats.json` is invalid JSON (shell text follows line 18). Its training log explicitly reports fallback normalization of mean=0.5/std=0.5 for both camera and anchor. Evaluate its checkpoints using `configs/paper80_effective_modality_stats.json`, not the invalid source file or the nominal calculated values.
- The local run chose `best.ckpt` using `val1/R@10` (strict), not the thesis-described Soft(5 m) R@10. In the 881-query training log, strict R@10 peaked at epoch 67; Soft(5 m) R@10 peaked at epoch 58. Neither file is established as the thesis's original final checkpoint.
- The 507-query C pairs and 754-anchor C database match `outputs/shimizu_retrieval_protocol_507q_754db/shimizu_test4_{pairs,db}.csv` in row order, query paths, anchor paths, pair IDs, and anchor IDs. This establishes local-protocol identity, not identity to every original thesis artifact.
- The 80-epoch run's validation manifest has 881 queries and 1197 DB anchors and contains all 507 evaluation queries and 754 anchors. The remaining three pair IDs contain 374 queries and 443 anchors, with no exact query-path overlap with the 507-query set. Training pairs have zero exact camera/anchor-path overlap with the 507-query evaluation set. Checkpoint selection on 881 queries therefore used the evaluation queries; do not call the 507-query result an independent final test for that run.
- Additional segment-level audit: the 28,772 training rows cover 196 unique segments; the five evaluation pairs cover 10 segments and the remaining three validation pairs cover six. There is zero segment overlap among training, validation-only, and evaluation sets.
- Search of inherited `handover_shimizu` and `/mnt/w/32line/experiment` for retrieval/NetVLAD checkpoints did not reveal a separate, verified thesis-final checkpoint. This is not proof it does not exist elsewhere.

| Retrieval metric | Thesis | C with local best.ckpt | C with local epoch_058.ckpt |
| --- | ---: | ---: | ---: |
| Strict R@5 | 73.4% | 68.44% | 70.02% |
| Strict R@10 | 87.8% | 87.57% | 87.57% |
| Soft(2 m) R@5 | 85.0% | 87.18% | 87.38% |
| Soft(5 m) R@5 | 92.3% | 92.11% | 92.31% |
| Soft(5 m) R@10 | 97.8% | 96.06% | 95.66% |

Local result files:

- `outputs/paperlike_retrieval_C_507q/paperlike_C_507q_metrics.json`
- `outputs/paperlike_retrieval_C_epoch058_507q/paperlike_C_epoch058_507q_metrics.json`

The next safe work is a provenance audit of the thesis-final checkpoint and its validation records. If unavailable, design a new reproduction with a valid modality-statistics file and disjoint validation (the three remaining pairs) and evaluation (the five 507-query pairs). The inherited training code currently supports `--best-by R@10` but not direct Soft(5 m) R@10 selection; any change should be made in the TULIP project, not in inherited code. Do not select an epoch using the 507-query evaluation scores.

## Paused A/B investigation — resume here when requested

With local `best.ckpt` and identical 507-query protocol, A (observed 16-line) versus B (TULIP-upsampled) gave Strict R@5 59.96% versus 44.77%, and Soft(5 m) R@10 95.07% versus 84.02%. Query/GT alignment was checked. Compared with A, B lost 122 Strict@5 queries and gained 45; for Soft(2 m)@5 it lost 113 and gained 29. The drop is strongest in `segpair_641049__236730` and `segpair_848780__183339`. This establishes a relative retrieval drop for these rendered images under this model; it does not isolate geometry, intensity, or rendering as the cause.

Audit artifacts:

- `outputs/paperlike_retrieval_ABC_query_audit/summary.json`
- `outputs/paperlike_retrieval_ABC_query_audit/largest_rank_regressions.csv`
- `outputs/paperlike_retrieval_A_507q/paperlike_A_507q_metrics.json`
- `outputs/paperlike_retrieval_B_507q/paperlike_B_507q_metrics.json`

When resuming A/B: first inspect the same query and anchor images for large B rank regressions, then use one-factor-at-a-time image/rendering controls under a fixed retrieval model and protocol. Do not conflate Retrieval with LoFTR/PnP or treat a change in Top-K threshold as a fix for the underlying ranking.
