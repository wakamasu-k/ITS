# Retrieval Top-10 -> LoFTR/PnP protocol audit

## Thesis evidence

Source: `/mnt/d/Users/wakamatsu.k/Desktop/shimizu.k.pdf`, extracted text `/tmp/shimizu_k.txt`.

- Section 3.4.5 says K=10 is the basic candidate count (text lines 1132–1133).
- Section 2.1.2.3, equation (2.6), selects the candidate with maximum RANSAC inliers.
- Section 3.5.1 explicitly says integration/selection performance is not evaluated there; isolated image pairs are evaluated. Consequently this integrated run implements the proposed pipeline but is not reproduction of a published integrated metric table.
- Section 3.5.4.1 restricts pose-error statistics to PnP-successful pairs. An explicit integrated all-candidates-failed rule and full-query denominator were not established from these sections.

## Local integrated evaluation rules

- Retain exactly the saved Top-10 candidates per query, including distant/wrong-pair candidates. No GT filtering or GT candidate selection.
- Prefer PnP-successful candidates, then maximum inliers. Tie breakers inherited from the evaluator: smaller reprojection median, larger valid-3D count. Complete ties use Retrieval rank, explicitly a local deterministic convention.
- All candidates failing yields no pose and remains in the denominator.
- If a wrong scene is selected, count a location failure; do not invent a metric distance between unrelated coordinate frames.
- Report PnP-return rate separately from location accuracy. Location thresholds use all 507 queries. Median XY error uses selected successful, same-pair, comparable, finite-error poses and reports this subset's count.
- Disable GT-derived z alignment. Use XY errors and the known per-scene ICP registration for comparable pairs.
- Ignore inherited `summary.json` success thresholds for the integrated report: that function discards nonfinite errors before taking the mean. Use `integrated_summary.json` instead.

## Fixed local model baseline

Input rankings: `outputs/paperlike_retrieval_C_507q/paperlike_C_507q_topk_top50.jsonl`, previously computed with local Retrieval `best.ckpt` (Strict validation selection) and effective normalization 0.5/0.5. This is not the verified thesis-final Retrieval model.

LoFTR: `stage1_final_gstep715958.pth`, CUDA fp16, inherited evaluator, no additional training. Stage-2 is not reproduced by this run.

Known training discrepancies already established: normalization fallback to 0.5/0.5; Strict R@10 rather than Soft(5m) R@10 checkpoint selection; 881-query validation included the 507 evaluation queries. New training with a separate 374-query/443-DB validation set can address these; changing evaluation alone cannot retroactively fix checkpoint selection.

Runner: `tools/evaluate_retrieval_topk_pipeline.py` (prepare, run, summarize).
Pilot: first and middle query in each of the five pairs, independent of outcomes (10 queries / 100 candidates).
Full: 507 queries / 5070 candidates.

Summary behavior tested with synthetic cases covering wrong-pair maximum-inlier selection, complete PnP failure, and the full-query denominator. GPU execution is left to the user's WSL environment, where CUDA is available.
