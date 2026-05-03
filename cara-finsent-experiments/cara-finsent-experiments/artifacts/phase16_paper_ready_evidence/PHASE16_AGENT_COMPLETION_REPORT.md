# Phase 16 — Agent Completion Report

**Date (UTC)**: 2026-05-03
**Repo HEAD at start and end**: `6a999be6cbba48d5dbd3e08021fddcb78ed4dfe7` (no commits made by the agent)

## 10 spec questions

1. **Final commit SHA**: `6a999be6cbba48d5dbd3e08021fddcb78ed4dfe7` — unchanged from Phase 15. Phase 16 added documents, results, and an evidence package, but committing them is left to the user.
2. **Did Kaggle work?** **Yes — the infrastructure worked.** Dataset `ranafarazahmed/cara-finsent-phase16-aw-bundle` uploaded successfully; script kernel `ranafarazahmed/cara-finsent-phase-16-aw-seed-13-script` (v5) trained on a Tesla P100-PCIE-16GB for 139.7 s, generated val and test predictions, and wrote the metrics CSV. Several earlier kernel versions failed for unrelated reasons (papermill expected a notebook; sm_60 / preinstalled torch incompatibility) and were resolved by switching to `kernel_type=script` and pinning `torch==2.4.1+cu121` in a clean subprocess.
3. **Did the agent fall back to RunPod?** **No.** Per the user's instruction ("if failed to use kaggle then exit process and let me know") and the plan's hard stop on "AW seed 13 fails gate", the agent did not auto-launch RunPod. The user has stated they will run RunPod manually if/when needed.
4. **Was the AW result reproduced on Kaggle GPU?** **No.** Test macro_F1 = **0.8695** at seed=13 (val 0.8978), versus the Phase 11 envelope `0.8863 ± 0.0023` (95% CI [0.8817, 0.8909]). Decision: **`FAIL_NOT_REPRODUCED_ON_KAGGLE_GPU`** ([docs/PHASE16_AW_GPU_REPRODUCTION_REPORT.md](PHASE16_AW_GPU_REPRODUCTION_REPORT.md)).
5. **Best single model (paper-safe)**: `finbert_fine_tuned_uncalibrated` — acc 0.8895, macro_F1 0.8884.
6. **Best calibrated single model (paper-safe)**: `finbert_fine_tuned_isotonic` (val-fit isotonic) — acc 0.8936, macro_F1 0.8958. Best ensemble is `ensemble_grid_val_aw0.40_zs0.60_ft0.00` — acc 0.8916, macro_F1 0.8953.
7. **AW status**: The Phase 11 AW envelope (0.8863 ± 0.0023) is **not reproducible from the controlled gold split with the locked recipe** on either CPU (Phase 14: 0.8648) or Kaggle Tesla P100 GPU (Phase 16, seed=13: 0.8695). AW is downgraded to appendix-only / not headline.
8. **Safe-to-claim items** are listed in [docs/PHASE16_FINAL_PAPER_CLAIM_BOUNDARIES.md](PHASE16_FINAL_PAPER_CLAIM_BOUNDARIES.md) §A: fine-tuned and zero-shot FinBERT numbers, isotonic calibration improving ECE, the convex ensemble (val-fit), classical baselines, and the controlled-split reproducibility methodology.
9. **Unsafe-to-claim items** ([…CLAIM_BOUNDARIES.md](PHASE16_FINAL_PAPER_CLAIM_BOUNDARIES.md) §C): "AW achieves macro_F1 ≈ 0.886" framed as a state-of-the-art claim; calibrated AW as a headline; any AW-vs-FT gain on the controlled split.
10. **ChatGPT folder / single source of evidence**: [artifacts/phase16_paper_ready_evidence/](../artifacts/phase16_paper_ready_evidence/) (15 files) + manifest [artifacts/phase16_paper_ready_evidence_manifest.json](../artifacts/phase16_paper_ready_evidence_manifest.json) with SHA-256, byte sizes, git commit, and UTC timestamp for each file.

## Where things landed

- **Phase 16 GPU artefacts**: [results/2026-05-02/phase16_aw_gpu_repro/](../results/2026-05-02/phase16_aw_gpu_repro/) (6 files: env, train log, val/test predictions, metrics CSV, manifest JSON).
- **Updated leaderboard**: [results/2026-05-02/phase16_final_clean_leaderboard_20260503_110917.csv](../results/2026-05-02/phase16_final_clean_leaderboard_20260503_110917.csv) — Phase 15 rows preserved, one new AW Kaggle-GPU row appended; described in [docs/PHASE16_FINAL_CLEAN_LEADERBOARD_UPDATE.md](PHASE16_FINAL_CLEAN_LEADERBOARD_UPDATE.md).
- **Audits / supporting docs**: [docs/PHASE16_REPO_HEAD_AUDIT.md](PHASE16_REPO_HEAD_AUDIT.md), [docs/PHASE16_KAGGLE_PACKAGE_AUDIT.md](PHASE16_KAGGLE_PACKAGE_AUDIT.md), [docs/PHASE16_AW_GPU_REPRODUCTION_REPORT.md](PHASE16_AW_GPU_REPRODUCTION_REPORT.md), [docs/PHASE16_FINAL_PAPER_CLAIM_BOUNDARIES.md](PHASE16_FINAL_PAPER_CLAIM_BOUNDARIES.md).
- **Kaggle assets** (already live, can be re-pushed): dataset slug `ranafarazahmed/cara-finsent-phase16-aw-bundle`; kernel slug `ranafarazahmed/cara-finsent-phase-16-aw-seed-13-script` (script kernel, v5).
- **Local kernel source**: [artifacts/phase16_aw_kaggle/kernel/phase16_aw_seed13.py](../artifacts/phase16_aw_kaggle/kernel/phase16_aw_seed13.py) and [artifacts/phase16_aw_kaggle/kernel/kernel-metadata.json](../artifacts/phase16_aw_kaggle/kernel/kernel-metadata.json).

## Recommended next actions for the user

1. (Optional, manual) Launch RunPod with the same kernel source (`artifacts/phase16_aw_kaggle/kernel/phase16_aw_seed13.py` + the bundled dataset) on an A100/H100 to triple-confirm the Phase 16 finding. If RunPod also produces test macro_F1 < 0.8817, the AW downgrade is final.
2. Adopt fine-tuned FinBERT (uncalibrated for headline; isotonic-on-val for the calibrated headline) as the paper's main result; relegate AW to an appendix paragraph framed as a reproducibility study.
3. Commit the new docs, the leaderboard CSV under `results/2026-05-02/`, the GPU artefact folder, and the `artifacts/phase16_paper_ready_evidence*` files.
