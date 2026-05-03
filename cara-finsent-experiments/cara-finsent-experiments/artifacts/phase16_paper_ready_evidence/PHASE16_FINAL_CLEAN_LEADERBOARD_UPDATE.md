# Phase 16 Final Clean Leaderboard Update

**File**: [results/2026-05-02/phase16_final_clean_leaderboard_20260503_110917.csv](../results/2026-05-02/phase16_final_clean_leaderboard_20260503_110917.csv)

## Changes vs. Phase 15

The Phase 15 leaderboard ([artifacts/phase15_paper_ready_evidence/phase15_final_clean_leaderboard.csv](../artifacts/phase15_paper_ready_evidence/phase15_final_clean_leaderboard.csv)) is preserved verbatim. Phase 16 **adds one new row** and changes none of the existing rows:

| change | row | rationale |
| --- | --- | --- |
| **added** | `finbert_agreement_weighted_kaggle_gpu_seed13_uncalibrated_phase16` (acc 0.8780, macro_F1 0.8695) | Kaggle Tesla P100 GPU reproduction at seed=13. Test macro_F1 = 0.8695 < Phase 11 envelope lower bound 0.8817 → AW number is **not reproducible** from the controlled gold split with the locked recipe, on either CPU or a Kaggle GPU. |
| unchanged | `finbert_agreement_weighted_phase11_gpu_envelope` (macro_F1 0.8863) | Retained as the *historical* Phase 11 envelope, but Phase 16 evidence now downgrades it from "paper headline AW number pending Kaggle reconfirmation" to "appendix-only / not reproducible from the controlled split" — see `PHASE16_FINAL_PAPER_CLAIM_BOUNDARIES.md`. |
| unchanged | all other rows | No re-runs were performed; no values change. |

## Headline numbers (Phase 16 view)

- **Best single model (paper-safe, uncalibrated)**: `finbert_fine_tuned_uncalibrated` — acc 0.8895, macro_F1 0.8884.
- **Best calibrated single model (val-fit)**: `finbert_fine_tuned_isotonic` — acc 0.8936, macro_F1 0.8958.
- **Best ensemble (val-fit weights)**: `ensemble_grid_val_aw0.40_zs0.60_ft0.00` — acc 0.8916, macro_F1 0.8953.
- **AW (paper-safe lower bound)**: `finbert_agreement_weighted_uncalibrated_cpu_retrain` 0.8648 / Kaggle-GPU seed=13 0.8695.
- **AW (Phase 11 envelope, NOT reproduced)**: 0.8863 ± 0.0023 — appendix-only.

The headline of the paper now rests on **fine-tuned FinBERT (with optional isotonic calibration on val)** rather than on the AW variant.
