# Phase 15 — Final Clean Leaderboard

**Generated (UTC):** 2026-05-02
**Source CSV:** [results/2026-05-02/phase15_final_clean_leaderboard_20260502_190410.csv](../results/2026-05-02/phase15_final_clean_leaderboard_20260502_190410.csv)
**Test set:** PhraseBank gold split, n=959 (controlled gold split, 3353/479/959).
**No test data was used to fit any calibrator, ensemble weight, or threshold.**

## 1. Leaderboard

| # | Model | Acc | macro-F1 | ECE₁₀ | Brier | Calibration | Fit split | Notes |
|---|---|---:|---:|---:|---:|---|---|---|
| 1 | majority_class_baseline | 0.5985 | 0.2496 | 0.4015 | 0.8029 | none | none | trivial baseline |
| 2 | tfidf_linear_svm | 0.7497 | 0.6941 | — | — | none | none | LinearSVC, no calibrated probabilities |
| 3 | finbert_zero_shot_uncalibrated | 0.8832 | 0.8836 | 0.0236 | 0.1762 | none | none | ProsusAI/finbert without fine-tuning |
| 4 | **finbert_zero_shot_isotonic** | **0.9030** | **0.9038** | **0.0165** | 0.1518 | isotonic | val | best-calibrated ZS variant |
| 5 | finbert_fine_tuned_uncalibrated | 0.8895 | 0.8884 | 0.0344 | 0.1577 | none | none | FT on PhraseBank gold train |
| 6 | finbert_fine_tuned_isotonic | 0.8936 | 0.8958 | **0.0138** | 0.1628 | isotonic | val | lowest ECE on test |
| 7 | finbert_agreement_weighted_uncalibrated_cpu_retrain | 0.8749 | 0.8648 | 0.0418 | 0.1812 | none | none | Phase 14 CPU retrain seed=13; **lower bound** vs Phase 11 GPU envelope (see row 10) |
| 8 | finbert_agreement_weighted_platt_cpu_retrain | 0.8822 | 0.8813 | 0.0385 | 0.1707 | platt | val | best calibrator for AW; same lower-bound caveat |
| 9 | **ensemble_grid_val_AW0.40_ZS0.60_FT0.00** | 0.8916 | **0.8953** | 0.0274 | 0.1595 | softmax-mean weighted | val | best of {simple-mean, a-priori, grid step 0.10, stacked LR}; weights tuned on val only |
| 10 | finbert_agreement_weighted_phase11_gpu_envelope | — | **0.8863 ± 0.0023** | — | — | — | — | Phase 11 5-seed GPU mean; envelope [0.8817, 0.8909]; **remains paper-headline AW number** pending Kaggle GPU reconfirmation |

(Bold = best in column / paper-headline.)

## 2. Reading guide

- **Best-performing single model on test macro-F1:** `finbert_zero_shot_isotonic` at 0.9038.
  This is a single-checkpoint, post-hoc-calibrated ZS FinBERT — no fine-tuning, no AW training.
- **Best-performing system on test macro-F1:** `ensemble_grid_val_AW0.40_ZS0.60_FT0.00` at 0.8953.
  The val-tuned weights coincidentally drop FT entirely; the gain over ZS-isotonic alone is ≈ −0.009 macro-F1, so on this metric the ensemble does **not** beat a single calibrated ZS checkpoint.
- **Best calibration (lowest ECE):** `finbert_fine_tuned_isotonic` at 0.0138, followed by `finbert_zero_shot_isotonic` at 0.0165.
- **Agreement-weighted training:** the CPU-retrain variant under-performs the Phase 11 GPU envelope by ~0.02 macro-F1; see [PHASE15_AW_REPRODUCIBILITY_DIAGNOSIS.md](PHASE15_AW_REPRODUCIBILITY_DIAGNOSIS.md). All AW rows in the table are therefore reported as lower bounds. The paper headline AW number stays at the Phase 11 multi-seed envelope `0.8863 ± 0.0023`, conditional on the pending Kaggle GPU reconfirmation (`artifacts/phase15_aw_kaggle/`).

## 3. What this leaderboard is — and is not

**Is:**
- A val-fit / test-eval comparison across one in-domain test set (PhraseBank, n=959).
- A reliability-aware view: every row reports calibration error and Brier score alongside accuracy / macro-F1.
- The single source of truth for the Phase 15 paper draft.

**Is not:**
- A SOTA claim. There is no statistically significant evidence that any FinBERT variant in the table dominates the others on macro-F1 within the Phase 11 paired-bootstrap CIs.
- A cross-domain ranking — see [PHASE15_EXTERNAL_VALIDATION_CLAIM_BOUNDARIES.md](PHASE15_EXTERNAL_VALIDATION_CLAIM_BOUNDARIES.md).
- A multi-seed comparison at this Phase 14 retrain — only the Phase 11 GPU envelope (row 10) is multi-seed.

## 4. Provenance

All numbers in rows 3–9 were re-verified in Phase 15 Task 3 by re-running [scripts/42_clean_calibration_eval.py](../scripts/42_clean_calibration_eval.py) on the same Phase 14 prediction CSVs; outputs landed in `results/2026-05-02/phase15_calibration_check/` and matched the Phase 14 numbers to all printed digits.

Row 10 is taken from `docs/PHASE11_RESULTS_NARRATIVE.md`.
