# Phase 13 — Step 3 AW Post-hoc Calibration Report

**Date:** 2026-05-02
**Git commit SHA:** `082c69df25d269f6dea1ab6a2543d38c055bbcd3`
**Script:** `scripts/36_calibrate_aw_finbert.py`
**Run timestamp:** `20260502_141058`
**Source predictions:** `results/2026-05-02/finbert_agreement_weighted_predictions_20260502_082228.csv` (PhraseBank test, n = 959)

---

## 1. Methodology

Three post-hoc calibration methods were applied to Agreement-Weighted FinBERT
probability outputs:

1. **Temperature scaling** — single scalar `T` minimising NLL on the calib set,
   applied to `softmax(log(p)/T)` on the eval set.
2. **Platt-style multinomial LR** — `sklearn.LogisticRegression` (lbfgs,
   `C = 1.0`, max_iter 2000) trained on calib probabilities → labels.
3. **Per-class isotonic regression** — one-vs-rest isotonic on calib max-prob
   per class, with row-renormalisation to a valid distribution.

### Why a test-internal split (caveat)

AW FinBERT was trained on Kaggle and only test-split predictions are on disk.
No AW val-split probabilities exist locally. Per the user's Phase 13 decision,
the AW test set (n = 959) was **stratified-split** into:

- Calib portion: n = 479 (50.0%, used to fit calibrators)
- Eval portion:  n = 480 (50.0%, used to report all post-calibration metrics)

`random_state = 42`, stratified on the gold label. Both portions are disjoint.
This substitutes for a true val split; results are honest but the calibrator's
generalisation to unseen domain is not separately bounded here. Documented
in every artefact via `caveat_no_val_split`.

### Why log(probs) is treated as logits for temperature scaling

Raw logits were not stored at AW inference time (only normalised probabilities
were saved). We therefore work with `log(p)` as a logit surrogate. For a
softmax model this is equivalent up to an additive per-row constant, which the
softmax invariant absorbs — so the temperature fit is exact.

---

## 2. Headline results (eval split, n = 480)

Source CSV: `results/2026-05-02/phase13_aw_calibration_comparison_20260502_141058.csv`
Safe copy: `artifacts/phase13_extended_validation/aw_calibration_comparison_20260502_141058.csv`

| Method | Accuracy | macro-F1 | **ECE@10** | Brier | mean conf |
|---|---:|---:|---:|---:|---:|
| AW uncalibrated (eval only) | 0.898 | 0.894 | 0.0646 | 0.163 | 0.954 |
| AW + temperature scaling | 0.898 | 0.894 | 0.0533 | 0.152 | 0.891 |
| AW + Platt multinomial LR | 0.888 | 0.882 | 0.0665 | 0.168 | 0.874 |
| **AW + per-class isotonic** | **0.885** | **0.878** | **0.0275** | **0.150** | 0.873 |

For reference (full-test baseline pre-calibration on n = 959):
- AW uncalibrated full test → macro-F1 = 0.886, ECE@10 = 0.0715.

---

## 3. Key findings

### 3.1 Per-class isotonic is the clear winner on calibration
- ECE@10 drops from **0.0646 → 0.0275** (≈ **57.5 % relative reduction**).
- Brier drops from 0.163 → 0.150 (≈ 8 % relative reduction).
- Mean confidence drops from 0.954 → 0.873 (overconfidence is partially fixed).
- Cost: macro-F1 drops from 0.894 → 0.878 (≈ −1.6 pp absolute), accuracy
  drops 1.3 pp. This is a small reliability-vs-sharpness trade.

### 3.2 Temperature scaling is a safe near-free improvement
- ECE drops 0.065 → 0.053 (≈ 17 % reduction).
- macro-F1 and accuracy unchanged (T-scaling preserves argmax ordering).
- If the paper reports a single calibrator for AW, this is the conservative
  default.

### 3.3 Platt-style LR underperforms in this regime
- ECE essentially unchanged (0.066 vs 0.065 baseline).
- Slight macro-F1 hit (-1.2 pp). Not recommended for AW.

### 3.4 AW + isotonic vs. ZS uncalibrated (the Phase 11 ECE leader)
| Model | macro-F1 | ECE@10 |
|---|---:|---:|
| Zero-shot FinBERT (Phase 11 baseline) | 0.884 | **0.0236** |
| **AW + per-class isotonic (this work)** | **0.878** | 0.0275 |

AW + isotonic now matches the **best-calibrated** baseline (Δ ECE = 0.004
absolute) while preserving AW's seed-stability advantage (std macro-F1 0.00227
vs 0.00923 for vanilla FT — see Phase 11). This is the most useful
reliability-side outcome of Phase 13.

### 3.5 Abstention curve (for confidence-based deferral)
Source: `results/2026-05-02/phase13_aw_calibration_abstention_20260502_141058.csv`.
At threshold 0.90 (typical operating point):
- Uncalibrated keeps ≈ 86 % of inputs at 0.94 accuracy.
- Isotonic-calibrated keeps fewer inputs but with sharper accuracy gradient,
  i.e. the abstention signal is more trustworthy.

---

## 4. Reliability-diagram figure

`figures/2026-05-02/phase13_aw_calibration_reliability_20260502_141058.png`

Four panels (Uncalibrated, Temperature, Platt, Isotonic) show empirical
accuracy vs predicted confidence bins (10 equal-width bins, marker size ∝ √n).
Isotonic is visibly the closest to the y = x diagonal across the high-confidence
bins where AW's overconfidence concentrated.

---

## 5. Recommended use

| Setting | Recommendation |
|---|---|
| Default reporting in the paper | AW + per-class isotonic (best ECE) **and** AW + temperature scaling (no accuracy cost). Report both. |
| Production / API-style serving | AW + temperature scaling — preserves argmax decisions; lowest implementation risk. |
| Confidence-based abstention systems | AW + per-class isotonic — most honest tail behaviour. |
| Direct claim that AW beats ZS on calibration | **Do not claim**. With isotonic, AW closes the gap to ZS on ECE (0.028 vs 0.024) but does not surpass it. The honest framing is "match" not "beat". |

---

## 6. Artefacts written

| Artefact | Path |
|---|---|
| Method-comparison summary | `results/2026-05-02/phase13_aw_calibration_comparison_20260502_141058.csv` |
| Abstention curve | `results/2026-05-02/phase13_aw_calibration_abstention_20260502_141058.csv` |
| Manifest | `results/2026-05-02/phase13_aw_calibration_manifest_20260502_141058.json` |
| Reliability figure | `figures/2026-05-02/phase13_aw_calibration_reliability_20260502_141058.png` |
| Safe summary copy | `artifacts/phase13_extended_validation/aw_calibration_comparison_20260502_141058.csv` |
| Safe abstention copy | `artifacts/phase13_extended_validation/aw_calibration_abstention_20260502_141058.csv` |
| Safe manifest copy | `artifacts/phase13_extended_validation/aw_calibration_manifest_20260502_141058.json` |

---

## 7. Caveats explicitly recorded in every artefact

1. AW val-split probabilities are unavailable on disk; calibrators were fit
   on a stratified split of the AW test predictions (50/50, seed 42).
2. Temperature scaling uses `log(p)` as a logit surrogate (mathematically
   equivalent for softmax models).
3. All reported metrics are on the held-out 480-row eval portion only — they
   are **not** comparable line-for-line with the Phase 11 full-test (n = 959)
   AW row in `final_leaderboard_mean_std_20260502_110922.csv`.
