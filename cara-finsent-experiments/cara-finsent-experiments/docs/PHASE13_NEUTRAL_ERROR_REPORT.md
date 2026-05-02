# Phase 13 — Step 4 Neutral→Positive Error Mitigation Report

**Date:** 2026-05-02
**Git commit SHA:** `082c69df25d269f6dea1ab6a2543d38c055bbcd3`
**Script:** `scripts/37_neutral_error_mitigation.py`
**Run timestamp (canonical):** `20260502_141335`
**Source predictions:** `results/2026-05-02/finbert_agreement_weighted_predictions_20260502_082228.csv` (AW FinBERT, PhraseBank test, n = 959)

---

## 1. Motivation

Phase 8 error analysis reported neutral→positive as the dominant misclassification
mode for AW FinBERT, with high mean confidence (≈ 0.874) on these wrong
predictions. Decision-only post-hoc rule:

```
if prediction == 'positive' and (proba_positive - proba_neutral) < margin:
    prediction <- 'neutral'
```

Probability vectors are **not** modified — only the argmax decision label.
This keeps the calibration (ECE, Brier) of the underlying model intact and
isolates the effect of the override on classwise precision/recall.

---

## 2. Margin distribution diagnostic (motivates the wide sweep)

On AW positive predictions (n = 258), the (proba_positive − proba_neutral)
margin is extremely peaked:

| Quantile | margin (positive predictions) | margin (wrong positive) | margin (neutral→positive errors) |
|---|---:|---:|---:|
| 5 % | 0.395 | 0.237 | 0.236 |
| 25 % | 0.915 | 0.556 | 0.558 |
| 50 % | 0.980 | 0.848 | 0.880 |
| 75 % | 0.987 | 0.942 | 0.943 |
| 95 % | — | — | 0.979 |

**Implication:** the original margin grid {0.02 … 0.10} barely fires — only
1–3 cases trigger. Useful margins for AW lie in {0.20 … 0.60}. The reported
sweep was widened accordingly.

---

## 3. Sweep results (full PhraseBank test, n = 959)

Source CSV: `results/2026-05-02/phase13_neutral_mitigation_summary_20260502_141335.csv`
Safe copy: `artifacts/phase13_extended_validation/neutral_mitigation_summary_20260502_141335.csv`

| Margin | Overridden | Acc | macro-F1 | neutral_recall | positive_precision | neutral→positive errors | positive→neutral errors | overconf-wrong | ECE@10 | Brier |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 (baseline) | 0  | 0.8895 | 0.8860 | 0.9164 | 0.8372 | **41** | 46 | 72 | 0.0715 | 0.1799 |
| 0.05 | 2  | 0.8874 | 0.8840 | 0.9164 | 0.8359 | 41 | 48 | 72 | 0.0711 | 0.1799 |
| 0.10 | 3  | 0.8863 | 0.8830 | 0.9164 | 0.8353 | 41 | 49 | 72 | 0.0721 | 0.1799 |
| 0.20 | 7  | 0.8843 | 0.8807 | 0.9181 | 0.8367 | 40 | 52 | 72 | 0.0742 | 0.1799 |
| 0.30 | 11 | 0.8884 | 0.8841 | 0.9251 | 0.8502 | 36 | 52 | 72 | 0.0701 | 0.1799 |
| **0.40** | **14** | **0.8895** | **0.8847** | **0.9286** | **0.8566** | **34** | 53 | 72 | 0.0690 | 0.1799 |
| 0.50 | 19 | 0.8874 | 0.8828 | 0.9303 | 0.8619 | 33 | 56 | 72 | 0.0711 | 0.1799 |
| **0.60** | **28** | **0.8884** | **0.8828** | **0.9390** | **0.8783** | **28** | **60** | 72 | 0.0701 | 0.1799 |

---

## 4. Findings

### 4.1 Margin 0.40 — recommended safe operating point
- neutral→positive errors: **41 → 34 (−17 %)**.
- positive precision: 0.837 → **0.857 (+2.0 pp)**.
- neutral recall: 0.916 → **0.929 (+1.3 pp)**.
- macro-F1: 0.886 → 0.885 (essentially flat, well within ±1 σ for AW).
- Cost: positive→neutral errors rise 46 → 53 (+15 %).
- 14 overridden predictions (≈ 1.5 % of test set).

### 4.2 Margin 0.60 — aggressive operating point
- neutral→positive errors: **41 → 28 (−32 %)**.
- positive precision: 0.837 → **0.878 (+4.1 pp)**.
- neutral recall: 0.916 → **0.939 (+2.3 pp)**.
- macro-F1: 0.886 → 0.883 (−0.3 pp; still inside Phase 11 paired-bootstrap
  uncertainty for AW vs ZS, which had CI ≈ ±0.02).
- Cost: positive→neutral errors 46 → 60 (+30 %).
- 28 overridden (≈ 2.9 % of test set).

### 4.3 Calibration unchanged (as designed)
ECE@10 and Brier score are essentially unchanged across all margins (0.069 –
0.074) because probability vectors are not rescored. The small ECE wobble
reflects only bin reassignment of overridden cases.

### 4.4 Overconfidence is unaffected
Overconfident-wrong count stays at **72** at every margin. The override
re-labels cases but does not lower the model's underlying confidence on the
re-labelled predictions. To actually reduce overconfidence, combine this rule
with the per-class isotonic calibrator from Step 3.

---

## 5. Honest caveats

1. **Margin tuned on test data.** A research-grade deployment would tune the
   margin on a held-out val/calib split, not on the same test set used for
   reporting. The Phase 13 finding here is an "in-sample best margin" — it is
   illustrative, not a direct deployment recommendation.
2. **Probability vector unchanged.** ECE/Brier improvements are not part of
   this intervention. For a calibration improvement, see Step 3 (isotonic).
3. **Asymmetric trade-off.** Every neutral→positive correction adds ~1 false
   positive→neutral error. The intervention shifts the error mode rather than
   eliminating it. The recommendation is to use this only when neutral
   precision matters more than positive recall (e.g. risk-averse advisory
   systems).
4. **Single-checkpoint AW source.** The prediction CSV is one AW seed
   (`finbert_agreement_weighted_predictions_20260502_082228.csv`); the result
   has not been averaged across the 5 AW seeds.

---

## 6. Recommendation

- **Include in paper appendix** as a small post-hoc reliability lever, framed
  as "in-sample illustration; for deployment, combine with Step 3 isotonic
  calibration and tune the margin on a true val split".
- **Do not** retroactively re-score the headline AW macro-F1 with this rule.
  The headline AW number remains 0.8860 ± 0.00227 as published in Phase 11.
- **Combine with Step 3 (isotonic)** in the integrated Phase 13 report
  (`docs/PHASE13_EXTENDED_VALIDATION_AND_IMPROVEMENT_REPORT.md`, Step 9) as
  the recommended reliability bundle.

---

## 7. Artefacts written

| Artefact | Path |
|---|---|
| Sweep summary | `results/2026-05-02/phase13_neutral_mitigation_summary_20260502_141335.csv` |
| Manifest | `results/2026-05-02/phase13_neutral_mitigation_manifest_20260502_141335.json` |
| Safe summary copy | `artifacts/phase13_extended_validation/neutral_mitigation_summary_20260502_141335.csv` |
| Safe manifest copy | `artifacts/phase13_extended_validation/neutral_mitigation_manifest_20260502_141335.json` |

(An earlier narrow sweep `_20260502_141258.csv` is also on disk; the wider
sweep `_141335` supersedes it.)
