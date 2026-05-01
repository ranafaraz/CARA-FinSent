# 08 · Evaluation and comparison

## Metrics

All metrics live in [`src/cara_finsent/metrics.py`](../src/cara_finsent/metrics.py).
Reported per model:

| Metric | Why we report it |
|---|---|
| `accuracy` | sanity check; not the headline because the test set is unbalanced |
| `macro_f1` | **headline metric** — averages F1 across negative/neutral/positive |
| `macro_precision`, `macro_recall` | diagnose precision/recall trade-offs |
| `weighted_f1` | matches what most pretrained models report |
| per-class precision / recall / F1 / support | written to `_perclass.csv` |
| confusion matrix | written to `_confusion.json` |
| `brier_score` (if `predict_proba` available) | calibration quality |
| `expected_calibration_error` (10-bin) | calibration quality |
| `runtime_seconds` | wall-clock per script |

We use **macro-F1** as the primary metric because the test set has a
moderate class imbalance (negative = 344, neutral = 483) and we don't
want a model to game the leaderboard by always predicting `neutral`.

## Per-source breakdowns

Script 17 (the comparison harness) also produces a per-source slice of
each model's macro-F1 across the test set. This is the single most
useful audit signal — e.g. a model that scores 0.85 on PhraseBank but
0.40 on FOMC is not a "good financial sentiment model", it has merely
memorised PhraseBank.

## Calibration

Two scalars per model:

* **Brier score** — mean squared error between predicted probability
  vector and one-hot truth.
* **Expected Calibration Error (ECE)** — bucket the predictions by
  confidence (10 equal-width bins), compare bin-mean confidence to
  bin-accuracy, weight by bin size. Reported in absolute terms.

Calibrated models (sigmoid / isotonic) are reported alongside the
uncalibrated ones in the same CSV so you can see both the metric shift
and the ECE shift.

## Comparison harness — script 17

[`scripts/17_compare_finbert_vs_classical.py`](../scripts/17_compare_finbert_vs_classical.py)
(present in the repo) does the following:

1. Pick the latest `<prefix>_<ts>.csv` per script in `results/`.
2. Stack them into one tall DataFrame.
3. Sort by `macro_f1` descending.
4. Write `results/comparison_<ts>.csv` and a per-class breakdown.
5. Render leaderboard plots into [`figures/`](../figures): a bar chart
   of macro-F1, a scatter of macro-F1 vs runtime, and a per-source
   heatmap.

Outputs (per run):

```
results/<YYYY-MM-DD>/comparison_<ts>.csv
results/<YYYY-MM-DD>/comparison_perclass_<ts>.csv
figures/leaderboard_<ts>.png
figures/runtime_vs_f1_<ts>.png
figures/per_source_heatmap_<ts>.png
```

## Reading the leaderboard

A defensible CARA-Lite story has, in order:

1. `majority_baseline` ≈ 0.20–0.25 macro-F1 (it predicts only neutral).
2. `tfidf_logistic_regression` ≈ 0.55–0.65 macro-F1.
3. `tfidf_logistic_regression + structured` ≈ +0.5–1.0 pt.
4. `tfidf + retrieval` ≈ +0.5–1.5 pt over (3).
5. `agreement-aware` ≈ comparable on macro-F1, better on minority recall.
6. `calibration` ≈ same accuracy, lower Brier / ECE.
7. `cara_lite_full` ≈ best classical, ~ FinBERT-zero-shot territory.
8. `finbert_finetuned` ≈ +5–10 pt macro-F1 over the best classical, at
   ~1000× the compute.

If your numbers come out very differently — especially if the gap
between (2) and (8) is small — re-check section 03; it almost always
means there's leakage.

## Sanity test

Whenever you generate a new comparison CSV, eyeball the per-class
breakdown for negatives. If `negative_recall < 0.30` for FinBERT,
something is wrong with class weighting (the upstream FinBERT head was
trained with a different class prior).
