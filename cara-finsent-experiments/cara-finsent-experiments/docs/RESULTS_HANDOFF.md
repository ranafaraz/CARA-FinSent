# Results Handoff Guide

After executing experiments, send back the `results/` folder and, if possible, the `figures/` folder.

## Minimum files needed for comparison

```text
results/classical_baseline_summary_*.csv
results/structured_features_summary_*.csv
results/retrieval_experiment_summary_*.csv
results/agreement_aware_summary_*.csv
results/calibration_summary_*.csv
results/cara_lite_summary_*.csv
```

## Stronger handoff

Also include:

```text
results/*_predictions_*.csv
results/*_confusion_matrix_*.csv
results/*_classwise_metrics_*.csv
results/*_reliability_bins_*.csv
results/*_abstention_curve_*.csv
figures/*.png
```

## What I will analyze after you send results

1. Which baseline is strongest.
2. Whether retrieval improves macro-F1 or neutral recall.
3. Whether structured features improve robustness.
4. Whether agreement-aware training helps ambiguous/low-agreement cases.
5. Whether calibration lowers ECE/Brier score.
6. Whether CARA-lite is stronger as a decision-grade system, even if raw F1 gains are modest.
7. What tables and figures should go into the IEEE paper.
