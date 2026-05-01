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
results/*_skipped_sources_*.csv
results/pipeline_step_report_*.csv
```

## What I will analyze after you send results

1. Which baseline is strongest.
2. Whether retrieval improves macro-F1 or neutral recall.
3. Whether structured features improve robustness.
4. Whether agreement-aware training helps ambiguous/low-agreement cases.
5. Whether calibration lowers ECE/Brier score.
6. Whether CARA-lite is stronger as a decision-grade system, even if raw F1 gains are modest.
7. What tables and figures should go into the IEEE paper.

## Phase 4 research-grade artifacts (ship these too)

```text
results/seed_sweep_summary_*.csv
results/seed_sweep_manifest_*.json
results/research_leaderboard_mean_std_*.csv
results/research_leaderboard_ci95_*.csv
results/research_best_model_summary_*.csv
results/calibration_summary_*.csv
results/calibration_reliability_bins_*.csv
results/abstention_curve_*.csv
results/error_analysis_summary_*.csv
results/error_examples_by_class_*.csv
results/neutral_confusion_analysis_*.csv
results/research_gate_report_*.csv
results/research_gate_manifest_*.json
data/audit/fiqa_*_audit_*.json
data/audit/retrieval_corpus_audit_*.csv
figures/research_macro_f1_mean_std_*.png
figures/research_accuracy_mean_std_*.png
figures/reliability_diagram_*.png
figures/abstention_curve_*.png
figures/error_confusion_heatmap_*.png
figures/neutral_error_breakdown_*.png
```

The `research_gate_report_*.csv` PASS/FAIL row set determines whether the
benchmark is ready for paper inclusion. Do not promote results to the paper
unless the gate prints `research_gate=PASS`.
