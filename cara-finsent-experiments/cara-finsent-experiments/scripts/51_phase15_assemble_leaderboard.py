#!/usr/bin/env python3
"""Phase 15 Task 6 — assemble final clean leaderboard from Phase 14 evidence."""
from __future__ import annotations
import csv
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Numbers sourced from (all val-fit / test-eval, no leakage):
#  - results/2026-05-02/phase14_calibration/clean_calibration_summary_zs_20260502_150835.csv
#  - results/2026-05-02/phase14_calibration/clean_calibration_summary_ft_20260502_151335.csv
#  - results/2026-05-02/phase14_calibration/clean_calibration_summary_aw_20260502_165724.csv
#  - results/2026-05-02/phase14_ensemble/clean_ensemble_summary_20260502_165749.csv
#  - results/2026-05-02/classical_baseline_summary_*.csv (latest)
ROWS = [
    # model_name, source, n_test, accuracy, macro_f1, ece_10_bins, brier, calibration_method, fit_split, notes
    ("majority_class_baseline", "classical_baseline_summary_20260502_081109.csv", 959, 0.5985, 0.2496, 0.4015, 0.8029, "none", "none", "trivial baseline"),
    ("tfidf_linear_svm", "classical_baseline_summary_20260502_081109.csv", 959, 0.7497, 0.6941, None, None, "none", "none", "no probabilities (LinearSVC)"),
    ("finbert_zero_shot_uncalibrated", "phase14_calibration/clean_calibration_summary_zs_20260502_150835.csv", 959, 0.8832, 0.8836, 0.0236, 0.1762, "none", "none", "ZS = ProsusAI/finbert without fine-tuning"),
    ("finbert_zero_shot_isotonic", "phase14_calibration/clean_calibration_summary_zs_20260502_150835.csv", 959, 0.9030, 0.9038, 0.0165, 0.1518, "isotonic", "val", "best-calibrated ZS variant"),
    ("finbert_fine_tuned_uncalibrated", "phase14_calibration/clean_calibration_summary_ft_20260502_151335.csv", 959, 0.8895, 0.8884, 0.0344, 0.1577, "none", "none", "FT on PhraseBank gold train"),
    ("finbert_fine_tuned_isotonic", "phase14_calibration/clean_calibration_summary_ft_20260502_151335.csv", 959, 0.8936, 0.8958, 0.0138, 0.1628, "isotonic", "val", "best-calibrated FT variant"),
    ("finbert_agreement_weighted_uncalibrated_cpu_retrain", "phase14_calibration/clean_calibration_summary_aw_20260502_165724.csv", 959, 0.8749, 0.8648, 0.0418, 0.1812, "none", "none", "Phase 14 CPU retrain seed=13; below Phase 11 GPU envelope [0.8817, 0.8909] by ~0.02 macro-F1; treat as lower bound; see PHASE15_AW_REPRODUCIBILITY_DIAGNOSIS.md"),
    ("finbert_agreement_weighted_platt_cpu_retrain", "phase14_calibration/clean_calibration_summary_aw_20260502_165724.csv", 959, 0.8822, 0.8813, 0.0385, 0.1707, "platt", "val", "Platt is the val-best calibrator for AW; same lower-bound caveat as above"),
    ("ensemble_grid_val_aw0.40_zs0.60_ft0.00", "phase14_ensemble/clean_ensemble_summary_20260502_165749.csv", 959, 0.8916, 0.8953, 0.0274, 0.1595, "softmax_mean_weighted", "val", "best of {simple-mean, a-priori, grid step 0.10, stacked LR}; weights tuned on val only"),
]

# Phase 11 GPU envelope (multi-seed) for AW — reported as a separate reference row
ROWS.append((
    "finbert_agreement_weighted_phase11_gpu_envelope", "PHASE11_RESULTS_NARRATIVE.md", 959, None, 0.8863, None, None, "none", "n/a",
    "Phase 11 5-seed mean +/- 0.0023; envelope [0.8817, 0.8909]; remains paper headline AW number pending Kaggle GPU reconfirmation (see artifacts/phase15_aw_kaggle/)",
))


def main():
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / "2026-05-02"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase15_final_clean_leaderboard_{ts}.csv"
    fields = [
        "model_name", "source_csv", "n_test", "accuracy", "macro_f1",
        "ece_10_bins", "brier_score", "calibration_method", "fit_split", "notes",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for r in ROWS:
            w.writerow(r)
    print(f"[OUT] {out_path}")


if __name__ == "__main__":
    main()
