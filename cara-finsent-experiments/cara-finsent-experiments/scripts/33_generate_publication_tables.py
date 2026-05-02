"""Phase 11 publication-grade table generator for CARA-FinSent.

Produces compact CSVs in paper_assets/tables/:
    table1_dataset_summary.csv        - dataset, splits, label distribution
    table2_model_leaderboard.csv      - leaderboard with best macro-F1 / ECE marked
    table3_calibration_abstention.csv - calibration + abstention tradeoff
    table4_statistical_validation.csv - paired comparisons summary
    table5_claim_boundary_matrix.csv  - claim vs supported vs caveat
    table6_ablation_summary.csv       - AW vs zero-shot vs fine-tuned ablation row

All numeric values are rounded to 4 decimals.

Usage:
    python scripts/33_generate_publication_tables.py \
        --results_dir results/2026-05-02 \
        --output_dir paper_assets/tables
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def round4(v):
    try:
        return round(float(v), 4)
    except Exception:
        return v


def latest(results_dir: Path, pattern: str) -> Path:
    files = sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(pattern)
    return files[-1]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def table1_dataset(results_dir: Path) -> pd.DataFrame:
    """Dataset / split / label distribution from any finbert baseline summary."""
    cand = sorted(results_dir.glob("finbert_baseline_summary_*.csv"))
    if not cand:
        return pd.DataFrame()
    df = pd.read_csv(cand[-1])
    row = df.iloc[0]
    train = json.loads(row["label_distribution_train"])
    val = json.loads(row["label_distribution_val"])
    test = json.loads(row["label_distribution_test"])
    out = pd.DataFrame([
        {
            "dataset": "Financial PhraseBank (controlled gold split)",
            "split": "train",
            "n": int(row["train_rows"]),
            "negative": train.get("negative", 0),
            "neutral": train.get("neutral", 0),
            "positive": train.get("positive", 0),
        },
        {
            "dataset": "Financial PhraseBank (controlled gold split)",
            "split": "val",
            "n": int(row["val_rows"]),
            "negative": val.get("negative", 0),
            "neutral": val.get("neutral", 0),
            "positive": val.get("positive", 0),
        },
        {
            "dataset": "Financial PhraseBank (controlled gold split)",
            "split": "test",
            "n": int(row["test_rows"]),
            "negative": test.get("negative", 0),
            "neutral": test.get("neutral", 0),
            "positive": test.get("positive", 0),
        },
    ])
    return out


def table2_leaderboard(results_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(latest(results_dir, "final_leaderboard_mean_std_*.csv"))
    df = df.sort_values("mean_macro_f1", ascending=False).reset_index(drop=True)
    df = df.rename(columns={
        "mean_macro_f1": "macro_f1_mean",
        "std_macro_f1": "macro_f1_std",
        "mean_accuracy": "accuracy_mean",
        "std_accuracy": "accuracy_std",
        "mean_ece_10_bins": "ece10_mean",
        "std_ece_10_bins": "ece10_std",
        "mean_brier_score": "brier_mean",
        "std_brier_score": "brier_std",
    })
    keep = [
        "experiment", "model", "n_seeds",
        "macro_f1_mean", "macro_f1_std",
        "accuracy_mean", "accuracy_std",
        "ece10_mean", "ece10_std",
        "brier_mean", "brier_std",
    ]
    out = df[keep].copy()
    for c in keep[3:]:
        out[c] = out[c].apply(round4)
    out["best_macro_f1"] = (out["macro_f1_mean"] == out["macro_f1_mean"].max()).map(
        {True: "*", False: ""})
    valid_ece = out["ece10_mean"].dropna()
    if len(valid_ece) > 0:
        best_ece = valid_ece.min()
        out["best_ece"] = (out["ece10_mean"] == best_ece).map({True: "*", False: ""})
    else:
        out["best_ece"] = ""
    return out


def table3_calibration_abstention(results_dir: Path) -> pd.DataFrame:
    cal = pd.read_csv(latest(results_dir, "calibration_summary_*.csv"))
    keep = [
        "model", "accuracy", "macro_f1", "ece_10_bins", "brier_score",
        "mean_confidence", "coverage_at_60", "accuracy_at_60", "macro_f1_at_60",
        "coverage_at_70", "accuracy_at_70", "macro_f1_at_70",
    ]
    keep = [c for c in keep if c in cal.columns]
    out = cal[keep].copy()
    for c in out.columns:
        if c != "model":
            out[c] = out[c].apply(round4)
    return out


def table4_statistical(results_dir: Path) -> pd.DataFrame:
    pairs = sorted(results_dir.glob("paired_model_comparison_*.csv"))
    if not pairs:
        return pd.DataFrame()
    df = pd.read_csv(pairs[-1])
    keep = [
        "model_a", "model_b", "delta_macro_f1_predictions",
        "bootstrap_ci_low", "bootstrap_ci_high",
        "p_value_or_empirical_probability",
        "mcnemar_statistic", "mcnemar_p_value",
        "delta_macro_f1_across_seeds", "cohens_d_across_seeds",
        "stability_std_a", "stability_std_b",
        "mean_ece_a", "mean_ece_b", "interpretation",
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    for c in out.columns:
        if c not in ("model_a", "model_b", "interpretation"):
            out[c] = out[c].apply(round4)
    return out


def table5_claim_matrix() -> pd.DataFrame:
    rows = [
        {
            "claim": "Agreement-weighted FinBERT achieves the highest mean macro-F1 in the in-domain Financial PhraseBank benchmark.",
            "supported": "yes",
            "evidence_file": "results/2026-05-02/final_leaderboard_mean_std_20260502_110922.csv",
            "strength": "moderate",
            "caveat": "Lift over zero-shot FinBERT is small (~0.003) and within paired bootstrap CI of zero.",
            "paper_wording": "AW FinBERT ranks first by mean macro-F1 (0.8863, n=5).",
        },
        {
            "claim": "Agreement weighting improves seed stability over vanilla fine-tuning.",
            "supported": "yes",
            "evidence_file": "results/2026-05-02/seed_sweep_summary_20260502_105326.csv",
            "strength": "strong",
            "caveat": "Stability gain estimated from n=5 seeds.",
            "paper_wording": "AW reduces across-seed std of macro-F1 from 0.00923 (vanilla FT) to 0.00227 (~4x more stable).",
        },
        {
            "claim": "Zero-shot FinBERT is the best-calibrated FinBERT variant.",
            "supported": "yes",
            "evidence_file": "results/2026-05-02/calibration_summary_20260502_081312.csv",
            "strength": "strong",
            "caveat": "Calibration measured by ECE@10 on PhraseBank test only.",
            "paper_wording": "Zero-shot FinBERT yields the lowest ECE@10 (0.0236), better than AW (0.0495) and FT (0.0608).",
        },
        {
            "claim": "Classical TF-IDF baselines remain much cheaper but materially less accurate.",
            "supported": "yes",
            "evidence_file": "results/2026-05-02/final_leaderboard_mean_std_20260502_110922.csv",
            "strength": "strong",
            "caveat": "Accuracy gap ~19 macro-F1 points; latency not formally benchmarked.",
            "paper_wording": "Best classical baseline (TF-IDF + linear SVM) reaches macro-F1 0.6941, ~19 points below the FinBERT family.",
        },
        {
            "claim": "Macro-F1 advantage of agreement-weighted FinBERT over zero-shot FinBERT is statistically decisive.",
            "supported": "no",
            "evidence_file": "results/2026-05-02/paired_model_comparison_*.csv",
            "strength": "weak",
            "caveat": "Paired bootstrap CI includes zero; McNemar p > 0.5; do NOT claim statistical superiority over zero-shot.",
            "paper_wording": "We do not claim a statistically decisive accuracy improvement over zero-shot FinBERT.",
        },
        {
            "claim": "Results generalize to live trading or real investment decisions.",
            "supported": "no",
            "evidence_file": "n/a",
            "strength": "n/a",
            "caveat": "Only in-domain PhraseBank evaluated; no live-market or PnL evidence.",
            "paper_wording": "All findings are in-domain on Financial PhraseBank; no live-trading utility is claimed.",
        },
    ]
    return pd.DataFrame(rows)


def table6_ablation(leaderboard: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in ("agreement_weighted", "finbert_zero_shot", "finbert_finetuned"):
        sel = leaderboard[leaderboard["experiment"] == variant]
        if sel.empty:
            continue
        r = sel.iloc[0]
        rows.append({
            "variant": variant,
            "model": r["model"],
            "n_seeds": r["n_seeds"],
            "macro_f1_mean": r["macro_f1_mean"],
            "macro_f1_std": r["macro_f1_std"],
            "ece10_mean": r["ece10_mean"],
            "ece10_std": r["ece10_std"],
            "brier_mean": r["brier_mean"],
            "brier_std": r["brier_std"],
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        ref = df[df["variant"] == "finbert_zero_shot"].iloc[0]
        df["delta_macro_f1_vs_zero_shot"] = (df["macro_f1_mean"] - ref["macro_f1_mean"]).apply(round4)
        df["delta_ece_vs_zero_shot"] = (df["ece10_mean"] - ref["ece10_mean"]).apply(round4)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/2026-05-02")
    ap.add_argument("--output_dir", default="paper_assets/tables")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t1 = table1_dataset(results_dir)
    t2 = table2_leaderboard(results_dir)
    t3 = table3_calibration_abstention(results_dir)
    t4 = table4_statistical(results_dir)
    t5 = table5_claim_matrix()
    t6 = table6_ablation(t2)

    pairs = [
        ("table1_dataset_summary.csv", t1),
        ("table2_model_leaderboard.csv", t2),
        ("table3_calibration_abstention.csv", t3),
        ("table4_statistical_validation.csv", t4),
        ("table5_claim_boundary_matrix.csv", t5),
        ("table6_ablation_summary.csv", t6),
    ]
    for name, df in pairs:
        path = out_dir / name
        df.to_csv(path, index=False)
        print(f"[OK] {path}  rows={len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
