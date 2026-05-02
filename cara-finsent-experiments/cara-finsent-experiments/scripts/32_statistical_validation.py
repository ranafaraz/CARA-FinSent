"""Phase 11 statistical validation for CARA-FinSent.

Computes:
- Per-model bootstrap confidence intervals for macro-F1 (across-seed bootstrap).
- Paired bootstrap comparisons (AW vs zero-shot, AW vs fine-tuned, zero-shot vs fine-tuned).
- McNemar test on prediction-level matched outputs for available seed pairs.
- Effect size estimate (Cohen's d-like) for macro-F1 gap.
- Stability comparison via per-seed std.

Outputs (timestamped, never overwriting):
    results/<date>/statistical_validation_summary_<ts>.csv
    results/<date>/paired_model_comparison_<ts>.csv
    figures/statistical_validation_macro_f1_ci_<ts>.png

Inputs (auto-discovered from results/<date>/):
    seed_sweep_summary_*.csv (concatenated; latest by mtime per (model, seed))
    finbert_baseline_summary_*.csv + finbert_baseline_predictions_*.csv
    finbert_agreement_weighted_summary_*.csv + finbert_agreement_weighted_predictions_*.csv

Usage:
    python scripts/32_statistical_validation.py --results_dir results/2026-05-02 \
        --figures_dir figures --bootstrap_n 2000 --seed 42
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import math
import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - figures optional
    plt = None  # type: ignore

from sklearn.metrics import f1_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LABELS = ["negative", "neutral", "positive"]


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))


def bootstrap_macro_f1_ci(
    y_true: np.ndarray, y_pred: np.ndarray, n: int, rng: np.random.Generator
) -> tuple[float, float]:
    n_obs = len(y_true)
    if n_obs == 0:
        return float("nan"), float("nan")
    boots = np.empty(n, dtype=float)
    for i in range(n):
        idx = rng.integers(0, n_obs, size=n_obs)
        boots[i] = macro_f1(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def paired_bootstrap_delta_macro_f1(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """Returns (delta, ci_low, ci_high, empirical_p_two_sided).

    delta = macro_f1(a) - macro_f1(b) on the full sample.
    Empirical p is 2 * min(P(boot_delta <= 0), P(boot_delta >= 0)).
    """
    n_obs = len(y_true)
    delta = macro_f1(y_true, y_pred_a) - macro_f1(y_true, y_pred_b)
    boots = np.empty(n, dtype=float)
    for i in range(n):
        idx = rng.integers(0, n_obs, size=n_obs)
        boots[i] = macro_f1(y_true[idx], y_pred_a[idx]) - macro_f1(y_true[idx], y_pred_b[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_two = 2.0 * min(float(np.mean(boots <= 0)), float(np.mean(boots >= 0)))
    p_two = min(p_two, 1.0)
    return float(delta), float(lo), float(hi), float(p_two)


def mcnemar_exact(b: int, c: int) -> tuple[float, float]:
    """Mid-p exact McNemar test on discordant counts (b, c).

    Returns (statistic, p_value) where statistic is the standard chi-square form
    and p_value is the exact two-sided binomial test.
    """
    if b + c == 0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    # exact two-sided binomial p
    from math import comb

    n = b + c
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p = min(1.0, 2.0 * tail)
    return float(stat), float(p)


def cohens_d_from_means(
    mean_a: float, mean_b: float, std_a: float, std_b: float, n_a: int, n_b: int
) -> float:
    if n_a < 2 or n_b < 2:
        return float("nan")
    pooled = math.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
    if pooled == 0:
        return float("inf") if mean_a != mean_b else 0.0
    return float((mean_a - mean_b) / pooled)


def interpret(delta: float, lo: float, hi: float, calibration_worse: bool) -> str:
    parts: list[str] = []
    if lo <= 0.0 <= hi:
        parts.append("not statistically decisive (95% CI includes zero)")
    elif delta > 0:
        if abs(delta) < 0.01:
            parts.append("small consistent lift")
        elif abs(delta) < 0.03:
            parts.append("moderate lift")
        else:
            parts.append("large lift")
    else:
        parts.append("negative effect")
    if calibration_worse:
        parts.append("calibration worsens for model_a vs model_b")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _normalize_model_key(raw: str) -> str:
    if raw.startswith("finbert_base_zero_shot"):
        return "finbert_zero_shot"
    if raw.startswith("finbert_finetuned"):
        return "finbert_finetuned"
    if raw == "finbert_agreement_weighted":
        return "finbert_agreement_weighted"
    return raw


def load_seed_sweeps(results_dir: Path) -> pd.DataFrame:
    files = sorted(results_dir.glob("seed_sweep_summary_*.csv"))
    if not files:
        raise FileNotFoundError(f"No seed_sweep_summary_*.csv under {results_dir}")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["__source"] = f.name
        df["__mtime"] = f.stat().st_mtime
        frames.append(df)
    big = pd.concat(frames, ignore_index=True)
    big["model_key"] = big["model"].astype(str).map(_normalize_model_key)
    big = big.sort_values("__mtime").drop_duplicates(
        subset=["model_key", "dataset_name", "seed"], keep="last"
    )
    return big


def discover_predictions(results_dir: Path) -> dict[tuple[str, int], Path]:
    """Map (model_key, seed) -> latest predictions CSV path."""
    out: dict[tuple[str, int], tuple[float, Path]] = {}

    def consider(summary_path: Path, predictions_path: Path) -> None:
        if not predictions_path.exists():
            return
        try:
            head = pd.read_csv(summary_path, nrows=1)
        except Exception:
            return
        if "model" not in head.columns or "seed" not in head.columns:
            return
        model_key = _normalize_model_key(str(head["model"].iloc[0]))
        try:
            seed = int(head["seed"].iloc[0])
        except Exception:
            return
        mtime = predictions_path.stat().st_mtime
        key = (model_key, seed)
        if key not in out or mtime > out[key][0]:
            out[key] = (mtime, predictions_path)

    for sp in results_dir.glob("finbert_baseline_summary_*.csv"):
        pp = sp.with_name(sp.name.replace("summary", "predictions"))
        consider(sp, pp)
    for sp in results_dir.glob("finbert_agreement_weighted_summary_*.csv"):
        pp = sp.with_name(sp.name.replace("summary", "predictions"))
        consider(sp, pp)
    return {k: v[1] for k, v in out.items()}


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["id", "label", "prediction"]].rename(
        columns={"label": "y_true", "prediction": "y_pred"}
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

PAIRS = [
    ("finbert_agreement_weighted", "finbert_zero_shot"),
    ("finbert_agreement_weighted", "finbert_finetuned"),
    ("finbert_zero_shot", "finbert_finetuned"),
]


def build_summary(sweep: pd.DataFrame, bootstrap_n: int, rng: np.random.Generator,
                  preds: dict[tuple[str, int], Path]) -> pd.DataFrame:
    rows = []
    for model_key, group in sweep.groupby("model_key"):
        n_seeds = group["seed"].nunique()
        # Use across-seed bootstrap on per-seed macro_f1 for stability CI
        f1_vals = group["macro_f1"].dropna().to_numpy()
        if len(f1_vals) >= 2:
            boots = rng.choice(f1_vals, size=(bootstrap_n, len(f1_vals)), replace=True).mean(axis=1)
            f1_lo, f1_hi = np.percentile(boots, [2.5, 97.5])
        else:
            f1_lo = f1_hi = float("nan")
        # If predictions available, also compute within-test bootstrap for one seed
        pred_lo = pred_hi = float("nan")
        for seed in sorted(group["seed"].unique()):
            key = (model_key, int(seed))
            if key in preds:
                pdf = load_predictions(preds[key])
                pred_lo, pred_hi = bootstrap_macro_f1_ci(
                    pdf["y_true"].to_numpy(), pdf["y_pred"].to_numpy(),
                    n=bootstrap_n, rng=rng,
                )
                break
        rows.append({
            "model": model_key,
            "n_seeds": int(n_seeds),
            "mean_accuracy": round(float(group["accuracy"].mean()), 6),
            "std_accuracy": round(float(group["accuracy"].std(ddof=1) if n_seeds > 1 else 0.0), 6),
            "mean_macro_f1": round(float(group["macro_f1"].mean()), 6),
            "std_macro_f1": round(float(group["macro_f1"].std(ddof=1) if n_seeds > 1 else 0.0), 6),
            "macro_f1_seed_ci_low": round(float(f1_lo), 6) if not np.isnan(f1_lo) else "",
            "macro_f1_seed_ci_high": round(float(f1_hi), 6) if not np.isnan(f1_hi) else "",
            "macro_f1_test_ci_low": round(float(pred_lo), 6) if not np.isnan(pred_lo) else "",
            "macro_f1_test_ci_high": round(float(pred_hi), 6) if not np.isnan(pred_hi) else "",
            "mean_ece_10_bins": round(float(group["ece_10_bins"].mean()), 6) if "ece_10_bins" in group else "",
            "std_ece_10_bins": round(float(group["ece_10_bins"].std(ddof=1) if n_seeds > 1 else 0.0), 6) if "ece_10_bins" in group else "",
            "mean_brier_score": round(float(group["brier_score"].mean()), 6) if "brier_score" in group else "",
            "std_brier_score": round(float(group["brier_score"].std(ddof=1) if n_seeds > 1 else 0.0), 6) if "brier_score" in group else "",
        })
    return pd.DataFrame(rows).sort_values("mean_macro_f1", ascending=False)


def build_paired(sweep: pd.DataFrame, preds: dict[tuple[str, int], Path],
                 bootstrap_n: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    by_model = {k: g for k, g in sweep.groupby("model_key")}
    for a, b in PAIRS:
        if a not in by_model or b not in by_model:
            continue
        ga, gb = by_model[a], by_model[b]
        # Find a common seed for prediction-level paired tests
        common = sorted(set(ga["seed"]).intersection(gb["seed"]))
        delta_pred = ci_lo = ci_hi = p_emp = float("nan")
        mc_stat = mc_p = float("nan")
        seed_used: Optional[int] = None
        for s in common:
            if (a, int(s)) in preds and (b, int(s)) in preds:
                pa = load_predictions(preds[(a, int(s))]).rename(
                    columns={"y_pred": "pred_a"})
                pb = load_predictions(preds[(b, int(s))]).rename(
                    columns={"y_pred": "pred_b"})
                merged = pa.merge(pb[["id", "pred_b"]], on="id", how="inner")
                if len(merged) == 0:
                    continue
                yt = merged["y_true"].to_numpy()
                ya = merged["pred_a"].to_numpy()
                yb = merged["pred_b"].to_numpy()
                delta_pred, ci_lo, ci_hi, p_emp = paired_bootstrap_delta_macro_f1(
                    yt, ya, yb, n=bootstrap_n, rng=rng,
                )
                # McNemar on correctness
                a_correct = ya == yt
                b_correct = yb == yt
                bcount = int(((a_correct) & (~b_correct)).sum())
                ccount = int(((~a_correct) & (b_correct)).sum())
                mc_stat, mc_p = mcnemar_exact(bcount, ccount)
                seed_used = int(s)
                break
        # Across-seed delta from sweep summary
        mean_a = float(ga["macro_f1"].mean())
        mean_b = float(gb["macro_f1"].mean())
        std_a = float(ga["macro_f1"].std(ddof=1)) if ga["seed"].nunique() > 1 else 0.0
        std_b = float(gb["macro_f1"].std(ddof=1)) if gb["seed"].nunique() > 1 else 0.0
        delta_seed = mean_a - mean_b
        d = cohens_d_from_means(mean_a, mean_b, std_a, std_b, ga["seed"].nunique(), gb["seed"].nunique())
        ece_a = float(ga["ece_10_bins"].mean()) if "ece_10_bins" in ga else float("nan")
        ece_b = float(gb["ece_10_bins"].mean()) if "ece_10_bins" in gb else float("nan")
        cal_worse = (not math.isnan(ece_a) and not math.isnan(ece_b) and ece_a > ece_b)
        interp = interpret(delta_pred if not math.isnan(delta_pred) else delta_seed,
                            ci_lo if not math.isnan(ci_lo) else delta_seed,
                            ci_hi if not math.isnan(ci_hi) else delta_seed,
                            calibration_worse=cal_worse)
        rows.append({
            "model_a": a,
            "model_b": b,
            "seed_used_for_predictions": seed_used if seed_used is not None else "",
            "delta_macro_f1_predictions": round(delta_pred, 6) if not math.isnan(delta_pred) else "",
            "bootstrap_ci_low": round(ci_lo, 6) if not math.isnan(ci_lo) else "",
            "bootstrap_ci_high": round(ci_hi, 6) if not math.isnan(ci_hi) else "",
            "p_value_or_empirical_probability": round(p_emp, 6) if not math.isnan(p_emp) else "",
            "mcnemar_statistic": round(mc_stat, 6) if not math.isnan(mc_stat) else "",
            "mcnemar_p_value": round(mc_p, 6) if not math.isnan(mc_p) else "",
            "delta_macro_f1_across_seeds": round(delta_seed, 6),
            "cohens_d_across_seeds": round(d, 6) if not math.isnan(d) else "",
            "stability_std_a": round(std_a, 6),
            "stability_std_b": round(std_b, 6),
            "mean_ece_a": round(ece_a, 6) if not math.isnan(ece_a) else "",
            "mean_ece_b": round(ece_b, 6) if not math.isnan(ece_b) else "",
            "interpretation": interp,
        })
    return pd.DataFrame(rows)


def render_figure(summary: pd.DataFrame, out_path: Path) -> None:
    if plt is None:
        return
    df = summary.copy()
    df = df.sort_values("mean_macro_f1", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(df))))
    y = np.arange(len(df))
    means = df["mean_macro_f1"].astype(float).to_numpy()
    los = pd.to_numeric(df["macro_f1_seed_ci_low"], errors="coerce").to_numpy()
    his = pd.to_numeric(df["macro_f1_seed_ci_high"], errors="coerce").to_numpy()
    err_low = np.where(np.isnan(los), 0.0, means - los)
    err_high = np.where(np.isnan(his), 0.0, his - means)
    ax.errorbar(means, y, xerr=[err_low, err_high], fmt="o", color="#1f4e79", ecolor="#888")
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"].tolist())
    ax.set_xlabel("Mean macro-F1 with 95% across-seed bootstrap CI")
    ax.set_title("CARA-FinSent — Macro-F1 with 95% bootstrap CIs")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/2026-05-02")
    p.add_argument("--figures_dir", default="figures")
    p.add_argument("--bootstrap_n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    sweep = load_seed_sweeps(results_dir)
    preds = discover_predictions(results_dir)

    summary = build_summary(sweep, args.bootstrap_n, rng, preds)
    paired = build_paired(sweep, preds, args.bootstrap_n, rng)

    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    sum_path = results_dir / f"statistical_validation_summary_{ts}.csv"
    pair_path = results_dir / f"paired_model_comparison_{ts}.csv"
    fig_path = figures_dir / f"statistical_validation_macro_f1_ci_{ts}.png"

    summary.to_csv(sum_path, index=False)
    paired.to_csv(pair_path, index=False)
    render_figure(summary, fig_path)

    print(f"[OK] Summary  -> {sum_path}")
    print(f"[OK] Paired   -> {pair_path}")
    print(f"[OK] Figure   -> {fig_path}")
    print(f"[INFO] Models analyzed: {sorted(summary['model'].tolist())}")
    print(f"[INFO] Pairs analyzed : {len(paired)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
