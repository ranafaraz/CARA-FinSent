"""Phase 11 publication-grade figure generator for CARA-FinSent.

Outputs (PNG @300 DPI, PDF where matplotlib supports it) under
paper_assets/figures/:
    fig1_pipeline_architecture.png
    fig2_macro_f1_leaderboard.png
    fig3_calibration_comparison.png
    fig4_seed_stability.png
    fig5_abstention_tradeoff.png
    fig6_claim_boundary_visual.png

Inputs (auto-discovered):
    results/<date>/final_leaderboard_mean_std_*.csv
    results/<date>/seed_sweep_summary_*.csv
    results/<date>/calibration_summary_*.csv
    results/<date>/abstention_curve_*.csv

Usage:
    python scripts/34_generate_publication_figures.py \
        --results_dir results/2026-05-02 \
        --output_dir paper_assets/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def latest(results_dir: Path, pattern: str) -> Path:
    files = sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(pattern)
    return files[-1]


def save(fig, out_dir: Path, name: str) -> None:
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    try:
        fig.savefig(pdf, bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)
    print(f"[OK] {png}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig1_pipeline(out_dir: Path) -> None:
    stages = [
        "Financial text",
        "Preprocessing",
        "Controlled gold split\n(no text-hash leakage)",
        "Classical baselines\n(TF-IDF + linear/tree models)",
        "FinBERT zero-shot",
        "FinBERT fine-tuning",
        "Agreement-weighted\nFinBERT",
        "Calibration + abstention",
        "Decision-grade reporting\n(macro-F1, ECE, Brier, coverage)",
    ]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off")
    n = len(stages)
    box_w, box_h = 1.8, 1.0
    gap_x = 0.35
    total_w = n * box_w + (n - 1) * gap_x
    x0 = -total_w / 2
    y0 = -box_h / 2
    for i, label in enumerate(stages):
        x = x0 + i * (box_w + gap_x)
        rect = mpatches.FancyBboxPatch(
            (x, y0), box_w, box_h, boxstyle="round,pad=0.04,rounding_size=0.18",
            linewidth=1.2, edgecolor="#1f4e79", facecolor="#dde7f5",
        )
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y0 + box_h / 2, label, ha="center", va="center",
                fontsize=9, color="#1f1f1f")
        if i < n - 1:
            ax.annotate(
                "", xy=(x + box_w + gap_x, 0), xytext=(x + box_w, 0),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.2),
            )
    ax.set_xlim(x0 - 0.5, x0 + total_w + 0.5)
    ax.set_ylim(-1.4, 1.4)
    ax.set_title("CARA-FinSent reliability-first pipeline", fontsize=12)
    save(fig, out_dir, "fig1_pipeline_architecture")


def fig2_leaderboard(results_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(latest(results_dir, "final_leaderboard_mean_std_*.csv"))
    df = df.sort_values("mean_macro_f1", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(df))
    means = df["mean_macro_f1"].astype(float).to_numpy()
    stds = df["std_macro_f1"].astype(float).fillna(0.0).to_numpy()
    colors = ["#1f4e79" if "finbert" in m or "agreement" in m else "#999"
              for m in df["model"]]
    ax.barh(y, means, xerr=stds, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"].tolist(), fontsize=9)
    ax.set_xlabel("Mean macro-F1 (error bars = std across seeds)")
    ax.set_xlim(0, 1.0)
    ax.set_title("CARA-FinSent — Macro-F1 leaderboard (PhraseBank, in-domain)")
    ax.grid(True, axis="x", alpha=0.3)
    save(fig, out_dir, "fig2_macro_f1_leaderboard")


def fig3_calibration(results_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(latest(results_dir, "final_leaderboard_mean_std_*.csv"))
    df = df.dropna(subset=["mean_ece_10_bins"]).copy()
    df = df.sort_values("mean_macro_f1", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    is_finbert = df["model"].str.contains("finbert|agreement", case=False, regex=True)
    ax.scatter(df.loc[is_finbert, "mean_ece_10_bins"],
               df.loc[is_finbert, "mean_macro_f1"],
               s=110, color="#1f4e79", label="FinBERT family", zorder=3)
    ax.scatter(df.loc[~is_finbert, "mean_ece_10_bins"],
               df.loc[~is_finbert, "mean_macro_f1"],
               s=80, color="#999", label="Classical", zorder=2)
    for _, r in df.iterrows():
        ax.annotate(r["model"].replace("ProsusAI/", ""),
                    (r["mean_ece_10_bins"], r["mean_macro_f1"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean ECE@10 (lower = better calibrated)")
    ax.set_ylabel("Mean macro-F1 (higher = better)")
    ax.set_title("Accuracy vs calibration trade-off")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    save(fig, out_dir, "fig3_calibration_comparison")


def fig4_seed_stability(results_dir: Path, out_dir: Path) -> None:
    files = sorted(results_dir.glob("seed_sweep_summary_*.csv"))
    frames = [pd.read_csv(f) for f in files]
    big = pd.concat(frames, ignore_index=True)
    big["model_key"] = big["model"].str.replace("ProsusAI/", "", regex=False)
    targets = ["finbert_agreement_weighted",
               "finbert_finetuned_finbert",
               "finbert_base_zero_shot_finbert"]
    keep = big[big["model_key"].isin(targets)]
    if keep.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {"finbert_agreement_weighted": "#1f4e79",
               "finbert_finetuned_finbert": "#c0392b",
               "finbert_base_zero_shot_finbert": "#27ae60"}
    nice = {"finbert_agreement_weighted": "Agreement-weighted",
            "finbert_finetuned_finbert": "Vanilla fine-tuned",
            "finbert_base_zero_shot_finbert": "Zero-shot"}
    for name, sub in keep.groupby("model_key"):
        sub = sub.sort_values("seed")
        ax.plot(sub["seed"].astype(str), sub["macro_f1"], marker="o",
                color=palette.get(name, "#444"), label=nice.get(name, name), linewidth=2)
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Macro-F1 on PhraseBank test")
    ax.set_title("Seed stability — agreement weighting reduces variance")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    save(fig, out_dir, "fig4_seed_stability")


def fig5_abstention(results_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(latest(results_dir, "abstention_curve_*.csv"))
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, sub in df.groupby("model"):
        sub = sub.sort_values("threshold")
        ax.plot(sub["coverage"], sub["accuracy_on_kept"], marker=".",
                label=name, linewidth=2)
    ax.set_xlabel("Coverage (fraction of test predictions kept)")
    ax.set_ylabel("Accuracy on kept predictions")
    ax.set_title("Abstention trade-off: coverage vs accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    save(fig, out_dir, "fig5_abstention_tradeoff")


def fig6_claim_boundary(out_dir: Path) -> None:
    claims = [
        ("AW: best mean macro-F1", "supported"),
        ("AW: stronger seed stability vs vanilla FT", "supported"),
        ("Zero-shot: best calibration", "supported"),
        ("Classical TF-IDF: cheaper but lower F1", "supported"),
        ("AW > zero-shot statistically decisive", "not supported"),
        ("Generalises to live trading", "not supported"),
    ]
    color = {"supported": "#27ae60", "not supported": "#c0392b"}
    fig, ax = plt.subplots(figsize=(9, 4))
    y = np.arange(len(claims))
    ax.barh(y, [1] * len(claims),
            color=[color[c[1]] for c in claims], edgecolor="black", linewidth=0.5)
    for i, (text, status) in enumerate(claims):
        ax.text(0.02, i, f"{text}  —  [{status}]", va="center",
                fontsize=9, color="white" if status == "not supported" else "black")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_title("Claim boundaries for the CARA-FinSent paper")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    save(fig, out_dir, "fig6_claim_boundary_visual")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/2026-05-02")
    ap.add_argument("--output_dir", default="paper_assets/figures")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1_pipeline(out_dir)
    fig2_leaderboard(results_dir, out_dir)
    fig3_calibration(results_dir, out_dir)
    fig4_seed_stability(results_dir, out_dir)
    fig5_abstention(results_dir, out_dir)
    fig6_claim_boundary(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
