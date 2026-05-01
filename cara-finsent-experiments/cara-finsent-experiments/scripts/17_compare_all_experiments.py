"""
17_compare_all_experiments.py
-----------------------------
Collect all experiment summary CSVs from results/, merge into a single
cross-experiment comparison table, and save bar-plot figures for key metrics.

Usage:
    python scripts/17_compare_all_experiments.py [--results_dir results]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import pandas as pd

from cara_finsent.io_utils import timestamped_path, save_dataframe, ensure_dir
from cara_finsent.plotting import save_metric_bar_plot

# ── columns we want to keep (subset of common cols across experiments) ──────
METRIC_COLS = ['accuracy', 'macro_f1', 'weighted_f1', 'mcc', 'ece', 'brier_score']
ID_COLS = ['model', 'experiment']

# ── pattern → experiment label mapping ─────────────────────────────────────
EXPERIMENT_PATTERNS = [
    ('classical_baseline_summary',    'classical_baselines'),
    ('finbert_baseline_summary',      'finbert'),
    ('structured_features_summary',   'structured_features'),
    ('retrieval_experiment_summary',  'retrieval'),
    ('agreement_aware_summary',       'agreement_aware'),
    ('calibration_summary',           'calibration'),
    ('cara_lite_summary',             'cara_lite'),
    ('cara_lite_ablation_summary',    'cara_lite_ablation'),
    ('pipeline_step_report',          'pipeline'),
]


def discover_summaries(results_dir: Path) -> list[tuple[str, Path]]:
    """Walk results_dir (all date subdirs) and return (experiment_label, path) pairs.
    When multiple runs exist for the same experiment label, keep only the most
    recent file (by filename timestamp).
    """
    best: dict[str, Path] = {}
    for csv_path in results_dir.rglob('*.csv'):
        # Skip anything under an archive folder (pre-rebuild artifacts)
        if any(part.startswith('_archive') or part == 'archive' for part in csv_path.parts):
            continue
        name = csv_path.stem
        for pattern, label in EXPERIMENT_PATTERNS:
            if pattern in name:
                existing = best.get(label)
                if existing is None or str(csv_path) > str(existing):
                    best[label] = csv_path
                break
    return sorted(best.items())


def load_summary(label: str, path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f'[WARN] Could not load {path}: {e}')
        return None

    if 'model' not in df.columns:
        # try to synthesise a model column from other id-like columns
        for candidate in ['step', 'method', 'name']:
            if candidate in df.columns:
                df = df.rename(columns={candidate: 'model'})
                break
        else:
            df['model'] = label

    df['experiment'] = label
    return df


def build_comparison(results_dir: Path) -> pd.DataFrame:
    summaries = discover_summaries(results_dir)
    if not summaries:
        sys.exit(f'[ERROR] No summary CSVs found under {results_dir}')

    print(f'[INFO] Found {len(summaries)} experiment summary file(s):')
    frames: list[pd.DataFrame] = []
    for label, path in summaries:
        print(f'  {label:35s} <- {path.relative_to(results_dir)}')
        df = load_summary(label, path)
        if df is not None:
            frames.append(df)

    if not frames:
        sys.exit('[ERROR] All summaries failed to load.')

    combined = pd.concat(frames, ignore_index=True, sort=False)

    # ensure metric cols exist (fill missing with NaN)
    for col in METRIC_COLS:
        if col not in combined.columns:
            combined[col] = float('nan')

    # reorder: id cols first, then metrics, then anything else
    other_cols = [c for c in combined.columns if c not in ID_COLS + METRIC_COLS]
    combined = combined[ID_COLS + METRIC_COLS + other_cols]
    combined = combined.sort_values(['experiment', 'model']).reset_index(drop=True)
    return combined


def save_plots(combined: pd.DataFrame, figures_dir: Path) -> None:
    ensure_dir(figures_dir)
    plot_metrics = [m for m in ['macro_f1', 'accuracy', 'weighted_f1', 'ece']
                    if m in combined.columns and combined[m].notna().any()]

    for metric in plot_metrics:
        plot_df = combined[['model', 'experiment', metric]].dropna(subset=[metric]).copy()
        plot_df['model'] = plot_df['experiment'] + '/' + plot_df['model'].astype(str)
        path = figures_dir / f'comparison_{metric}.png'
        save_metric_bar_plot(plot_df, metric, path,
                             title=f'All experiments — {metric}')
        print(f'[PLOT] {path.name}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Cross-experiment comparison table and plots')
    parser.add_argument('--results_dir', default='results',
                        help='Root results directory (default: results)')
    parser.add_argument('--figures_dir', default='figures',
                        help='Output directory for plots (default: figures)')
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / args.results_dir
    figures_dir = PROJECT_ROOT / args.figures_dir

    if not results_dir.exists():
        sys.exit(f'[ERROR] results_dir not found: {results_dir}')

    combined = build_comparison(results_dir)

    # ── save combined CSV ───────────────────────────────────────────────────
    out_path = save_dataframe(combined, results_dir, 'all_experiments_comparison')
    print(f'\n[DONE] Comparison table ({len(combined)} rows) -> {out_path}')

    # ── save per-metric bar plots ───────────────────────────────────────────
    save_plots(combined, figures_dir)

    # ── print human-readable leaderboard ────────────────────────────────────
    if 'macro_f1' in combined.columns:
        print('\n-- Leaderboard (macro_f1, descending) --')
        lb = (combined[['experiment', 'model', 'macro_f1', 'accuracy']]
              .dropna(subset=['macro_f1'])
              .sort_values('macro_f1', ascending=False)
              .reset_index(drop=True))
        print(lb.to_string(index=False))


if __name__ == '__main__':
    main()
