#!/usr/bin/env python3
"""Phase 4: Aggregate seed-level sweeps into publication-ready leaderboards.

Reads all ``seed_sweep_summary_*.csv`` files from ``results/`` and produces:
  - results/<date>/research_leaderboard_mean_std_<ts>.csv
  - results/<date>/research_leaderboard_ci95_<ts>.csv
  - results/<date>/research_best_model_summary_<ts>.csv
  - figures/research_macro_f1_mean_std_<ts>.png
  - figures/research_accuracy_mean_std_<ts>.png

Acceptance rule: only (model, dataset_name) groups with at least
``--min_seeds`` rows are flagged ``research_grade=True``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from cara_finsent.io_utils import (  # noqa: E402
    ensure_dir,
    git_commit_sha,
    save_dataframe,
    save_json,
    timestamp,
)


METRIC_COLUMNS = ['accuracy', 'macro_f1', 'weighted_f1', 'mcc',
                  'ece_10_bins', 'brier_score', 'mean_confidence']
GROUP_COLUMNS = ['model', 'experiment', 'dataset_name', 'benchmark_mode']


def load_sweeps(results_dir: Path) -> pd.DataFrame:
    files: List[Path] = []
    for path in results_dir.rglob('seed_sweep_summary_*.csv'):
        if '_archive' in path.parts:
            continue
        files.append(path)
    if not files:
        raise SystemExit(
            f'No seed_sweep_summary_*.csv files found under {results_dir}. '
            'Run scripts/19_run_seed_sweep.py first.'
        )
    frames = []
    for path in files:
        df = pd.read_csv(path)
        df['source_sweep_file'] = str(path)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined


def ci95(series: pd.Series) -> tuple[float, float]:
    arr = series.dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return (float('nan'), float('nan'))
    mean = float(arr.mean())
    if arr.size < 2:
        return (mean, mean)
    sem = float(arr.std(ddof=1)) / float(np.sqrt(arr.size))
    half = 1.96 * sem
    return (mean - half, mean + half)


def aggregate(df: pd.DataFrame, min_seeds: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_cols = [c for c in METRIC_COLUMNS if c in df.columns]
    rows_mean = []
    rows_ci = []
    for keys, sub in df.groupby(GROUP_COLUMNS, dropna=False):
        seeds = sorted(sub['seed'].dropna().unique().tolist()) if 'seed' in sub.columns else []
        n_seeds = len(seeds)
        record_mean = dict(zip(GROUP_COLUMNS, keys))
        record_mean['n_seeds'] = n_seeds
        record_mean['seeds'] = ','.join(str(int(s)) for s in seeds)
        for metric in metric_cols:
            record_mean[f'mean_{metric}'] = float(sub[metric].astype(float).mean())
            record_mean[f'std_{metric}'] = float(sub[metric].astype(float).std(ddof=1)) if n_seeds > 1 else 0.0
        record_mean['research_grade'] = bool(n_seeds >= min_seeds)
        rows_mean.append(record_mean)

        record_ci = dict(zip(GROUP_COLUMNS, keys))
        record_ci['n_seeds'] = n_seeds
        for metric in metric_cols:
            lo, hi = ci95(sub[metric].astype(float))
            record_ci[f'{metric}_mean'] = float(sub[metric].astype(float).mean())
            record_ci[f'ci95_{metric}_lower'] = lo
            record_ci[f'ci95_{metric}_upper'] = hi
        record_ci['research_grade'] = bool(n_seeds >= min_seeds)
        rows_ci.append(record_ci)

    mean_df = pd.DataFrame(rows_mean).sort_values(['dataset_name', 'mean_macro_f1'], ascending=[True, False])
    ci_df = pd.DataFrame(rows_ci).sort_values(['dataset_name', 'macro_f1_mean'], ascending=[True, False])

    best_rows = []
    for ds, sub in mean_df.groupby('dataset_name'):
        eligible = sub[sub['research_grade']]
        target = eligible if not eligible.empty else sub
        idx = target['mean_macro_f1'].idxmax()
        row = target.loc[idx].to_dict()
        row['selection_basis'] = 'research_grade' if not eligible.empty else 'best_available'
        best_rows.append(row)
    best_df = pd.DataFrame(best_rows)
    return mean_df, ci_df, best_df


def plot_metric(mean_df: pd.DataFrame, metric: str, figures_dir: Path, ts: str) -> Path:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mean_col = f'mean_{metric}'
    std_col = f'std_{metric}'
    if mean_col not in mean_df.columns:
        raise SystemExit(f'Missing column {mean_col}')

    plot_df = mean_df.copy()
    plot_df['label'] = plot_df['model'].astype(str) + ' [' + plot_df['dataset_name'].astype(str) + ']'
    plot_df = plot_df.sort_values(mean_col, ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(plot_df) + 2)))
    ax.barh(plot_df['label'], plot_df[mean_col],
            xerr=plot_df.get(std_col, pd.Series([0] * len(plot_df))),
            color='#3a78c2', edgecolor='black')
    ax.set_xlabel(f'mean {metric} (+/- std)')
    ax.set_title(f'CARA-FinSent research leaderboard: {metric}')
    ax.set_xlim(0, max(1.0, float(plot_df[mean_col].max()) * 1.1))
    fig.tight_layout()
    figures_dir = ensure_dir(figures_dir)
    out = figures_dir / f'research_{metric}_mean_std_{ts}.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='results')
    ap.add_argument('--figures_dir', default='figures')
    ap.add_argument('--min_seeds', type=int, default=5)
    args = ap.parse_args()

    ts = timestamp()
    df = load_sweeps(Path(args.results_dir))
    print(f'[INFO] Loaded {len(df)} sweep rows from {df["source_sweep_file"].nunique()} files')

    mean_df, ci_df, best_df = aggregate(df, args.min_seeds)

    mean_path = save_dataframe(mean_df, args.results_dir, 'research_leaderboard_mean_std', ts)
    ci_path = save_dataframe(ci_df, args.results_dir, 'research_leaderboard_ci95', ts)
    best_path = save_dataframe(best_df, args.results_dir, 'research_best_model_summary', ts)

    figures_dir = Path(args.figures_dir)
    f1_plot = plot_metric(mean_df, 'macro_f1', figures_dir, ts)
    acc_plot = plot_metric(mean_df, 'accuracy', figures_dir, ts)

    manifest = {
        'timestamp_utc': ts,
        'min_seeds_required': args.min_seeds,
        'rows_loaded': int(len(df)),
        'unique_sweeps': int(df['source_sweep_file'].nunique()),
        'leaderboard_mean_std': str(mean_path),
        'leaderboard_ci95': str(ci_path),
        'best_model_summary': str(best_path),
        'figure_macro_f1': str(f1_plot),
        'figure_accuracy': str(acc_plot),
        'git_commit_sha': git_commit_sha(PROJECT_ROOT),
    }
    save_json(manifest, args.results_dir, 'research_leaderboard_manifest', ts)
    print(f'[DONE] Mean/Std leaderboard -> {mean_path}')
    print(f'[DONE] CI95 leaderboard     -> {ci_path}')
    print(f'[DONE] Best model summary   -> {best_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
