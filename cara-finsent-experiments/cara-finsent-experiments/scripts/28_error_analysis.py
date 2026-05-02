#!/usr/bin/env python3
"""Phase 5: error analysis with the schema required by the Phase 5 brief.

Produces:
    results/<date>/error_analysis_summary_<ts>.csv
    results/<date>/error_examples_<ts>.csv
    figures/error_type_distribution_<ts>.png

Summary columns (one row per (model, dataset_name) pair):
    model, dataset_name, rows, errors, error_rate,
    neutral_as_positive, neutral_as_negative,
    positive_as_negative, negative_as_positive,
    mean_confidence_correct, mean_confidence_wrong,
    overconfident_wrong_count, low_confidence_correct_count

`overconfident_wrong` := prediction != label AND confidence >= 0.80
`low_confidence_correct` := prediction == label AND confidence <  0.60

Examples file: top --top_n highest-confidence wrong predictions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from cara_finsent.data_utils import STANDARD_LABELS  # noqa: E402
from cara_finsent.io_utils import ensure_dir, save_dataframe, timestamp  # noqa: E402

OVERCONF_THRESHOLD = 0.80
LOWCONF_THRESHOLD = 0.60


def confusion_count(df: pd.DataFrame, true_label: str, pred_label: str) -> int:
    return int(((df['label'] == true_label) & (df['prediction'] == pred_label)).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--model_name', required=True)
    ap.add_argument('--dataset_name', required=True)
    ap.add_argument('--results_dir', default='results')
    ap.add_argument('--figures_dir', default='figures')
    ap.add_argument('--top_n', type=int, default=50)
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    needed = {'label', 'prediction'}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f'predictions CSV missing required columns: {sorted(missing)}')

    df['label'] = df['label'].astype(str)
    df['prediction'] = df['prediction'].astype(str)

    proba_cols = [f'proba_{c}' for c in STANDARD_LABELS]
    has_proba = all(c in df.columns for c in proba_cols)
    if has_proba:
        df['confidence'] = df[proba_cols].to_numpy(dtype=float).max(axis=1)
    else:
        df['confidence'] = np.nan

    df['correct'] = df['label'] == df['prediction']
    n_total = len(df)
    n_errors = int((~df['correct']).sum())
    err_rate = n_errors / n_total if n_total else 0.0

    overconf_wrong = int(((~df['correct']) & (df['confidence'] >= OVERCONF_THRESHOLD)).sum())
    lowconf_correct = int((df['correct'] & (df['confidence'] < LOWCONF_THRESHOLD)).sum())
    mean_conf_correct = float(df.loc[df['correct'], 'confidence'].mean()) if has_proba and df['correct'].any() else np.nan
    mean_conf_wrong = float(df.loc[~df['correct'], 'confidence'].mean()) if has_proba and (~df['correct']).any() else np.nan

    summary = {
        'model': args.model_name,
        'dataset_name': args.dataset_name,
        'rows': n_total,
        'errors': n_errors,
        'error_rate': err_rate,
        'neutral_as_positive': confusion_count(df, 'neutral', 'positive'),
        'neutral_as_negative': confusion_count(df, 'neutral', 'negative'),
        'positive_as_negative': confusion_count(df, 'positive', 'negative'),
        'negative_as_positive': confusion_count(df, 'negative', 'positive'),
        'mean_confidence_correct': mean_conf_correct,
        'mean_confidence_wrong': mean_conf_wrong,
        'overconfident_wrong_count': overconf_wrong,
        'low_confidence_correct_count': lowconf_correct,
    }
    summary_df = pd.DataFrame([summary])

    ts = timestamp()
    summary_path = save_dataframe(summary_df, args.results_dir, 'error_analysis_summary', ts)

    # Top-N highest-confidence wrong predictions.
    wrong = df[~df['correct']].copy()
    if has_proba and not wrong.empty:
        wrong = wrong.sort_values('confidence', ascending=False)
    keep_cols = [c for c in ['id', 'text', 'label', 'prediction', 'confidence',
                             'proba_negative', 'proba_neutral', 'proba_positive',
                             'agreement', 'dataset_name']
                 if c in wrong.columns or c == 'confidence']
    examples = wrong.head(args.top_n)[[c for c in keep_cols if c in wrong.columns]]
    examples_path = save_dataframe(examples, args.results_dir, 'error_examples', ts)

    # Error type distribution figure.
    error_types = {
        'neutral_as_positive': summary['neutral_as_positive'],
        'neutral_as_negative': summary['neutral_as_negative'],
        'positive_as_negative': summary['positive_as_negative'],
        'negative_as_positive': summary['negative_as_positive'],
        'overconfident_wrong': overconf_wrong,
        'low_confidence_correct': lowconf_correct,
    }
    fig_dir = ensure_dir(Path(args.figures_dir))
    fig_path = fig_dir / f'error_type_distribution_{ts}.png'
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(error_types.keys(), error_types.values(), color='#cc6677')
    ax.set_ylabel('count')
    ax.set_title(f'Error type distribution: {args.model_name} on {args.dataset_name}')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print(f'[DONE] error summary  -> {summary_path}')
    print(f'[DONE] error examples -> {examples_path}')
    print(f'[DONE] figure         -> {fig_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
