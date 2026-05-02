#!/usr/bin/env python3
"""Phase 6: per-agreement-level evaluation for PhraseBank predictions.

Reads a predictions CSV that includes an ``agreement`` column and reports
metrics grouped by agreement level (rounded to 2 decimals; expected PhraseBank
levels: 0.50, 0.66, 0.75, 1.00).

Outputs
-------
results/<date>/agreement_subset_eval_<ts>.csv
results/<date>/agreement_subset_classwise_<ts>.csv
figures/agreement_subset_macro_f1_<ts>.png
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
from cara_finsent.io_utils import save_dataframe, ensure_dir, timestamp  # noqa: E402
from cara_finsent.metrics import (  # noqa: E402
    classification_metrics,
    classwise_metrics,
    expected_calibration_error,
)

PROBA_COLS = [f'proba_{c}' for c in STANDARD_LABELS]
OVERCONF = 0.80


def _agreement_lookup(df: pd.DataFrame, gold_path: str | None) -> pd.DataFrame:
    if 'agreement' in df.columns:
        return df
    if not gold_path:
        raise SystemExit('predictions CSV has no `agreement` column. '
                         'Pass --gold_split to merge agreement values by id/text.')
    gold = pd.read_csv(gold_path)
    if 'agreement' not in gold.columns:
        raise SystemExit(f'{gold_path} has no `agreement` column.')
    on = 'id' if ('id' in df.columns and 'id' in gold.columns) else 'text'
    if on not in gold.columns or on not in df.columns:
        raise SystemExit('cannot merge agreement: need shared `id` or `text` column.')
    merged = df.merge(gold[[on, 'agreement']].drop_duplicates(on), on=on, how='left')
    if merged['agreement'].isna().all():
        raise SystemExit('failed to merge any agreement values from gold split.')
    return merged


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--model_name', required=True)
    ap.add_argument('--dataset_name', required=True)
    ap.add_argument('--gold_split', default=None,
                    help='Optional gold-split CSV used to merge `agreement` if missing.')
    ap.add_argument('--results_dir', default='results')
    ap.add_argument('--figures_dir', default='figures')
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    df = _agreement_lookup(df, args.gold_split)
    df = df.dropna(subset=['agreement']).copy()
    df['label'] = df['label'].astype(str)
    df['prediction'] = df['prediction'].astype(str)

    has_proba = all(c in df.columns for c in PROBA_COLS)
    if has_proba:
        proba_all = df[PROBA_COLS].to_numpy(dtype=float)
        df['confidence'] = proba_all.max(axis=1)
    else:
        df['confidence'] = np.nan

    df['agreement_bin'] = df['agreement'].astype(float).round(2)

    rows = []
    classwise_rows = []
    ts = timestamp()

    for level, sub in df.groupby('agreement_bin'):
        if sub.empty:
            continue
        cm = classification_metrics(sub['label'], sub['prediction'])
        rec = {
            'model': args.model_name,
            'dataset_name': args.dataset_name,
            'agreement_level': float(level),
            'n_samples': int(len(sub)),
            'accuracy': cm['accuracy'],
            'macro_f1': cm['macro_f1'],
            'macro_recall': cm['macro_recall'],
        }
        if has_proba:
            sub_proba = sub[PROBA_COLS].to_numpy(dtype=float)
            rec['mean_confidence'] = float(sub_proba.max(axis=1).mean())
            rec['ece_10_bins'] = expected_calibration_error(
                sub['label'], sub['prediction'], sub_proba, n_bins=10)
            rec['overconfident_wrong_count'] = int(
                ((sub['label'] != sub['prediction']) & (sub['confidence'] >= OVERCONF)).sum())
        else:
            rec['mean_confidence'] = np.nan
            rec['ece_10_bins'] = np.nan
            rec['overconfident_wrong_count'] = np.nan
        rows.append(rec)

        cw = classwise_metrics(sub['label'], sub['prediction'])
        cw['agreement_level'] = float(level)
        cw['model'] = args.model_name
        cw['dataset_name'] = args.dataset_name
        classwise_rows.append(cw)

    summary_df = pd.DataFrame(rows).sort_values('agreement_level')
    classwise_df = pd.concat(classwise_rows, ignore_index=True) if classwise_rows else pd.DataFrame()

    summary_path = save_dataframe(summary_df, args.results_dir, 'agreement_subset_eval', ts)
    classwise_path = save_dataframe(classwise_df, args.results_dir, 'agreement_subset_classwise', ts)

    fig_dir = ensure_dir(Path(args.figures_dir))
    fig_path = fig_dir / f'agreement_subset_macro_f1_{ts}.png'
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([f'{x:.2f}' for x in summary_df['agreement_level']], summary_df['macro_f1'],
           color='#4477aa')
    for i, n in enumerate(summary_df['n_samples']):
        ax.text(i, summary_df['macro_f1'].iloc[i] + 0.01, f'n={n}',
                ha='center', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('agreement level')
    ax.set_ylabel('macro-F1')
    ax.set_title(f'{args.model_name} on {args.dataset_name} — by agreement')
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print(f'[DONE] subset summary  -> {summary_path}')
    print(f'[DONE] classwise file  -> {classwise_path}')
    print(f'[DONE] figure          -> {fig_path}')
    print(summary_df.to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
