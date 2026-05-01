#!/usr/bin/env python3
"""Phase 4: Paper-ready error analysis from a prediction CSV.

Inputs
------
A predictions CSV with at minimum:
    label, prediction, proba_negative, proba_neutral, proba_positive
Optional columns used when present: text, agreement, dataset_name, id.

Outputs
-------
results/<date>/error_analysis_summary_<ts>.csv
results/<date>/error_examples_by_class_<ts>.csv
results/<date>/neutral_confusion_analysis_<ts>.csv
figures/error_confusion_heatmap_<ts>.png
figures/neutral_error_breakdown_<ts>.png
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from cara_finsent.data_utils import STANDARD_LABELS  # noqa: E402
from cara_finsent.io_utils import ensure_dir, save_dataframe, save_json, timestamp  # noqa: E402
from cara_finsent.metrics import classwise_metrics, confusion_matrix_df  # noqa: E402

NUMERIC_RE = re.compile(r'(\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?)')


def length_bucket(n: int) -> str:
    if n < 50:
        return '00_lt50'
    if n < 100:
        return '01_50to99'
    if n < 200:
        return '02_100to199'
    return '03_200plus'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--model_name', required=True)
    ap.add_argument('--dataset_name', required=True)
    ap.add_argument('--results_dir', default='results')
    ap.add_argument('--figures_dir', default='figures')
    ap.add_argument('--worst_n', type=int, default=25)
    args = ap.parse_args()

    ts = timestamp()
    df = pd.read_csv(args.predictions)
    needed = {'label', 'prediction'}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f'predictions CSV missing required columns: {sorted(missing)}')

    proba_cols = [f'proba_{c}' for c in STANDARD_LABELS]
    has_proba = all(c in df.columns for c in proba_cols)
    if has_proba:
        probs = df[proba_cols].to_numpy(dtype=float)
        df['confidence'] = probs.max(axis=1)
    else:
        df['confidence'] = np.nan

    df['correct'] = df['label'].astype(str) == df['prediction'].astype(str)

    cm = confusion_matrix_df(df['label'], df['prediction'])
    cw = classwise_metrics(df['label'], df['prediction'])

    n_total = len(df)
    n_errors = int((~df['correct']).sum())
    err_rate = n_errors / n_total if n_total else 0.0

    summary_rows = [{
        'model_name': args.model_name,
        'dataset_name': args.dataset_name,
        'predictions_file': str(args.predictions),
        'rows': n_total,
        'errors': n_errors,
        'error_rate': err_rate,
        'mean_confidence_overall': float(df['confidence'].mean()) if has_proba else float('nan'),
        'mean_confidence_correct': float(df.loc[df['correct'], 'confidence'].mean()) if has_proba and df['correct'].any() else float('nan'),
        'mean_confidence_wrong': float(df.loc[~df['correct'], 'confidence'].mean()) if has_proba and (~df['correct']).any() else float('nan'),
    }]

    # Class-wise error rate
    for _, row in cw.iterrows():
        summary_rows.append({
            'model_name': args.model_name,
            'dataset_name': args.dataset_name,
            'predictions_file': str(args.predictions),
            'rows': int(row['support']),
            'errors': int(row['support']) - int(row['support'] * row['recall']),
            'error_rate': 1.0 - float(row['recall']),
            'class': row['class'],
            'precision': float(row['precision']),
            'recall': float(row['recall']),
            'f1': float(row['f1']),
        })

    # Length bucket / numeric content patterns
    if 'text' in df.columns:
        df['text_len'] = df['text'].astype(str).str.len()
        df['len_bucket'] = df['text_len'].apply(length_bucket)
        df['has_numeric'] = df['text'].astype(str).apply(lambda s: bool(NUMERIC_RE.search(s)))
        for bucket, sub in df.groupby('len_bucket'):
            summary_rows.append({
                'model_name': args.model_name,
                'dataset_name': args.dataset_name,
                'predictions_file': str(args.predictions),
                'rows': int(len(sub)),
                'errors': int((~sub['correct']).sum()),
                'error_rate': float((~sub['correct']).mean()),
                'pattern': f'len_bucket={bucket}',
            })
        for has_num, sub in df.groupby('has_numeric'):
            summary_rows.append({
                'model_name': args.model_name,
                'dataset_name': args.dataset_name,
                'predictions_file': str(args.predictions),
                'rows': int(len(sub)),
                'errors': int((~sub['correct']).sum()),
                'error_rate': float((~sub['correct']).mean()),
                'pattern': f'has_numeric={bool(has_num)}',
            })

    if 'agreement' in df.columns:
        for level, sub in df.groupby(pd.cut(df['agreement'].astype(float), bins=[0, 0.5, 0.66, 0.75, 1.0], include_lowest=True)):
            summary_rows.append({
                'model_name': args.model_name,
                'dataset_name': args.dataset_name,
                'predictions_file': str(args.predictions),
                'rows': int(len(sub)),
                'errors': int((~sub['correct']).sum()),
                'error_rate': float((~sub['correct']).mean()) if len(sub) else 0.0,
                'pattern': f'agreement={level}',
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = save_dataframe(summary_df, args.results_dir, 'error_analysis_summary', ts)

    # Worst errors by confidence + low-confidence correct
    text_col = 'text' if 'text' in df.columns else None
    base_cols = [c for c in ['id', text_col, 'label', 'prediction', 'confidence', 'agreement'] if c]
    error_df = df[~df['correct']].copy()
    if has_proba and len(error_df):
        worst_high_conf = error_df.sort_values('confidence', ascending=False).head(args.worst_n).copy()
        worst_high_conf['error_type'] = 'high_confidence_wrong'
    else:
        worst_high_conf = pd.DataFrame(columns=base_cols + ['error_type'])

    correct_df = df[df['correct']].copy()
    if has_proba and len(correct_df):
        low_conf_correct = correct_df.sort_values('confidence', ascending=True).head(args.worst_n).copy()
        low_conf_correct['error_type'] = 'low_confidence_correct'
    else:
        low_conf_correct = pd.DataFrame(columns=base_cols + ['error_type'])

    worst_by_conf = error_df.sort_values('confidence', ascending=True).head(args.worst_n).copy() if has_proba and len(error_df) else pd.DataFrame()
    if len(worst_by_conf):
        worst_by_conf['error_type'] = 'worst_low_confidence_wrong'

    examples_by_class_frames = []
    for cls in STANDARD_LABELS:
        sub = error_df[error_df['label'] == cls]
        if has_proba and len(sub):
            ex = sub.sort_values('confidence', ascending=False).head(args.worst_n).copy()
        else:
            ex = sub.head(args.worst_n).copy()
        ex['error_type'] = f'top_errors_actual_{cls}'
        examples_by_class_frames.append(ex)
    examples_concat = pd.concat([worst_high_conf, low_conf_correct, worst_by_conf] + examples_by_class_frames,
                                ignore_index=True, sort=False)
    keep_cols = [c for c in base_cols + ['error_type'] if c in examples_concat.columns]
    examples_path = save_dataframe(examples_concat[keep_cols] if not examples_concat.empty else examples_concat,
                                   args.results_dir, 'error_examples_by_class', ts)

    # Neutral confusion analysis
    neutral_rows = []
    for direction, mask in (('neutral_pred_but_wrong', (df['prediction'] == 'neutral') & ~df['correct']),
                            ('actual_neutral_but_wrong', (df['label'] == 'neutral') & ~df['correct'])):
        sub = df[mask]
        breakdown = sub.groupby([df.loc[sub.index, 'label'], df.loc[sub.index, 'prediction']]).size().reset_index(name='count')
        breakdown['analysis'] = direction
        neutral_rows.append(breakdown)
    neutral_df = pd.concat(neutral_rows, ignore_index=True) if neutral_rows else pd.DataFrame()
    neutral_path = save_dataframe(neutral_df, args.results_dir, 'neutral_confusion_analysis', ts)

    # Plots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figures_dir = ensure_dir(Path(args.figures_dir))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cm_arr = cm.to_numpy(dtype=float)
    cm_norm = cm_arr / np.where(cm_arr.sum(axis=1, keepdims=True) == 0, 1, cm_arr.sum(axis=1, keepdims=True))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(STANDARD_LABELS)), STANDARD_LABELS, rotation=30)
    ax.set_yticks(range(len(STANDARD_LABELS)), STANDARD_LABELS)
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, f'{int(cm_arr[i, j])}\n({cm_norm[i, j]:.2f})',
                    ha='center', va='center', color='black' if cm_norm[i, j] < 0.5 else 'white', fontsize=9)
    ax.set_xlabel('predicted')
    ax.set_ylabel('actual')
    ax.set_title(f'Errors heatmap: {args.model_name} on {args.dataset_name}')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    heatmap_path = figures_dir / f'error_confusion_heatmap_{ts}.png'
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)

    # Neutral breakdown bar chart
    if 'count' in neutral_df.columns:
        actual_neutral = neutral_df[neutral_df['analysis'] == 'actual_neutral_but_wrong']
        pred_neutral = neutral_df[neutral_df['analysis'] == 'neutral_pred_but_wrong']
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        if not actual_neutral.empty:
            actual_neutral.plot(kind='bar', x='prediction', y='count', ax=axes[0], legend=False, color='#c2553a')
            axes[0].set_title('Actual neutral -> wrong predictions')
            axes[0].set_xlabel('predicted class')
        if not pred_neutral.empty:
            pred_neutral.plot(kind='bar', x='label', y='count', ax=axes[1], legend=False, color='#3a78c2')
            axes[1].set_title('Predicted neutral but wrong')
            axes[1].set_xlabel('actual class')
        fig.tight_layout()
        neutral_fig = figures_dir / f'neutral_error_breakdown_{ts}.png'
        fig.savefig(neutral_fig, dpi=150)
        plt.close(fig)
    else:
        neutral_fig = None

    manifest = {
        'timestamp_utc': ts,
        'model_name': args.model_name,
        'dataset_name': args.dataset_name,
        'predictions_file': str(args.predictions),
        'summary_csv': str(summary_path),
        'examples_csv': str(examples_path),
        'neutral_csv': str(neutral_path),
        'heatmap_png': str(heatmap_path),
        'neutral_breakdown_png': str(neutral_fig) if neutral_fig else None,
        'rows': n_total,
        'errors': n_errors,
        'error_rate': err_rate,
    }
    save_json(manifest, args.results_dir, 'error_analysis_manifest', ts)
    print(f'[DONE] Error summary -> {summary_path}')
    print(f'[DONE] Heatmap       -> {heatmap_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
