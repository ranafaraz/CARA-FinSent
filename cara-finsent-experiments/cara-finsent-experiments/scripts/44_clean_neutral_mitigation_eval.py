#!/usr/bin/env python3
"""Phase 14 Step 6 — Clean neutral mitigation (val-select / test-eval).

Selection rule on validation:
    maximise neutral recall, subject to macro-F1 drop <= ``--max_f1_drop``.

Override rule applied at test time:
    if pred_label == 'positive' and (proba_positive - proba_neutral) < margin:
        pred_label := 'neutral'

The selected margin is determined ONLY on validation predictions; the test
split is evaluated exactly once with that margin.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from cara_finsent.data_utils import STANDARD_LABELS
from cara_finsent.io_utils import git_commit_sha, timestamp
from cara_finsent.metrics import expected_calibration_error, multiclass_brier_score

PROBA_COLS = [f'proba_{c}' for c in STANDARD_LABELS]


def _apply_margin(df, margin):
    out = df.copy()
    pos = out['pred_label'] == 'positive'
    diff = out['proba_positive'] - out['proba_neutral']
    flip = pos & (diff < margin)
    out.loc[flip, 'pred_label'] = 'neutral'
    out['_overridden'] = flip
    return out


def _evaluate_split(df_aug, label_col='true_label', pred_col='pred_label'):
    y_true = df_aug[label_col].astype(str).to_numpy()
    y_pred = df_aug[pred_col].astype(str).to_numpy()
    P = df_aug[PROBA_COLS].to_numpy(dtype=float)
    cm = confusion_matrix(y_true, y_pred, labels=STANDARD_LABELS)
    n2p = int(cm[STANDARD_LABELS.index('neutral'), STANDARD_LABELS.index('positive')])
    p2n = int(cm[STANDARD_LABELS.index('positive'), STANDARD_LABELS.index('neutral')])
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'neutral_recall': float(recall_score(y_true, y_pred, labels=['neutral'], average='macro', zero_division=0)),
        'neutral_to_positive_errors': n2p,
        'positive_to_neutral_errors': p2n,
        'overridden_count': int(df_aug['_overridden'].sum()),
        'ece_10_bins': float(expected_calibration_error(y_true, y_pred, P, n_bins=10)),
        'brier_score': float(multiclass_brier_score(y_true, P)),
    }


def main():
    parser = argparse.ArgumentParser(description='Phase 14 Step 6 - clean neutral mitigation.')
    parser.add_argument('--val_predictions', required=True)
    parser.add_argument('--test_predictions', required=True)
    parser.add_argument('--margins', default='0.02,0.04,0.06,0.08,0.10,0.20,0.30,0.40,0.50')
    parser.add_argument('--max_f1_drop', type=float, default=0.002)
    parser.add_argument('--model_label', default='aw')
    parser.add_argument('--results_dir', default='results')
    args = parser.parse_args()

    ts = timestamp()
    git_sha = git_commit_sha(PROJECT_ROOT)
    date_subdir = f'{ts[:4]}-{ts[4:6]}-{ts[6:8]}'

    val_df = pd.read_csv(args.val_predictions)
    test_df = pd.read_csv(args.test_predictions)
    print(f'[INFO] val_n={len(val_df)} test_n={len(test_df)}')

    val_df['_overridden'] = False
    val_baseline = _evaluate_split(val_df)
    val_baseline_macro = val_baseline['macro_f1']

    margins = [float(m) for m in args.margins.split(',')]
    rows = []
    rows.append({'split': 'val', 'margin': 0.0, **val_baseline})

    selected = {'margin': 0.0, 'val_neutral_recall': val_baseline['neutral_recall'],
                'val_macro_f1': val_baseline_macro, 'reason': 'baseline (no override)'}

    for m in margins:
        v_aug = _apply_margin(val_df, m)
        v_metrics = _evaluate_split(v_aug)
        v_metrics['split'] = 'val'; v_metrics['margin'] = m
        rows.append({'split': 'val', 'margin': m, **v_metrics})

        f1_drop = val_baseline_macro - v_metrics['macro_f1']
        if f1_drop <= args.max_f1_drop and v_metrics['neutral_recall'] > selected['val_neutral_recall']:
            selected = {
                'margin': m,
                'val_neutral_recall': v_metrics['neutral_recall'],
                'val_macro_f1': v_metrics['macro_f1'],
                'reason': f'val: neutral recall {v_metrics["neutral_recall"]:.4f} '
                          f'(F1 drop {f1_drop:+.4f} <= {args.max_f1_drop})',
            }

    print(f'[SELECT] margin={selected["margin"]:.2f} reason={selected["reason"]}')

    test_df['_overridden'] = False
    test_baseline = _evaluate_split(test_df)
    rows.append({'split': 'test', 'margin': 0.0, **test_baseline})

    if selected['margin'] > 0.0:
        t_aug = _apply_margin(test_df, selected['margin'])
        t_metrics = _evaluate_split(t_aug)
        t_metrics['split'] = 'test'; t_metrics['margin'] = selected['margin']
        rows.append({'split': 'test', 'margin': selected['margin'], **t_metrics})
        cm_test = confusion_matrix(t_aug['true_label'], t_aug['pred_label'], labels=STANDARD_LABELS)
    else:
        t_metrics = test_baseline
        cm_test = confusion_matrix(test_df['true_label'], test_df['pred_label'], labels=STANDARD_LABELS)

    summary = pd.DataFrame(rows)
    summary['model_label'] = args.model_label
    summary['git_commit_sha'] = git_sha
    summary['generated_at_utc'] = ts
    summary['selected_margin'] = selected['margin']
    summary['max_f1_drop_constraint'] = args.max_f1_drop

    out_dir = PROJECT_ROOT / args.results_dir / date_subdir / 'phase14_neutral'
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f'clean_neutral_mitigation_summary_{args.model_label}_{ts}.csv'
    summary.to_csv(summary_path, index=False)
    print(f'\n[OUT] {summary_path}\n')
    cols_view = ['split', 'margin', 'macro_f1', 'neutral_recall',
                 'neutral_to_positive_errors', 'positive_to_neutral_errors', 'overridden_count']
    print(summary[cols_view].to_string(index=False))

    cm_df = pd.DataFrame(cm_test, index=STANDARD_LABELS, columns=STANDARD_LABELS)
    cm_df.index.name = 'true'
    cm_path = out_dir / f'clean_neutral_confusion_{args.model_label}_{ts}.csv'
    cm_df.to_csv(cm_path)
    print(f'[OUT] {cm_path}')

    manifest = {
        'phase': 'phase14_step6_clean_neutral_mitigation',
        'model_label': args.model_label,
        'val_predictions': args.val_predictions,
        'test_predictions': args.test_predictions,
        'val_n': int(len(val_df)),
        'test_n': int(len(test_df)),
        'margins_swept': margins,
        'max_f1_drop_constraint': float(args.max_f1_drop),
        'selected_margin': float(selected['margin']),
        'val_macro_f1_baseline': float(val_baseline_macro),
        'val_macro_f1_selected': float(selected['val_macro_f1']),
        'val_neutral_recall_baseline': float(val_baseline['neutral_recall']),
        'val_neutral_recall_selected': float(selected['val_neutral_recall']),
        'test_macro_f1_baseline': float(test_baseline['macro_f1']),
        'test_macro_f1_selected': float(t_metrics['macro_f1']),
        'test_neutral_recall_selected': float(t_metrics['neutral_recall']),
        'test_neutral_to_positive_errors_selected': int(t_metrics['neutral_to_positive_errors']),
        'fit_rule': 'val-only select, test-only eval',
        'git_commit_sha': git_sha,
        'generated_at_utc': ts,
    }
    manifest_path = out_dir / f'clean_neutral_mitigation_manifest_{args.model_label}_{ts}.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))

    artifacts_dir = PROJECT_ROOT / 'artifacts' / 'phase14_corrected_validation' / 'neutral'
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(artifacts_dir / summary_path.name, index=False)
    cm_df.to_csv(artifacts_dir / cm_path.name)
    (artifacts_dir / manifest_path.name).write_text(json.dumps(manifest, indent=2))
    print(f'[OUT] safe artefacts in {artifacts_dir}')


if __name__ == '__main__':
    main()
