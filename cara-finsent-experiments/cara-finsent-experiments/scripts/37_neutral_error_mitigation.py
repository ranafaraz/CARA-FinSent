#!/usr/bin/env python3
"""Phase 13 Step 4 — Neutral->positive error mitigation for AW FinBERT.

Phase 8 error analysis identified neutral->positive as the dominant error class
(true label = neutral, predicted = positive; 50 cases on PhraseBank test set
with mean confidence 0.874 on the wrong predictions). This script applies a
simple post-hoc override: when the AW model predicts `positive` but the margin
`proba_positive - proba_neutral` is below a threshold, the prediction is
flipped to `neutral`. Probabilities are NOT renormalised (the override only
changes the decision label); the original calibration of the proba vector is
preserved.

A margin sweep is performed over {0.02, 0.04, 0.06, 0.08, 0.10}. For each
margin we report: macro-F1, accuracy, neutral precision/recall, positive
precision/recall, ECE@10, Brier, overconfident-wrong count
(confidence > 0.80 AND wrong), and the number of overridden predictions.

Usage:
    python scripts/37_neutral_error_mitigation.py
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef

from cara_finsent.data_utils import STANDARD_LABELS
from cara_finsent.io_utils import git_commit_sha, save_dataframe, save_json, timestamp, write_manifest
from cara_finsent.metrics import expected_calibration_error, multiclass_brier_score


def evaluate(y_true, y_pred, probs):
    out = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
        'ece_10_bins': float(expected_calibration_error(y_true, y_pred, probs, n_bins=10)),
        'brier_score': float(multiclass_brier_score(y_true, probs)),
    }
    for cls in STANDARD_LABELS:
        out[f'precision_{cls}'] = float(precision_score(y_true, y_pred, labels=[cls], average='macro', zero_division=0))
        out[f'recall_{cls}'] = float(recall_score(y_true, y_pred, labels=[cls], average='macro', zero_division=0))
    # neutral->positive specifically
    arr_t = np.asarray(y_true)
    arr_p = np.asarray(y_pred)
    out['neutral_to_positive_errors'] = int(((arr_t == 'neutral') & (arr_p == 'positive')).sum())
    out['positive_to_neutral_errors'] = int(((arr_t == 'positive') & (arr_p == 'neutral')).sum())
    # overconfident-wrong
    conf = probs.max(axis=1)
    wrong = arr_t != arr_p
    out['overconfident_wrong_count'] = int(((conf > 0.80) & wrong).sum())
    out['mean_confidence_wrong'] = float(conf[wrong].mean()) if wrong.any() else 0.0
    return out


def apply_margin_override(df, margin):
    """Override prediction to 'neutral' when prediction=='positive' and (p_pos - p_neu) < margin."""
    pred_new = df['prediction'].copy().to_numpy()
    margins_pos_neu = (df['proba_positive'] - df['proba_neutral']).to_numpy()
    mask = (pred_new == 'positive') & (margins_pos_neu < margin)
    pred_new[mask] = 'neutral'
    return pred_new, int(mask.sum())


def main():
    parser = argparse.ArgumentParser(description='Phase 13 Step 4 - neutral->positive error mitigation.')
    parser.add_argument('--aw_predictions',
                        default='results/2026-05-02/finbert_agreement_weighted_predictions_20260502_082228.csv')
    parser.add_argument('--margins', default='0.02,0.04,0.06,0.08,0.10',
                        help='Comma-separated list of margins to sweep.')
    parser.add_argument('--results_dir', default='results')
    args = parser.parse_args()

    ts = timestamp()
    git_sha = git_commit_sha(PROJECT_ROOT)

    df = pd.read_csv(args.aw_predictions)
    proba_cols = [f'proba_{c}' for c in STANDARD_LABELS]
    probs = df[proba_cols].to_numpy(dtype=float)
    y_true = df['label'].astype(str).to_numpy()
    y_pred_base = df['prediction'].astype(str).to_numpy()
    print(f'[INFO] Loaded {len(df)} AW predictions from {args.aw_predictions}')

    rows = []
    base = evaluate(y_true, y_pred_base, probs)
    base.update({'margin': 0.0, 'overridden_count': 0, 'method': 'baseline'})
    rows.append(base)
    print(f'[BASELINE] macro_f1={base["macro_f1"]:.4f} '
          f'neutral_recall={base["recall_neutral"]:.4f} positive_precision={base["precision_positive"]:.4f} '
          f'neutral->positive_errors={base["neutral_to_positive_errors"]} '
          f'overconfident_wrong={base["overconfident_wrong_count"]}')

    margins = [float(m) for m in args.margins.split(',') if m.strip()]
    for m in margins:
        y_pred_new, n_over = apply_margin_override(df, m)
        metrics = evaluate(y_true, y_pred_new, probs)
        metrics.update({'margin': m, 'overridden_count': n_over,
                        'method': f'pos_to_neu_override_margin_{m:.2f}'})
        rows.append(metrics)

    summary_df = pd.DataFrame(rows)
    summary_df['source_predictions'] = args.aw_predictions
    summary_df['git_commit_sha'] = git_sha
    summary_df['generated_at_utc'] = ts

    summary_path = save_dataframe(summary_df, args.results_dir, 'phase13_neutral_mitigation_summary', ts)
    cols_view = ['method', 'margin', 'overridden_count', 'accuracy', 'macro_f1',
                 'recall_neutral', 'precision_positive',
                 'neutral_to_positive_errors', 'positive_to_neutral_errors',
                 'overconfident_wrong_count', 'ece_10_bins', 'brier_score']
    print(f'\n[OUT] {summary_path}\n')
    print(summary_df[cols_view].to_string(index=False))

    # Manifest + safe artefacts
    manifest_meta = {
        'phase': 'phase13_step4_neutral_mitigation',
        'aw_predictions_source': args.aw_predictions,
        'margins_swept': margins,
        'override_rule': "prediction == 'positive' AND (proba_positive - proba_neutral) < margin -> 'neutral'",
        'note_probs_unchanged': 'Only the argmax label is overridden; probability vector is preserved as-is so calibration metrics remain comparable.',
        'baseline_neutral_to_positive_errors': int(base['neutral_to_positive_errors']),
        'baseline_macro_f1': float(base['macro_f1']),
        'git_commit_sha': git_sha,
        'generated_at_utc': ts,
        'caveats': [
            'Override is decision-only (probs unchanged); ECE differences across margins reflect the bin reassignment of overridden cases, not a true recalibration.',
            'Single-checkpoint AW predictions; not seed-averaged.',
            'Best margin is selected on test data; for an honest deployment the margin should be tuned on a held-out val/calib split.',
        ],
    }
    write_manifest(args.results_dir, run_name='phase13_neutral_mitigation',
                   files={'summary': str(summary_path)}, metadata=manifest_meta, ts=ts)

    artifacts_dir = PROJECT_ROOT / 'artifacts' / 'phase13_extended_validation'
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(artifacts_dir / f'neutral_mitigation_summary_{ts}.csv', index=False)
    save_json(manifest_meta, str(artifacts_dir), 'neutral_mitigation_manifest', ts)
    print(f'\n[OUT] safe artefacts in {artifacts_dir}')


if __name__ == '__main__':
    main()
