#!/usr/bin/env python3
"""Phase 13 Step 3 — Post-hoc calibration of Agreement-Weighted FinBERT.

Applies temperature scaling, Platt-style multi-class calibration (logistic
regression over class probabilities), and per-class isotonic regression to the
AW FinBERT probability outputs on the PhraseBank test split. Because no AW
val-split probabilities exist on disk (Kaggle-trained, weights not synced),
calibrators are fit on a stratified split of the AW *test* probabilities and
evaluated on the held-out portion. This caveat is recorded in every output.

Outputs (results/<date>/):
  - phase13_aw_calibration_comparison_<ts>.csv
  - phase13_aw_calibration_abstention_<ts>.csv
  - phase13_aw_calibration_manifest_<ts>.json
  - figures/<date>/phase13_aw_calibration_reliability_<ts>.png

Usage:
    python scripts/36_calibrate_aw_finbert.py \\
        --aw_predictions results/2026-05-02/finbert_agreement_weighted_predictions_20260502_082228.csv
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
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cara_finsent.data_utils import STANDARD_LABELS
from cara_finsent.io_utils import git_commit_sha, save_dataframe, save_json, timestamp, write_manifest
from cara_finsent.metrics import metrics_with_optional_proba, expected_calibration_error, multiclass_brier_score


def _safe_log(p, eps=1e-12):
    return np.log(np.clip(p, eps, 1.0))


def temperature_scale(probs_calib, y_calib_idx, probs_eval):
    """Fit a single temperature T on calib log-probs (treated as logits) and apply to eval."""
    logits_c = _safe_log(probs_calib)
    logits_e = _safe_log(probs_eval)

    def nll(T):
        T = max(float(T), 1e-3)
        scaled = logits_c / T
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        p = np.exp(scaled)
        p = p / p.sum(axis=1, keepdims=True)
        return -np.mean(_safe_log(p[np.arange(len(y_calib_idx)), y_calib_idx]))

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method='bounded')
    T_star = float(res.x)

    scaled_e = logits_e / T_star
    scaled_e = scaled_e - scaled_e.max(axis=1, keepdims=True)
    p_e = np.exp(scaled_e)
    p_e = p_e / p_e.sum(axis=1, keepdims=True)
    return p_e, {'temperature': T_star, 'nll_calib': float(res.fun)}


def platt_multiclass(probs_calib, y_calib_idx, probs_eval):
    """Fit a multinomial logistic regression on calib probs, apply to eval."""
    # sklearn >=1.5 dropped multi_class kwarg; lbfgs is multinomial by default for >2 classes.
    clf = LogisticRegression(max_iter=2000, solver='lbfgs', C=1.0)
    clf.fit(probs_calib, y_calib_idx)
    p_e = clf.predict_proba(probs_eval)
    return p_e, {'classes': clf.classes_.tolist()}


def isotonic_per_class(probs_calib, y_calib_idx, probs_eval, n_classes=3):
    """Per-class one-vs-rest isotonic on max-prob, then renormalise."""
    calibrated = np.zeros_like(probs_eval)
    irs = []
    for c in range(n_classes):
        ir = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        y_c = (y_calib_idx == c).astype(float)
        ir.fit(probs_calib[:, c], y_c)
        calibrated[:, c] = ir.predict(probs_eval[:, c])
        irs.append(ir)
    # renormalise to a valid distribution
    row_sum = calibrated.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum < 1e-9, 1.0, row_sum)
    calibrated = calibrated / row_sum
    return calibrated, {'method': 'per_class_isotonic_renormalised'}


def evaluate(y_true_labels, y_pred_labels, probs, name):
    return {
        'method': name,
        'accuracy': float(np.mean(np.asarray(y_true_labels) == np.asarray(y_pred_labels))),
        'macro_f1': metrics_with_optional_proba(y_true_labels, y_pred_labels, probs, model_name=name)['macro_f1'],
        'weighted_f1': metrics_with_optional_proba(y_true_labels, y_pred_labels, probs, model_name=name)['weighted_f1'],
        'mcc': metrics_with_optional_proba(y_true_labels, y_pred_labels, probs, model_name=name)['mcc'],
        'ece_10_bins': float(expected_calibration_error(y_true_labels, y_pred_labels, probs, n_bins=10)),
        'brier_score': float(multiclass_brier_score(y_true_labels, probs)),
        'mean_confidence': float(probs.max(axis=1).mean()),
    }


def reliability_curve(y_true_labels, y_pred_labels, probs, n_bins=10):
    conf = probs.max(axis=1)
    correct = (np.asarray(y_true_labels) == np.asarray(y_pred_labels)).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys, ws = [], [], []
    for i in range(n_bins):
        mask = (conf >= bins[i]) & (conf < bins[i + 1] if i < n_bins - 1 else conf <= bins[i + 1])
        if mask.sum() > 0:
            xs.append(conf[mask].mean())
            ys.append(correct[mask].mean())
            ws.append(int(mask.sum()))
    return np.array(xs), np.array(ys), np.array(ws)


def main():
    parser = argparse.ArgumentParser(description='Phase 13 Step 3 — AW post-hoc calibration.')
    parser.add_argument('--aw_predictions',
                        default='results/2026-05-02/finbert_agreement_weighted_predictions_20260502_082228.csv',
                        help='AW predictions CSV with proba_* columns.')
    parser.add_argument('--calib_size', type=float, default=0.5,
                        help='Stratified fraction of AW test rows used to fit calibrators.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--figures_dir', default='figures')
    args = parser.parse_args()

    ts = timestamp()
    git_sha = git_commit_sha(PROJECT_ROOT)
    date_subdir = f'{ts[:4]}-{ts[4:6]}-{ts[6:8]}'
    figures_dir = Path(args.figures_dir) / date_subdir
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.aw_predictions)
    print(f'[INFO] Loaded {len(df)} AW test rows from {args.aw_predictions}')
    proba_cols = [f'proba_{c}' for c in STANDARD_LABELS]
    probs = df[proba_cols].to_numpy(dtype=float)
    y_true_labels = df['label'].astype(str).to_numpy()
    label_to_idx = {c: i for i, c in enumerate(STANDARD_LABELS)}
    y_idx = np.array([label_to_idx[l] for l in y_true_labels])
    y_pred_pre = np.array([STANDARD_LABELS[int(i)] for i in probs.argmax(axis=1)])

    pre_full = evaluate(y_true_labels, y_pred_pre, probs, name='AW_uncalibrated_full_test')
    print(f'[INFO] AW uncalibrated (full test n={len(df)}): macro_f1={pre_full["macro_f1"]:.4f} '
          f'ECE@10={pre_full["ece_10_bins"]:.4f} Brier={pre_full["brier_score"]:.4f}')

    # Stratified split of test into calib / eval (caveat: substitutes for missing AW val).
    idx_calib, idx_eval = train_test_split(
        np.arange(len(df)), test_size=1 - args.calib_size, stratify=y_idx, random_state=args.seed,
    )
    probs_c, probs_e = probs[idx_calib], probs[idx_eval]
    y_idx_c, y_idx_e = y_idx[idx_calib], y_idx[idx_eval]
    y_lbl_e = y_true_labels[idx_eval]

    print(f'[INFO] Calib n={len(idx_calib)} | Eval n={len(idx_eval)}')

    # Pre-calibration metrics on the EVAL portion only (apples-to-apples vs calibrated)
    y_pred_e_pre = np.array([STANDARD_LABELS[int(i)] for i in probs_e.argmax(axis=1)])
    pre_eval = evaluate(y_lbl_e, y_pred_e_pre, probs_e, name='AW_uncalibrated_eval_only')

    # Temperature
    probs_e_T, T_info = temperature_scale(probs_c, y_idx_c, probs_e)
    y_pred_T = np.array([STANDARD_LABELS[int(i)] for i in probs_e_T.argmax(axis=1)])
    metrics_T = evaluate(y_lbl_e, y_pred_T, probs_e_T, name='AW_temperature_scaling')
    metrics_T['fit_info'] = json.dumps(T_info)

    # Platt
    probs_e_P, P_info = platt_multiclass(probs_c, y_idx_c, probs_e)
    y_pred_P = np.array([STANDARD_LABELS[int(i)] for i in probs_e_P.argmax(axis=1)])
    metrics_P = evaluate(y_lbl_e, y_pred_P, probs_e_P, name='AW_platt_multinomial_LR')
    metrics_P['fit_info'] = json.dumps(P_info)

    # Isotonic
    probs_e_I, I_info = isotonic_per_class(probs_c, y_idx_c, probs_e, n_classes=len(STANDARD_LABELS))
    y_pred_I = np.array([STANDARD_LABELS[int(i)] for i in probs_e_I.argmax(axis=1)])
    metrics_I = evaluate(y_lbl_e, y_pred_I, probs_e_I, name='AW_per_class_isotonic')
    metrics_I['fit_info'] = json.dumps(I_info)

    rows = [pre_full, pre_eval, metrics_T, metrics_P, metrics_I]
    for r in rows:
        r['calib_n'] = int(len(idx_calib))
        r['eval_n'] = int(len(idx_eval))
        r['source_predictions'] = args.aw_predictions
        r['git_commit_sha'] = git_sha
        r['generated_at_utc'] = ts
        r['caveat_no_val_split'] = ('AW val-split probabilities are unavailable on disk; calibrators '
                                    'were fit on a stratified split of the AW test predictions.')

    summary_df = pd.DataFrame(rows)
    summary_path = save_dataframe(summary_df, args.results_dir, 'phase13_aw_calibration_comparison', ts)
    print(f'[OUT] {summary_path}')
    print(summary_df[['method', 'eval_n', 'accuracy', 'macro_f1', 'ece_10_bins', 'brier_score', 'mean_confidence']].to_string(index=False))

    # Abstention curve (post-temperature, since usually best for confidence-based abstention)
    abst_rows = []
    for thr in np.arange(0.5, 1.01, 0.05):
        for name, p, yp in [
            ('uncalibrated', probs_e, y_pred_e_pre),
            ('temperature', probs_e_T, y_pred_T),
            ('platt', probs_e_P, y_pred_P),
            ('isotonic', probs_e_I, y_pred_I),
        ]:
            keep = p.max(axis=1) >= thr
            if keep.any():
                acc = float(np.mean(y_lbl_e[keep] == yp[keep]))
            else:
                acc = float('nan')
            abst_rows.append({
                'method': name,
                'threshold': float(thr),
                'coverage': float(keep.mean()),
                'abstention_rate': float(1 - keep.mean()),
                'accuracy_on_kept': acc,
                'kept_count': int(keep.sum()),
            })
    abst_df = pd.DataFrame(abst_rows)
    abst_path = save_dataframe(abst_df, args.results_dir, 'phase13_aw_calibration_abstention', ts)
    print(f'[OUT] {abst_path}')

    # Reliability figure (eval split)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, (name, p, yp) in zip(axes, [
        ('Uncalibrated', probs_e, y_pred_e_pre),
        ('Temperature', probs_e_T, y_pred_T),
        ('Platt', probs_e_P, y_pred_P),
        ('Isotonic', probs_e_I, y_pred_I),
    ]):
        xs, ys, ws = reliability_curve(y_lbl_e, yp, p)
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6, label='Perfect')
        ax.scatter(xs, ys, s=np.sqrt(ws) * 12, alpha=0.7)
        ax.plot(xs, ys, lw=1.2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel('Mean predicted confidence')
        ax.set_title(f'{name}\nECE@10 = {expected_calibration_error(y_lbl_e, yp, p, n_bins=10):.4f}')
    axes[0].set_ylabel('Empirical accuracy')
    fig.suptitle(f'AW FinBERT — Reliability diagram (eval split, n={len(idx_eval)})', fontsize=12)
    fig.tight_layout()
    fig_path = figures_dir / f'phase13_aw_calibration_reliability_{ts}.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[OUT] figure = {fig_path}')

    # Manifest + safe artefacts
    manifest_meta = {
        'phase': 'phase13_step3_aw_calibration',
        'aw_predictions_source': args.aw_predictions,
        'calib_n': int(len(idx_calib)),
        'eval_n': int(len(idx_eval)),
        'calib_size_fraction': float(args.calib_size),
        'split_seed': int(args.seed),
        'methods_evaluated': ['temperature_scaling', 'platt_multinomial_LR', 'per_class_isotonic'],
        'caveats': [
            'No AW val-split probabilities on disk; calibrators fit on stratified split of AW test.',
            'Temperature scaling treats log(probs) as logits since raw logits were not stored.',
            'All metrics are reported on the held-out eval portion of the test split (apples-to-apples).',
        ],
        'git_commit_sha': git_sha,
        'generated_at_utc': ts,
    }
    write_manifest(args.results_dir, run_name='phase13_aw_calibration',
                   files={'summary': str(summary_path), 'abstention': str(abst_path), 'figure': str(fig_path)},
                   metadata=manifest_meta, ts=ts)

    artifacts_dir = PROJECT_ROOT / 'artifacts' / 'phase13_extended_validation'
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(artifacts_dir / f'aw_calibration_comparison_{ts}.csv', index=False)
    abst_df.to_csv(artifacts_dir / f'aw_calibration_abstention_{ts}.csv', index=False)
    save_json(manifest_meta, str(artifacts_dir), 'aw_calibration_manifest', ts)
    print(f'[OUT] safe artefacts in {artifacts_dir}')


if __name__ == '__main__':
    main()
