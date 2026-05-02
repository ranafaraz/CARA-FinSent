#!/usr/bin/env python3
"""Phase 14 Step 4 — Clean post-hoc calibration.

Validation-fit / test-evaluate. No test data is used during calibrator fit.

Methods: temperature scaling, Platt multinomial logistic regression,
per-class isotonic regression. Reports ECE@10, Brier, accuracy and macro-F1
on test for every method and the uncalibrated baseline.

Required inputs are the Phase 14 prediction CSVs produced by
``scripts/41_generate_val_test_predictions.py`` (val + test for one model).
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
from sklearn.metrics import accuracy_score, f1_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cara_finsent.data_utils import STANDARD_LABELS
from cara_finsent.io_utils import git_commit_sha, timestamp
from cara_finsent.metrics import expected_calibration_error, multiclass_brier_score

PROBA_COLS = [f'proba_{c}' for c in STANDARD_LABELS]
LABEL2IDX = {c: i for i, c in enumerate(STANDARD_LABELS)}


def _load_pred(path):
    df = pd.read_csv(path)
    P = df[PROBA_COLS].to_numpy(dtype=float)
    y = df['true_label'].astype(str).to_numpy()
    return df, P, y


def _evaluate(y_true, P, name):
    pred = np.array([STANDARD_LABELS[int(i)] for i in P.argmax(axis=1)])
    return {
        'method': name,
        'accuracy': float(accuracy_score(y_true, pred)),
        'macro_f1': float(f1_score(y_true, pred, average='macro', zero_division=0)),
        'ece_10_bins': float(expected_calibration_error(y_true, pred, P, n_bins=10)),
        'brier_score': float(multiclass_brier_score(y_true, P)),
        'mean_confidence': float(P.max(axis=1).mean()),
    }


def _temperature_scale_fit(P_val, y_val):
    eps = 1e-9
    log_p = np.log(np.clip(P_val, eps, 1.0))
    y_idx = np.array([LABEL2IDX[c] for c in y_val])

    def nll(T):
        scaled = log_p / float(T)
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        ex = np.exp(scaled)
        sm = ex / ex.sum(axis=1, keepdims=True)
        return -np.log(np.clip(sm[np.arange(len(y_idx)), y_idx], eps, 1.0)).mean()

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method='bounded')
    return float(res.x)


def _temperature_scale_apply(P, T):
    log_p = np.log(np.clip(P, 1e-9, 1.0)) / T
    log_p -= log_p.max(axis=1, keepdims=True)
    ex = np.exp(log_p)
    return ex / ex.sum(axis=1, keepdims=True)


def _platt_fit(P_val, y_val):
    y_idx = np.array([LABEL2IDX[c] for c in y_val])
    log_p = np.log(np.clip(P_val, 1e-9, 1.0))
    clf = LogisticRegression(max_iter=4000, solver='lbfgs', C=1.0)
    clf.fit(log_p, y_idx)
    return clf


def _platt_apply(P, clf):
    log_p = np.log(np.clip(P, 1e-9, 1.0))
    out = clf.predict_proba(log_p)
    classes = list(clf.classes_)
    reorder = [classes.index(LABEL2IDX[c]) for c in STANDARD_LABELS]
    return out[:, reorder]


def _isotonic_fit(P_val, y_val):
    isos = []
    y_idx = np.array([LABEL2IDX[c] for c in y_val])
    for k in range(len(STANDARD_LABELS)):
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        iso.fit(P_val[:, k], (y_idx == k).astype(int))
        isos.append(iso)
    return isos


def _isotonic_apply(P, isos):
    out = np.zeros_like(P)
    for k, iso in enumerate(isos):
        out[:, k] = iso.predict(P[:, k])
    out = np.clip(out, 1e-9, 1.0)
    return out / out.sum(axis=1, keepdims=True)


def _reliability_curve(y_true, P, n_bins=10):
    pred_idx = P.argmax(axis=1)
    conf = P.max(axis=1)
    correct = np.array([STANDARD_LABELS[int(i)] for i in pred_idx]) == y_true
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if mask.sum() > 0:
            centers.append(float(conf[mask].mean()))
            accs.append(float(correct[mask].mean()))
            counts.append(int(mask.sum()))
        else:
            centers.append(float((lo + hi) / 2))
            accs.append(np.nan)
            counts.append(0)
    return np.array(centers), np.array(accs), np.array(counts)


def main():
    parser = argparse.ArgumentParser(description='Phase 14 Step 4 - clean calibration (val-fit / test-eval).')
    parser.add_argument('--val_predictions', required=True, help='Phase 14 val predictions CSV')
    parser.add_argument('--test_predictions', required=True, help='Phase 14 test predictions CSV')
    parser.add_argument('--model_label', default='aw', help='Short label used in output names')
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--figures_dir', default='figures')
    args = parser.parse_args()

    ts = timestamp()
    git_sha = git_commit_sha(PROJECT_ROOT)
    date_subdir = f'{ts[:4]}-{ts[4:6]}-{ts[6:8]}'

    val_df, P_val, y_val = _load_pred(args.val_predictions)
    test_df, P_test, y_test = _load_pred(args.test_predictions)
    print(f'[INFO] val_n={len(val_df)} test_n={len(test_df)}')

    rows = []
    rows.append({**_evaluate(y_test, P_test, f'{args.model_label}_uncalibrated'),
                 'fit_split': 'none', 'fit_n': 0})

    T = _temperature_scale_fit(P_val, y_val)
    P_test_T = _temperature_scale_apply(P_test, T)
    rows.append({**_evaluate(y_test, P_test_T, f'{args.model_label}_temperature'),
                 'fit_split': 'val', 'fit_n': int(len(val_df)), 'temperature': float(T)})

    platt = _platt_fit(P_val, y_val)
    P_test_P = _platt_apply(P_test, platt)
    rows.append({**_evaluate(y_test, P_test_P, f'{args.model_label}_platt'),
                 'fit_split': 'val', 'fit_n': int(len(val_df))})

    isos = _isotonic_fit(P_val, y_val)
    P_test_I = _isotonic_apply(P_test, isos)
    rows.append({**_evaluate(y_test, P_test_I, f'{args.model_label}_isotonic'),
                 'fit_split': 'val', 'fit_n': int(len(val_df))})

    summary = pd.DataFrame(rows)
    summary['git_commit_sha'] = git_sha
    summary['generated_at_utc'] = ts
    summary['val_predictions_source'] = args.val_predictions
    summary['test_predictions_source'] = args.test_predictions

    out_dir = PROJECT_ROOT / args.results_dir / date_subdir / 'phase14_calibration'
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f'clean_calibration_summary_{args.model_label}_{ts}.csv'
    summary.to_csv(summary_path, index=False)
    print(f'\n[OUT] {summary_path}\n')
    cols_view = ['method', 'accuracy', 'macro_f1', 'ece_10_bins', 'brier_score', 'mean_confidence', 'fit_split']
    print(summary[cols_view].to_string(index=False))

    # Abstention curve on best-calibrated method (lowest ECE)
    best_idx = int(np.argmin(summary['ece_10_bins'].values))
    method_name = summary.loc[best_idx, 'method']
    P_best = [P_test, P_test_T, P_test_P, P_test_I][best_idx]
    pred_best = np.array([STANDARD_LABELS[int(i)] for i in P_best.argmax(axis=1)])
    conf_best = P_best.max(axis=1)
    correct = pred_best == y_test
    thresholds = np.linspace(0.0, 0.99, 50)
    abst_rows = []
    for t in thresholds:
        m = conf_best >= t
        cov = float(m.mean())
        acc = float(correct[m].mean()) if m.sum() > 0 else float('nan')
        abst_rows.append({'threshold': float(t), 'coverage': cov, 'accuracy_at_coverage': acc, 'n_kept': int(m.sum())})
    abst_df = pd.DataFrame(abst_rows)
    abst_df['method'] = method_name
    abst_path = out_dir / f'clean_abstention_curve_{args.model_label}_{ts}.csv'
    abst_df.to_csv(abst_path, index=False)
    print(f'\n[OUT] {abst_path}')

    # Reliability diagram
    figures_dir = PROJECT_ROOT / args.figures_dir / date_subdir
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / f'phase14_clean_calibration_reliability_{args.model_label}_{ts}.png'
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='perfect')
    for label, P in [('uncalibrated', P_test), ('temperature', P_test_T),
                     ('platt', P_test_P), ('isotonic', P_test_I)]:
        c, a, _ = _reliability_curve(y_test, P, n_bins=10)
        mask = ~np.isnan(a)
        ax.plot(c[mask], a[mask], marker='o', label=label, alpha=0.85)
    ax.set_xlabel('Predicted confidence')
    ax.set_ylabel('Empirical accuracy')
    ax.set_title(f'Phase 14 clean calibration — {args.model_label} (val-fit / test-eval)')
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[OUT] {fig_path}')

    manifest = {
        'phase': 'phase14_step4_clean_calibration',
        'model_label': args.model_label,
        'val_predictions': args.val_predictions,
        'test_predictions': args.test_predictions,
        'val_n': int(len(val_df)),
        'test_n': int(len(test_df)),
        'best_method': method_name,
        'best_ece_10_bins': float(summary.loc[best_idx, 'ece_10_bins']),
        'temperature': float(T),
        'fit_rule': 'val-only fit, test-only eval',
        'git_commit_sha': git_sha,
        'generated_at_utc': ts,
    }
    manifest_path = out_dir / f'clean_calibration_manifest_{args.model_label}_{ts}.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))

    artifacts_dir = PROJECT_ROOT / 'artifacts' / 'phase14_corrected_validation' / 'calibration'
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(artifacts_dir / summary_path.name, index=False)
    abst_df.to_csv(artifacts_dir / abst_path.name, index=False)
    (artifacts_dir / manifest_path.name).write_text(json.dumps(manifest, indent=2))
    print(f'[OUT] safe artefacts in {artifacts_dir}')


if __name__ == '__main__':
    main()
