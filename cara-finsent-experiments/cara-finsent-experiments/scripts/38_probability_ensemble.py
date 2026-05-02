#!/usr/bin/env python3
"""Phase 13 Step 5 — Probability-level ensembles of FinBERT family.

Combines per-id probability vectors from:
  * FinBERT zero-shot
  * FinBERT vanilla fine-tuned
  * FinBERT agreement-weighted (AW)

Strategies:
  1. Simple mean of probabilities
  2. Manual weighted average (default 0.5 AW + 0.3 ZS + 0.2 FT)
  3. Grid-searched weights on a stratified calib portion (50/50 split of the
     three predictions joined on id; eval on the held-out half)
  4. Logistic regression meta-classifier on stacked probabilities (calib fit,
     eval evaluation)

Statistical comparison versus AW baseline uses paired bootstrap on macro-F1
and exact McNemar on discordant counts (n=2000 bootstraps, seed=42), reusing
the helpers from `scripts/32_statistical_validation.py`.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cara_finsent.data_utils import STANDARD_LABELS
from cara_finsent.io_utils import git_commit_sha, save_dataframe, save_json, timestamp, write_manifest
from cara_finsent.metrics import expected_calibration_error, multiclass_brier_score


def _load_phase11_helpers():
    spec = importlib.util.spec_from_file_location(
        'phase11_stats', str(PROJECT_ROOT / 'scripts' / '32_statistical_validation.py')
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PROBA_COLS = [f'proba_{c}' for c in STANDARD_LABELS]


def load_probs(path: str, model_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not all(c in df.columns for c in PROBA_COLS):
        raise SystemExit(f'{path} missing one of {PROBA_COLS}')
    df = df[['id', 'label'] + PROBA_COLS].copy()
    df = df.rename(columns={c: f'{model_label}__{c}' for c in PROBA_COLS})
    return df


def evaluate(y_true, y_pred, probs, name):
    return {
        'method': name,
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
        'ece_10_bins': float(expected_calibration_error(y_true, y_pred, probs, n_bins=10)),
        'brier_score': float(multiclass_brier_score(y_true, probs)),
        'mean_confidence': float(probs.max(axis=1).mean()),
    }


def predict_from_probs(probs):
    return np.array([STANDARD_LABELS[int(i)] for i in probs.argmax(axis=1)])


def main():
    parser = argparse.ArgumentParser(description='Phase 13 Step 5 - probability-level ensembles.')
    parser.add_argument('--zs_predictions',
                        default='results/2026-05-02/finbert_baseline_predictions_20260502_081128.csv')
    parser.add_argument('--ft_predictions',
                        default='results/2026-05-02/finbert_baseline_predictions_20260502_081140.csv')
    parser.add_argument('--aw_predictions',
                        default='results/2026-05-02/finbert_agreement_weighted_predictions_20260502_082228.csv')
    parser.add_argument('--manual_weights', default='aw=0.5,zs=0.3,ft=0.2')
    parser.add_argument('--bootstrap_n', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--figures_dir', default='figures')
    args = parser.parse_args()

    ts = timestamp()
    git_sha = git_commit_sha(PROJECT_ROOT)
    date_subdir = f'{ts[:4]}-{ts[4:6]}-{ts[6:8]}'
    figures_dir = Path(args.figures_dir) / date_subdir
    figures_dir.mkdir(parents=True, exist_ok=True)

    zs = load_probs(args.zs_predictions, 'zs')
    ft = load_probs(args.ft_predictions, 'ft')
    aw = load_probs(args.aw_predictions, 'aw')
    print(f'[INFO] rows zs={len(zs)} ft={len(ft)} aw={len(aw)}')

    merged = zs.merge(ft, on=['id', 'label'], how='inner').merge(aw, on=['id', 'label'], how='inner')
    print(f'[INFO] merged rows = {len(merged)}')
    if len(merged) == 0:
        raise SystemExit('No overlapping ids across the three prediction files.')

    y_true = merged['label'].astype(str).to_numpy()
    P_zs = merged[[f'zs__{c}' for c in PROBA_COLS]].to_numpy(dtype=float)
    P_ft = merged[[f'ft__{c}' for c in PROBA_COLS]].to_numpy(dtype=float)
    P_aw = merged[[f'aw__{c}' for c in PROBA_COLS]].to_numpy(dtype=float)

    # Baselines
    rows = []
    for name, P in [('aw_baseline', P_aw), ('zs_baseline', P_zs), ('ft_baseline', P_ft)]:
        rows.append(evaluate(y_true, predict_from_probs(P), P, name))

    # Strategy 1: simple mean
    P_mean = (P_zs + P_ft + P_aw) / 3.0
    rows.append(evaluate(y_true, predict_from_probs(P_mean), P_mean, 'ensemble_simple_mean'))

    # Strategy 2: manual weighted
    weights = {k: float(v) for k, v in (kv.split('=') for kv in args.manual_weights.split(','))}
    w_aw, w_zs, w_ft = weights.get('aw', 0.5), weights.get('zs', 0.3), weights.get('ft', 0.2)
    norm = w_aw + w_zs + w_ft
    w_aw, w_zs, w_ft = w_aw / norm, w_zs / norm, w_ft / norm
    P_manual = w_aw * P_aw + w_zs * P_zs + w_ft * P_ft
    manual_label = f'ensemble_manual_aw{w_aw:.2f}_zs{w_zs:.2f}_ft{w_ft:.2f}'
    rows.append(evaluate(y_true, predict_from_probs(P_manual), P_manual, manual_label))

    # Strategy 3: grid-search weights on calib half
    label_to_idx = {c: i for i, c in enumerate(STANDARD_LABELS)}
    y_idx = np.array([label_to_idx[l] for l in y_true])
    idx_calib, idx_eval = train_test_split(np.arange(len(merged)), test_size=0.5, stratify=y_idx, random_state=args.seed)
    grid = np.arange(0.0, 1.01, 0.1)
    best = {'macro_f1': -1.0, 'w': None}
    for wa, wz in product(grid, grid):
        wf = 1.0 - wa - wz
        if wf < -1e-9 or wf > 1 + 1e-9:
            continue
        wf = max(0.0, wf)
        P_c = wa * P_aw[idx_calib] + wz * P_zs[idx_calib] + wf * P_ft[idx_calib]
        yp = predict_from_probs(P_c)
        f = float(f1_score(y_true[idx_calib], yp, average='macro', zero_division=0))
        if f > best['macro_f1']:
            best = {'macro_f1': f, 'w': (round(float(wa), 2), round(float(wz), 2), round(float(wf), 2))}
    wa, wz, wf = best['w']
    P_grid_eval = wa * P_aw[idx_eval] + wz * P_zs[idx_eval] + wf * P_ft[idx_eval]
    grid_label = f'ensemble_grid_eval_aw{wa:.2f}_zs{wz:.2f}_ft{wf:.2f}'
    grid_eval_metrics = evaluate(y_true[idx_eval], predict_from_probs(P_grid_eval), P_grid_eval, grid_label)
    grid_eval_metrics['eval_n'] = int(len(idx_eval))
    grid_eval_metrics['calib_n'] = int(len(idx_calib))
    grid_eval_metrics['fit_info'] = json.dumps({'best_calib_macro_f1': best['macro_f1'], 'weights': {'aw': wa, 'zs': wz, 'ft': wf}})
    rows.append(grid_eval_metrics)

    # Strategy 4: logistic regression meta-classifier on stacked probs
    X_calib = np.concatenate([P_aw[idx_calib], P_zs[idx_calib], P_ft[idx_calib]], axis=1)
    X_eval = np.concatenate([P_aw[idx_eval], P_zs[idx_eval], P_ft[idx_eval]], axis=1)
    clf = LogisticRegression(max_iter=4000, solver='lbfgs', C=1.0)
    clf.fit(X_calib, y_idx[idx_calib])
    P_stack_eval = clf.predict_proba(X_eval)
    # Reorder columns to STANDARD_LABELS order
    class_order = list(clf.classes_)
    reorder = [class_order.index(label_to_idx[c]) for c in STANDARD_LABELS]
    P_stack_eval = P_stack_eval[:, reorder]
    stack_metrics = evaluate(y_true[idx_eval], predict_from_probs(P_stack_eval), P_stack_eval, 'ensemble_stacked_LR')
    stack_metrics['eval_n'] = int(len(idx_eval))
    stack_metrics['calib_n'] = int(len(idx_calib))
    rows.append(stack_metrics)

    summary_df = pd.DataFrame(rows)
    summary_df['source_zs'] = args.zs_predictions
    summary_df['source_ft'] = args.ft_predictions
    summary_df['source_aw'] = args.aw_predictions
    summary_df['git_commit_sha'] = git_sha
    summary_df['generated_at_utc'] = ts

    summary_path = save_dataframe(summary_df, args.results_dir, 'phase13_ensemble_summary', ts)
    cols_view = ['method', 'accuracy', 'macro_f1', 'weighted_f1', 'mcc', 'ece_10_bins', 'brier_score', 'mean_confidence']
    print(f'\n[OUT] {summary_path}\n')
    print(summary_df[cols_view].to_string(index=False))

    # Statistical comparison vs AW baseline (full-test for full-test methods,
    # eval-half for grid + stacked methods)
    helpers = _load_phase11_helpers()
    rng = np.random.default_rng(args.seed)
    y_pred_aw_full = predict_from_probs(P_aw)
    y_pred_aw_eval = predict_from_probs(P_aw[idx_eval])

    paired_rows = []
    for name, P, idx_subset in [
        ('ensemble_simple_mean', P_mean, None),
        (manual_label, P_manual, None),
        (grid_label, P_grid_eval, idx_eval),
        ('ensemble_stacked_LR', P_stack_eval, idx_eval),
    ]:
        if idx_subset is None:
            yt = y_true
            yp_a = predict_from_probs(P)
            yp_b = y_pred_aw_full
        else:
            yt = y_true[idx_subset]
            yp_a = predict_from_probs(P)
            yp_b = y_pred_aw_eval
        delta, lo, hi, p_boot = helpers.paired_bootstrap_delta_macro_f1(yt, yp_a, yp_b, args.bootstrap_n, rng)
        # McNemar discordant counts: a correct & b wrong vs vice versa
        a_correct = yp_a == yt
        b_correct = yp_b == yt
        b_count = int(((~a_correct) & b_correct).sum())  # AW correct, ensemble wrong
        c_count = int((a_correct & (~b_correct)).sum())  # ensemble correct, AW wrong
        chi2, p_mcn = helpers.mcnemar_exact(b_count, c_count)
        paired_rows.append({
            'comparison': f'{name}_vs_aw_baseline',
            'sample_n': int(len(yt)),
            'delta_macro_f1_a_minus_b': delta,
            'ci_low_95': lo,
            'ci_high_95': hi,
            'paired_bootstrap_p_two_sided': p_boot,
            'mcnemar_b_aw_only_correct': b_count,
            'mcnemar_c_ensemble_only_correct': c_count,
            'mcnemar_chi2': chi2,
            'mcnemar_p_exact_two_sided': p_mcn,
        })
    paired_df = pd.DataFrame(paired_rows)
    paired_df['git_commit_sha'] = git_sha
    paired_df['generated_at_utc'] = ts
    paired_path = save_dataframe(paired_df, args.results_dir, 'phase13_ensemble_paired_comparison', ts)
    print(f'\n[OUT] {paired_path}\n')
    print(paired_df.to_string(index=False))

    # Predictions of all ensemble methods (full-test for eligible methods)
    pred_df = merged[['id', 'label']].copy()
    pred_df['aw_pred'] = predict_from_probs(P_aw)
    pred_df['zs_pred'] = predict_from_probs(P_zs)
    pred_df['ft_pred'] = predict_from_probs(P_ft)
    pred_df['simple_mean_pred'] = predict_from_probs(P_mean)
    pred_df['manual_pred'] = predict_from_probs(P_manual)
    pred_path = save_dataframe(pred_df, args.results_dir, 'phase13_ensemble_predictions', ts)

    # Figure: leaderboard bar chart of macro-F1
    leaderboard = summary_df.copy()
    leaderboard = leaderboard.sort_values('macro_f1', ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(leaderboard['method'], leaderboard['macro_f1'], color='steelblue')
    for i, v in enumerate(leaderboard['macro_f1']):
        ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=8)
    ax.set_xlim(0.85, max(0.92, float(leaderboard['macro_f1'].max()) + 0.01))
    ax.set_xlabel('macro-F1 (PhraseBank test)')
    ax.set_title('Phase 13 — Probability-level ensemble macro-F1')
    fig.tight_layout()
    fig_path = figures_dir / f'phase13_ensemble_leaderboard_{ts}.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[OUT] figure = {fig_path}')

    manifest_meta = {
        'phase': 'phase13_step5_probability_ensemble',
        'sources': {'zs': args.zs_predictions, 'ft': args.ft_predictions, 'aw': args.aw_predictions},
        'merged_n': int(len(merged)),
        'calib_n': int(len(idx_calib)),
        'eval_n': int(len(idx_eval)),
        'split_seed': int(args.seed),
        'bootstrap_n': int(args.bootstrap_n),
        'manual_weights_normalised': {'aw': w_aw, 'zs': w_zs, 'ft': w_ft},
        'grid_best_weights': {'aw': wa, 'zs': wz, 'ft': wf, 'calib_macro_f1': best['macro_f1']},
        'caveats': [
            'grid_search and stacked_LR results are reported on the eval half of a 50/50 stratified split; baselines are also re-evaluated on that subset for the paired test.',
            'simple_mean and manual_weighted are reported on the full merged test set.',
            'No re-training is performed; this is a pure post-hoc combination over frozen probabilities.',
            'AW source is a single seed; ensemble result inherits that seed.',
        ],
        'git_commit_sha': git_sha,
        'generated_at_utc': ts,
    }
    write_manifest(args.results_dir, run_name='phase13_ensemble',
                   files={'summary': str(summary_path), 'paired': str(paired_path),
                          'predictions': str(pred_path), 'figure': str(fig_path)},
                   metadata=manifest_meta, ts=ts)

    artifacts_dir = PROJECT_ROOT / 'artifacts' / 'phase13_extended_validation'
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(artifacts_dir / f'ensemble_summary_{ts}.csv', index=False)
    paired_df.to_csv(artifacts_dir / f'ensemble_paired_comparison_{ts}.csv', index=False)
    save_json(manifest_meta, str(artifacts_dir), 'ensemble_manifest', ts)
    print(f'[OUT] safe artefacts in {artifacts_dir}')


if __name__ == '__main__':
    main()
