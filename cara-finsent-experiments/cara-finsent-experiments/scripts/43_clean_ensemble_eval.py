#!/usr/bin/env python3
"""Phase 14 Step 5 — Clean probability-level ensembles (val-tune / test-eval).

Inputs are the Phase 14 prediction CSVs from
``scripts/41_generate_val_test_predictions.py`` for ZS, FT, and AW. Strategies:

  1. Simple mean of probabilities (no tuning).
  2. Manual fixed weights (default AW=0.5, ZS=0.3, FT=0.2; renormalised).
  3. Validation-tuned weighted average (grid step 0.1, sum-to-1 constraint).
  4. Logistic-regression stacking trained on validation predictions.

Final evaluation is a single pass on the held-out test predictions. Paired
bootstrap and exact McNemar tests vs an AW baseline are computed using the
helpers from ``scripts/32_statistical_validation.py``.
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cara_finsent.data_utils import STANDARD_LABELS
from cara_finsent.io_utils import git_commit_sha, timestamp
from cara_finsent.metrics import expected_calibration_error, multiclass_brier_score

PROBA_COLS = [f'proba_{c}' for c in STANDARD_LABELS]
LABEL2IDX = {c: i for i, c in enumerate(STANDARD_LABELS)}


def _load_phase11_helpers():
    spec = importlib.util.spec_from_file_location(
        'phase11_stats', str(PROJECT_ROOT / 'scripts' / '32_statistical_validation.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_pair(val_path, test_path, prefix):
    v = pd.read_csv(val_path)[['text_hash', 'true_label'] + PROBA_COLS].copy()
    t = pd.read_csv(test_path)[['text_hash', 'true_label'] + PROBA_COLS].copy()
    v = v.rename(columns={c: f'{prefix}__{c}' for c in PROBA_COLS})
    t = t.rename(columns={c: f'{prefix}__{c}' for c in PROBA_COLS})
    return v, t


def _to_pred(P):
    return np.array([STANDARD_LABELS[int(i)] for i in P.argmax(axis=1)])


def _evaluate(y_true, P, name):
    pred = _to_pred(P)
    return {
        'method': name,
        'accuracy': float(accuracy_score(y_true, pred)),
        'macro_f1': float(f1_score(y_true, pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, pred, average='weighted', zero_division=0)),
        'mcc': float(matthews_corrcoef(y_true, pred)),
        'ece_10_bins': float(expected_calibration_error(y_true, pred, P, n_bins=10)),
        'brier_score': float(multiclass_brier_score(y_true, P)),
        'mean_confidence': float(P.max(axis=1).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description='Phase 14 Step 5 - clean ensemble (val-tune / test-eval).')
    parser.add_argument('--zs_val', required=True)
    parser.add_argument('--zs_test', required=True)
    parser.add_argument('--ft_val', required=True)
    parser.add_argument('--ft_test', required=True)
    parser.add_argument('--aw_val', required=True)
    parser.add_argument('--aw_test', required=True)
    parser.add_argument('--manual_weights', default='aw=0.5,zs=0.3,ft=0.2')
    parser.add_argument('--bootstrap_n', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--figures_dir', default='figures')
    args = parser.parse_args()

    ts = timestamp()
    git_sha = git_commit_sha(PROJECT_ROOT)
    date_subdir = f'{ts[:4]}-{ts[4:6]}-{ts[6:8]}'

    zs_v, zs_t = _load_pair(args.zs_val, args.zs_test, 'zs')
    ft_v, ft_t = _load_pair(args.ft_val, args.ft_test, 'ft')
    aw_v, aw_t = _load_pair(args.aw_val, args.aw_test, 'aw')

    val = zs_v.merge(ft_v, on=['text_hash', 'true_label']).merge(aw_v, on=['text_hash', 'true_label'])
    test = zs_t.merge(ft_t, on=['text_hash', 'true_label']).merge(aw_t, on=['text_hash', 'true_label'])
    print(f'[INFO] merged val={len(val)} test={len(test)}')

    def _split_arrays(df):
        y = df['true_label'].astype(str).to_numpy()
        Pa = df[[f'aw__{c}' for c in PROBA_COLS]].to_numpy(dtype=float)
        Pz = df[[f'zs__{c}' for c in PROBA_COLS]].to_numpy(dtype=float)
        Pf = df[[f'ft__{c}' for c in PROBA_COLS]].to_numpy(dtype=float)
        return y, Pa, Pz, Pf

    y_v, Pa_v, Pz_v, Pf_v = _split_arrays(val)
    y_t, Pa_t, Pz_t, Pf_t = _split_arrays(test)

    rows = []
    for name, P in [('aw_baseline', Pa_t), ('zs_baseline', Pz_t), ('ft_baseline', Pf_t)]:
        rows.append({**_evaluate(y_t, P, name), 'tune_split': 'none'})

    # 1. Simple mean
    Pmean_t = (Pa_t + Pz_t + Pf_t) / 3.0
    rows.append({**_evaluate(y_t, Pmean_t, 'ensemble_simple_mean'), 'tune_split': 'none'})

    # 2. Manual fixed weights
    weights = {k: float(v) for k, v in (kv.split('=') for kv in args.manual_weights.split(','))}
    wa, wz, wf = weights.get('aw', 0.5), weights.get('zs', 0.3), weights.get('ft', 0.2)
    s = wa + wz + wf; wa, wz, wf = wa/s, wz/s, wf/s
    Pman_t = wa * Pa_t + wz * Pz_t + wf * Pf_t
    rows.append({**_evaluate(y_t, Pman_t, f'ensemble_manual_aw{wa:.2f}_zs{wz:.2f}_ft{wf:.2f}'),
                 'tune_split': 'a-priori'})

    # 3. Validation-tuned weighted average (grid 0.1)
    grid = np.arange(0.0, 1.01, 0.1)
    best = {'macro_f1': -1.0, 'w': None}
    for ga, gz in product(grid, grid):
        gf = 1.0 - ga - gz
        if gf < -1e-9 or gf > 1 + 1e-9:
            continue
        gf = max(0.0, gf)
        P_v = ga * Pa_v + gz * Pz_v + gf * Pf_v
        f = float(f1_score(y_v, _to_pred(P_v), average='macro', zero_division=0))
        if f > best['macro_f1']:
            best = {'macro_f1': f, 'w': (round(float(ga), 2), round(float(gz), 2), round(float(gf), 2))}
    ga, gz, gf = best['w']
    Pgrid_t = ga * Pa_t + gz * Pz_t + gf * Pf_t
    grid_label = f'ensemble_grid_val_aw{ga:.2f}_zs{gz:.2f}_ft{gf:.2f}'
    grid_metrics = _evaluate(y_t, Pgrid_t, grid_label)
    grid_metrics['tune_split'] = 'val'
    grid_metrics['tune_macro_f1_on_val'] = float(best['macro_f1'])
    rows.append(grid_metrics)

    # 4. Logistic-regression stacking on val
    Xv = np.concatenate([Pa_v, Pz_v, Pf_v], axis=1)
    Xt = np.concatenate([Pa_t, Pz_t, Pf_t], axis=1)
    yv_idx = np.array([LABEL2IDX[c] for c in y_v])
    clf = LogisticRegression(max_iter=4000, solver='lbfgs', C=1.0)
    clf.fit(Xv, yv_idx)
    Pstack_t = clf.predict_proba(Xt)
    classes = list(clf.classes_)
    reorder = [classes.index(LABEL2IDX[c]) for c in STANDARD_LABELS]
    Pstack_t = Pstack_t[:, reorder]
    stack_metrics = _evaluate(y_t, Pstack_t, 'ensemble_stacked_LR')
    stack_metrics['tune_split'] = 'val'
    rows.append(stack_metrics)

    summary = pd.DataFrame(rows)
    summary['git_commit_sha'] = git_sha
    summary['generated_at_utc'] = ts
    summary['val_n'] = int(len(val))
    summary['test_n'] = int(len(test))

    out_dir = PROJECT_ROOT / args.results_dir / date_subdir / 'phase14_ensemble'
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f'clean_ensemble_summary_{ts}.csv'
    summary.to_csv(summary_path, index=False)
    print(f'\n[OUT] {summary_path}\n')
    cols_view = ['method', 'accuracy', 'macro_f1', 'weighted_f1', 'mcc',
                 'ece_10_bins', 'brier_score', 'mean_confidence', 'tune_split']
    print(summary[cols_view].to_string(index=False))

    helpers = _load_phase11_helpers()
    rng = np.random.default_rng(args.seed)
    yp_aw = _to_pred(Pa_t)
    paired = []
    for name, P in [('ensemble_simple_mean', Pmean_t),
                    (f'ensemble_manual_aw{wa:.2f}_zs{wz:.2f}_ft{wf:.2f}', Pman_t),
                    (grid_label, Pgrid_t),
                    ('ensemble_stacked_LR', Pstack_t)]:
        yp = _to_pred(P)
        delta, lo, hi, p_boot = helpers.paired_bootstrap_delta_macro_f1(y_t, yp, yp_aw, args.bootstrap_n, rng)
        a_correct = yp == y_t
        b_correct = yp_aw == y_t
        b_count = int(((~a_correct) & b_correct).sum())
        c_count = int((a_correct & (~b_correct)).sum())
        chi2, p_mcn = helpers.mcnemar_exact(b_count, c_count)
        paired.append({
            'comparison': f'{name}_vs_aw_baseline',
            'sample_n': int(len(y_t)),
            'delta_macro_f1_a_minus_b': delta,
            'ci_low_95': lo, 'ci_high_95': hi,
            'paired_bootstrap_p_two_sided': p_boot,
            'mcnemar_b_aw_only_correct': b_count,
            'mcnemar_c_ensemble_only_correct': c_count,
            'mcnemar_chi2': chi2,
            'mcnemar_p_exact_two_sided': p_mcn,
        })
    paired_df = pd.DataFrame(paired)
    paired_df['git_commit_sha'] = git_sha
    paired_df['generated_at_utc'] = ts
    paired_path = out_dir / f'clean_ensemble_paired_comparison_{ts}.csv'
    paired_df.to_csv(paired_path, index=False)
    print(f'\n[OUT] {paired_path}\n')
    print(paired_df.to_string(index=False))

    # Figure
    figures_dir = PROJECT_ROOT / args.figures_dir / date_subdir
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / f'phase14_clean_ensemble_leaderboard_{ts}.png'
    leaderboard = summary.sort_values('macro_f1')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(leaderboard['method'], leaderboard['macro_f1'], color='steelblue')
    for i, v in enumerate(leaderboard['macro_f1']):
        ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=8)
    ax.set_xlim(0.85, max(0.92, float(leaderboard['macro_f1'].max()) + 0.01))
    ax.set_xlabel('macro-F1 (PhraseBank test, single pass)')
    ax.set_title('Phase 14 — clean ensemble macro-F1 (val-tuned, test-evaluated)')
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[OUT] {fig_path}')

    manifest = {
        'phase': 'phase14_step5_clean_ensemble',
        'sources': {
            'zs': {'val': args.zs_val, 'test': args.zs_test},
            'ft': {'val': args.ft_val, 'test': args.ft_test},
            'aw': {'val': args.aw_val, 'test': args.aw_test},
        },
        'val_n': int(len(val)),
        'test_n': int(len(test)),
        'manual_weights_normalised': {'aw': wa, 'zs': wz, 'ft': wf},
        'val_tuned_grid_weights': {'aw': ga, 'zs': gz, 'ft': gf, 'val_macro_f1': best['macro_f1']},
        'bootstrap_n': int(args.bootstrap_n),
        'split_seed': int(args.seed),
        'fit_rule': 'val-only tune, test-only eval',
        'git_commit_sha': git_sha,
        'generated_at_utc': ts,
    }
    manifest_path = out_dir / f'clean_ensemble_manifest_{ts}.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))

    artifacts_dir = PROJECT_ROOT / 'artifacts' / 'phase14_corrected_validation' / 'ensemble'
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(artifacts_dir / summary_path.name, index=False)
    paired_df.to_csv(artifacts_dir / paired_path.name, index=False)
    (artifacts_dir / manifest_path.name).write_text(json.dumps(manifest, indent=2))
    print(f'[OUT] safe artefacts in {artifacts_dir}')


if __name__ == '__main__':
    main()
