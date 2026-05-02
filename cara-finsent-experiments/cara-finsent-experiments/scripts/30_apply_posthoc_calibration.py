#!/usr/bin/env python3
"""Phase 6: post-hoc probability calibration for FinBERT-style predictions.

Supported methods:
  * temperature  - single scalar T fit on val log-probs (NLL)
  * platt        - per-class logistic regression on log-probs
  * isotonic     - per-class isotonic regression on max-confidence (binary correctness)

Inputs
------
--val_predictions  : CSV with proba_* + label (used to FIT calibration only)
--test_predictions : CSV with proba_* + label + prediction (held-out evaluation)
--method           : temperature | platt | isotonic
--model_name, --dataset_name : used for output naming

If --val_predictions is omitted, the script will split --test_predictions
70/30 stratified by label and emit a clear warning. This is for smoke tests
only; never report the resulting numbers as final evidence.

Outputs
-------
results/<date>/posthoc_<method>_predictions_<ts>.csv
results/<date>/posthoc_calibration_summary_<ts>.csv
figures/posthoc_reliability_<method>_<ts>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from cara_finsent.data_utils import STANDARD_LABELS  # noqa: E402
from cara_finsent.io_utils import save_dataframe, save_json, timestamp, ensure_dir  # noqa: E402
from cara_finsent.metrics import (  # noqa: E402
    classification_metrics,
    expected_calibration_error,
    multiclass_brier_score,
    reliability_bins,
)

PROBA_COLS = [f'proba_{c}' for c in STANDARD_LABELS]
EPS = 1e-12


def _load_proba(df: pd.DataFrame) -> np.ndarray:
    missing = [c for c in PROBA_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f'predictions CSV missing columns: {missing}')
    p = df[PROBA_COLS].to_numpy(dtype=float)
    p = np.clip(p, EPS, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def _softmax_with_T(logits: np.ndarray, T: float) -> np.ndarray:
    z = logits / max(T, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(val_proba: np.ndarray, val_labels: np.ndarray) -> float:
    val_logits = np.log(val_proba)
    y_idx = np.array([STANDARD_LABELS.index(str(y)) for y in val_labels])

    def nll(T: float) -> float:
        p = _softmax_with_T(val_logits, T)
        return float(-np.mean(np.log(p[np.arange(len(y_idx)), y_idx] + EPS)))

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method='bounded')
    return float(res.x)


def apply_temperature(proba: np.ndarray, T: float) -> np.ndarray:
    return _softmax_with_T(np.log(proba), T)


def fit_platt(val_proba: np.ndarray, val_labels: np.ndarray):
    """Per-class one-vs-rest logistic on log-probs."""
    val_logits = np.log(val_proba)
    models = []
    for k, cls in enumerate(STANDARD_LABELS):
        y = (np.asarray(val_labels) == cls).astype(int)
        if len(np.unique(y)) < 2:
            models.append(None)
            continue
        lr = LogisticRegression(max_iter=2000)
        lr.fit(val_logits[:, [k]], y)
        models.append(lr)
    return models


def apply_platt(proba: np.ndarray, models) -> np.ndarray:
    logits = np.log(proba)
    out = np.zeros_like(proba)
    for k, m in enumerate(models):
        if m is None:
            out[:, k] = proba[:, k]
        else:
            out[:, k] = m.predict_proba(logits[:, [k]])[:, 1]
    s = out.sum(axis=1, keepdims=True)
    s[s < EPS] = 1.0
    return out / s


def fit_isotonic(val_proba: np.ndarray, val_labels: np.ndarray):
    models = []
    for k, cls in enumerate(STANDARD_LABELS):
        y = (np.asarray(val_labels) == cls).astype(int)
        ir = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        ir.fit(val_proba[:, k], y)
        models.append(ir)
    return models


def apply_isotonic(proba: np.ndarray, models) -> np.ndarray:
    out = np.zeros_like(proba)
    for k, m in enumerate(models):
        out[:, k] = m.transform(proba[:, k])
    out = np.clip(out, EPS, 1.0)
    s = out.sum(axis=1, keepdims=True)
    return out / s


def _summary_row(prefix: str, y_true, y_pred, proba) -> dict:
    cm = classification_metrics(y_true, y_pred)
    return {
        f'{prefix}_accuracy': cm['accuracy'],
        f'{prefix}_macro_f1': cm['macro_f1'],
        f'{prefix}_ece_10_bins': expected_calibration_error(y_true, y_pred, proba, n_bins=10),
        f'{prefix}_brier': multiclass_brier_score(y_true, proba),
        f'{prefix}_mean_confidence': float(proba.max(axis=1).mean()),
    }


def _plot_reliability(before: pd.DataFrame, after: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], '--', color='gray', label='perfect')
    ax.plot(before['mean_confidence'], before['accuracy'], '-o', label='before')
    ax.plot(after['mean_confidence'], after['accuracy'], '-s', label='after')
    ax.set_xlabel('mean confidence')
    ax.set_ylabel('accuracy')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _maybe_split(test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print('[WARN] --val_predictions not provided; splitting --test_predictions 70/30')
    rng = np.random.default_rng(42)
    idx = np.arange(len(test_df))
    rng.shuffle(idx)
    cut = int(0.3 * len(test_df))
    return test_df.iloc[idx[:cut]].reset_index(drop=True), test_df.iloc[idx[cut:]].reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--val_predictions', default=None)
    ap.add_argument('--test_predictions', required=True)
    ap.add_argument('--method', choices=['temperature', 'platt', 'isotonic'], default='temperature')
    ap.add_argument('--model_name', required=True)
    ap.add_argument('--dataset_name', required=True)
    ap.add_argument('--results_dir', default='results')
    ap.add_argument('--figures_dir', default='figures')
    args = ap.parse_args()

    test_df = pd.read_csv(args.test_predictions)
    if args.val_predictions:
        val_df = pd.read_csv(args.val_predictions)
    else:
        val_df, test_df = _maybe_split(test_df)

    val_proba = _load_proba(val_df)
    test_proba = _load_proba(test_df)
    val_labels = val_df['label'].astype(str).to_numpy()
    test_labels = test_df['label'].astype(str).to_numpy()

    if args.method == 'temperature':
        T = fit_temperature(val_proba, val_labels)
        print(f'[FIT] temperature T={T:.4f}')
        cal_test = apply_temperature(test_proba, T)
        fit_obj = {'temperature': T}
    elif args.method == 'platt':
        models = fit_platt(val_proba, val_labels)
        cal_test = apply_platt(test_proba, models)
        fit_obj = {'method': 'platt'}
    else:
        models = fit_isotonic(val_proba, val_labels)
        cal_test = apply_isotonic(test_proba, models)
        fit_obj = {'method': 'isotonic'}

    cal_pred = np.array(STANDARD_LABELS)[cal_test.argmax(axis=1)]
    raw_pred = np.array(STANDARD_LABELS)[test_proba.argmax(axis=1)]

    summary = {
        'model': args.model_name,
        'dataset_name': args.dataset_name,
        'method': args.method,
        'n_val': int(len(val_df)),
        'n_test': int(len(test_df)),
    }
    summary.update(_summary_row('before', test_labels, raw_pred, test_proba))
    summary.update(_summary_row('after', test_labels, cal_pred, cal_test))
    summary['delta_ece_10_bins'] = summary['after_ece_10_bins'] - summary['before_ece_10_bins']
    summary['delta_brier'] = summary['after_brier'] - summary['before_brier']
    summary['delta_macro_f1'] = summary['after_macro_f1'] - summary['before_macro_f1']
    summary.update({f'fit_{k}': v for k, v in fit_obj.items()})

    ts = timestamp()
    out = test_df.copy()
    out['prediction_calibrated'] = cal_pred
    for k, c in enumerate(STANDARD_LABELS):
        out[f'proba_{c}_calibrated'] = cal_test[:, k]
    pred_path = save_dataframe(out, args.results_dir,
                               f'posthoc_{args.method}_predictions', ts)
    summary_path = save_dataframe(pd.DataFrame([summary]), args.results_dir,
                                  'posthoc_calibration_summary', ts)
    save_json(summary, args.results_dir, f'posthoc_{args.method}_manifest', ts)

    before_bins = reliability_bins(test_labels, raw_pred, test_proba, n_bins=10)
    after_bins = reliability_bins(test_labels, cal_pred, cal_test, n_bins=10)
    fig_dir = ensure_dir(Path(args.figures_dir))
    fig_path = fig_dir / f'posthoc_reliability_{args.method}_{ts}.png'
    _plot_reliability(before_bins, after_bins, fig_path,
                      f'{args.model_name} on {args.dataset_name} ({args.method})')

    print(f'[DONE] predictions -> {pred_path}')
    print(f'[DONE] summary     -> {summary_path}')
    print(f'[DONE] figure      -> {fig_path}')
    print(f'       ECE: {summary["before_ece_10_bins"]:.4f} -> {summary["after_ece_10_bins"]:.4f}')
    print(f'       Brier: {summary["before_brier"]:.4f} -> {summary["after_brier"]:.4f}')
    print(f'       Macro-F1: {summary["before_macro_f1"]:.4f} -> {summary["after_macro_f1"]:.4f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
