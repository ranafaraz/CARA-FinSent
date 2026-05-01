#!/usr/bin/env python3
"""Phase 4: Calibration & abstention report from a prediction CSV.

Required columns: label, prediction, proba_negative, proba_neutral, proba_positive.

Outputs
-------
results/<date>/calibration_reliability_bins_<ts>.csv
results/<date>/abstention_curve_<ts>.csv
results/<date>/calibration_summary_<ts>.csv
figures/reliability_diagram_<ts>.png
figures/abstention_curve_<ts>.png
figures/confidence_histogram_<ts>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from cara_finsent.data_utils import STANDARD_LABELS  # noqa: E402
from cara_finsent.io_utils import ensure_dir, save_dataframe, save_json, timestamp  # noqa: E402
from cara_finsent.metrics import (  # noqa: E402
    abstention_curve,
    classification_metrics,
    expected_calibration_error,
    multiclass_brier_score,
    reliability_bins,
)


THRESHOLDS = [round(x, 2) for x in np.arange(0.0, 1.0, 0.05)]


def _metrics_at_threshold(df: pd.DataFrame, t: float) -> dict:
    keep = df['confidence'] >= t
    coverage = float(keep.mean())
    if not keep.any():
        return {
            f'coverage_at_{int(t * 100):02d}': coverage,
            f'accuracy_at_{int(t * 100):02d}': float('nan'),
            f'macro_f1_at_{int(t * 100):02d}': float('nan'),
        }
    sub = df[keep]
    metrics = classification_metrics(sub['label'].values, sub['prediction'].values)
    return {
        f'coverage_at_{int(t * 100):02d}': coverage,
        f'accuracy_at_{int(t * 100):02d}': float(metrics['accuracy']),
        f'macro_f1_at_{int(t * 100):02d}': float(metrics['macro_f1']),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--model_name', required=True)
    ap.add_argument('--dataset_name', required=True)
    ap.add_argument('--results_dir', default='results')
    ap.add_argument('--figures_dir', default='figures')
    ap.add_argument('--n_bins', type=int, default=10)
    args = ap.parse_args()

    ts = timestamp()
    df = pd.read_csv(args.predictions)
    proba_cols = [f'proba_{c}' for c in STANDARD_LABELS]
    needed = {'label', 'prediction', *proba_cols}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f'predictions CSV missing columns: {sorted(missing)}')

    probs = df[proba_cols].to_numpy(dtype=float)
    df['confidence'] = probs.max(axis=1)
    y_true = df['label'].astype(str).values
    y_pred = df['prediction'].astype(str).values

    base_metrics = classification_metrics(y_true, y_pred)
    ece = expected_calibration_error(y_true, y_pred, probs, n_bins=args.n_bins)
    brier = multiclass_brier_score(y_true, probs)
    mean_conf = float(df['confidence'].mean())

    bins_df = reliability_bins(y_true, y_pred, probs, n_bins=args.n_bins)
    bins_df['model'] = args.model_name
    bins_df['dataset_name'] = args.dataset_name
    bins_path = save_dataframe(bins_df, args.results_dir, 'calibration_reliability_bins', ts)

    abst_df = abstention_curve(y_true, y_pred, probs, thresholds=THRESHOLDS)
    abst_df['model'] = args.model_name
    abst_df['dataset_name'] = args.dataset_name
    abst_path = save_dataframe(abst_df, args.results_dir, 'abstention_curve', ts)

    summary = {
        'model': args.model_name,
        'dataset_name': args.dataset_name,
        'predictions_file': str(args.predictions),
        'rows': int(len(df)),
        'accuracy': float(base_metrics['accuracy']),
        'macro_f1': float(base_metrics['macro_f1']),
        'weighted_f1': float(base_metrics['weighted_f1']),
        'mcc': float(base_metrics['mcc']),
        'ece_10_bins': float(ece),
        'brier_score': float(brier),
        'mean_confidence': mean_conf,
    }
    for t in (0.60, 0.70):
        summary.update(_metrics_at_threshold(df, t))
    summary_df = pd.DataFrame([summary])
    summary_path = save_dataframe(summary_df, args.results_dir, 'calibration_summary', ts)

    # Figures
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figures_dir = ensure_dir(Path(args.figures_dir))

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], '--', color='gray', label='perfect')
    populated = bins_df[bins_df['count'] > 0]
    if not populated.empty:
        ax.plot(populated['mean_confidence'], populated['accuracy'],
                marker='o', color='#3a78c2', label=args.model_name)
    ax.set_xlabel('mean confidence')
    ax.set_ylabel('accuracy')
    ax.set_title(f'Reliability diagram: {args.model_name} on {args.dataset_name}\nECE={ece:.4f}, Brier={brier:.4f}')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    rel_path = figures_dir / f'reliability_diagram_{ts}.png'
    fig.savefig(rel_path, dpi=150)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(abst_df['threshold'], abst_df['coverage'], marker='o', color='#3a78c2', label='coverage')
    ax1.set_xlabel('confidence threshold')
    ax1.set_ylabel('coverage')
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(abst_df['threshold'], abst_df['macro_f1_on_kept'], marker='s', color='#c2553a', label='macro-F1 on kept')
    ax2.set_ylabel('macro-F1 on kept')
    ax2.set_ylim(0, 1.05)
    ax1.set_title(f'Abstention curve: {args.model_name} on {args.dataset_name}')
    fig.tight_layout()
    abst_fig = figures_dir / f'abstention_curve_{ts}.png'
    fig.savefig(abst_fig, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(df['confidence'], bins=20, range=(0, 1), color='#3a78c2', edgecolor='black')
    ax.set_xlabel('predicted confidence')
    ax.set_ylabel('count')
    ax.set_title(f'Confidence histogram: {args.model_name}')
    fig.tight_layout()
    hist_fig = figures_dir / f'confidence_histogram_{ts}.png'
    fig.savefig(hist_fig, dpi=150)
    plt.close(fig)

    manifest = {
        'timestamp_utc': ts,
        'model_name': args.model_name,
        'dataset_name': args.dataset_name,
        'predictions_file': str(args.predictions),
        'reliability_csv': str(bins_path),
        'abstention_csv': str(abst_path),
        'summary_csv': str(summary_path),
        'reliability_png': str(rel_path),
        'abstention_png': str(abst_fig),
        'confidence_histogram_png': str(hist_fig),
        'ece_10_bins': float(ece),
        'brier_score': float(brier),
        'mean_confidence': mean_conf,
    }
    save_json(manifest, args.results_dir, 'calibration_manifest', ts)
    print(f'[DONE] Calibration summary -> {summary_path}')
    print(f'       ECE={ece:.4f} Brier={brier:.4f} mean_conf={mean_conf:.4f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
