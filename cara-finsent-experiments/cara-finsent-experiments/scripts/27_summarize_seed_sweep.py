#!/usr/bin/env python3
"""Phase 5: aggregate seed sweep results into the final leaderboard.

Reads all ``seed_sweep_summary_*.csv`` files (recursively under --results_dir)
and writes a single mean/std leaderboard with the Phase 5 schema:

    results/<date>/final_leaderboard_mean_std_<ts>.csv

Group keys: dataset_name, benchmark_mode, experiment, model
Metrics:    accuracy, macro_f1, ece_10_bins, brier_score, latency_or_seconds (n/a)

Use --min_seeds to flag rows as research_grade.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from cara_finsent.io_utils import save_dataframe, save_json, timestamp  # noqa: E402

GROUP_COLS = ['dataset_name', 'benchmark_mode', 'experiment', 'model']
METRIC_COLS = ['accuracy', 'macro_f1', 'ece_10_bins', 'brier_score']


def load_sweeps(root: Path) -> pd.DataFrame:
    files = [p for p in root.rglob('seed_sweep_summary_*.csv') if '_archive' not in p.parts]
    if not files:
        raise SystemExit(f'no seed_sweep_summary_*.csv under {root}')
    frames = []
    for p in files:
        try:
            frames.append(pd.read_csv(p))
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as exc:
            print(f'[WARN] could not read {p}: {exc}')
    df = pd.concat(frames, ignore_index=True)
    print(f'[INFO] loaded {len(df)} rows from {len(files)} sweep files')
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='results')
    ap.add_argument('--min_seeds', type=int, default=5)
    args = ap.parse_args()

    df = load_sweeps(Path(args.results_dir))
    for col in GROUP_COLS + METRIC_COLS + ['seed']:
        if col not in df.columns:
            df[col] = np.nan

    rows = []
    for keys, sub in df.groupby(GROUP_COLS, dropna=False):
        n_seeds = sub['seed'].nunique()
        rec = dict(zip(GROUP_COLS, keys))
        rec['n_seeds'] = int(n_seeds)
        for m in METRIC_COLS:
            vals = pd.to_numeric(sub[m], errors='coerce').dropna()
            rec[f'mean_{m}'] = float(vals.mean()) if len(vals) else np.nan
            rec[f'std_{m}'] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rec['mean_latency_or_seconds'] = np.nan
        rec['research_grade'] = bool(n_seeds >= args.min_seeds)
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values(
        ['dataset_name', 'benchmark_mode', 'mean_macro_f1'], ascending=[True, True, False]
    )

    ts = timestamp()
    out_path = save_dataframe(out, args.results_dir, 'final_leaderboard_mean_std', ts)
    print(f'[DONE] final leaderboard -> {out_path}')

    manifest = {
        'timestamp': ts,
        'min_seeds': args.min_seeds,
        'n_sweep_rows': int(len(df)),
        'n_groups': int(len(out)),
        'n_research_grade_groups': int(out['research_grade'].sum()),
    }
    save_json(manifest, args.results_dir, 'final_leaderboard_manifest', ts)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
