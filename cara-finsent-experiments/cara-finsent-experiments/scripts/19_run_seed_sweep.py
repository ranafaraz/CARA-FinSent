#!/usr/bin/env python3
"""Phase 4: Run selected experiments across multiple seeds and aggregate results.

Wraps existing scripts as subprocesses, never resplits the gold split, and
emits a unified seed-level summary CSV plus a JSON manifest.

Usage:
  python scripts/19_run_seed_sweep.py \
      --dataset_name phrasebank \
      --data data/processed/gold/latest_gold_phrasebank_split.csv \
      --models classical,finbert_zero_shot,finbert_finetuned,agreement_weighted \
      --seeds 42 43 44 45 46
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from cara_finsent.data_utils import auto_detect_gold_split, load_gold_split  # noqa: E402
from cara_finsent.io_utils import git_commit_sha, save_dataframe, save_json, timestamp  # noqa: E402

PYTHON = sys.executable

EXPERIMENTS: Dict[str, Dict[str, str]] = {
    'classical': {
        'script': 'scripts/10_run_classical_baselines.py',
        'summary_prefix': 'classical_baseline_summary',
    },
    'finbert_zero_shot': {
        'script': 'scripts/11c_eval_finbert_zero_shot.py',
        'summary_prefix': 'finbert_baseline_summary',
    },
    'finbert_finetuned': {
        'script': 'scripts/11_run_finbert_baseline.py',
        'summary_prefix': 'finbert_baseline_summary',
    },
    'agreement_weighted': {
        'script': 'scripts/11f_train_finbert_agreement_weighted.py',
        'summary_prefix': 'finbert_agreement_weighted_summary',
    },
}

REQUIRED_COLUMNS = [
    'model', 'experiment', 'dataset_name', 'benchmark_mode', 'seed',
    'accuracy', 'macro_f1', 'weighted_f1', 'mcc',
    'ece_10_bins', 'brier_score', 'mean_confidence',
    'train_rows', 'val_rows', 'test_rows',
    'text_hash_leakage_count', 'git_commit_sha', 'summary_file',
]


def parse_iso_ts_from_filename(name: str) -> Optional[datetime]:
    """Extract YYYYMMDD_HHMMSS from filename and return UTC datetime."""
    stem = Path(name).stem
    parts = stem.split('_')
    for i in range(len(parts) - 1):
        date_part = parts[i]
        time_part = parts[i + 1]
        if len(date_part) == 8 and date_part.isdigit() and len(time_part) == 6 and time_part.isdigit():
            try:
                return datetime.strptime(date_part + time_part, '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


def find_summary_after(results_dir: Path, prefix: str, ts_before: datetime) -> Path:
    """Return the newest *_summary_*.csv with given prefix produced after ts_before."""
    candidates: List[Path] = []
    for path in results_dir.rglob(f'{prefix}_*.csv'):
        if '_archive' in path.parts:
            continue
        ts_in_name = parse_iso_ts_from_filename(path.name)
        if ts_in_name is not None and ts_in_name >= ts_before:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f'No summary file matching {prefix}_*.csv found under {results_dir} after {ts_before.isoformat()}'
        )
    return sorted(candidates, key=lambda p: parse_iso_ts_from_filename(p.name) or datetime.min.replace(tzinfo=timezone.utc))[-1]


def build_command(experiment: str, dataset_name: str, data_path: str, seed: int,
                  results_dir: str, figures_dir: str, models_dir: str,
                  finbert_epochs: float, finbert_batch_size: int,
                  weight_schedule: str) -> List[str]:
    spec = EXPERIMENTS[experiment]
    cmd = [PYTHON, str(PROJECT_ROOT / spec['script']),
           '--data', data_path,
           '--dataset_name', dataset_name,
           '--seed', str(seed),
           '--results_dir', results_dir,
           '--figures_dir', figures_dir]
    if experiment == 'finbert_finetuned':
        cmd += ['--epochs', str(finbert_epochs),
                '--batch_size', str(finbert_batch_size),
                '--models_dir', models_dir]
    elif experiment == 'agreement_weighted':
        cmd += ['--num_epochs', str(int(round(finbert_epochs))),
                '--batch_size', str(finbert_batch_size),
                '--weight_schedule', weight_schedule,
                '--models_dir', models_dir]
    return cmd


def normalize_row(row: Dict, experiment: str, summary_path: Path, dataset_name: str, seed: int) -> Dict:
    out: Dict[str, object] = {col: None for col in REQUIRED_COLUMNS}
    out['model'] = str(row.get('model', experiment))
    out['experiment'] = experiment
    out['dataset_name'] = str(row.get('dataset_name', dataset_name))
    out['benchmark_mode'] = str(row.get('benchmark_mode', f'{dataset_name}_in_domain'))
    out['seed'] = int(row.get('seed', seed))
    for key in ('accuracy', 'macro_f1', 'weighted_f1', 'mcc',
                'ece_10_bins', 'brier_score', 'mean_confidence',
                'train_rows', 'val_rows', 'test_rows', 'text_hash_leakage_count'):
        if key in row and pd.notna(row[key]):
            out[key] = row[key]
    out['git_commit_sha'] = str(row.get('git_commit_sha', '') or '')
    out['summary_file'] = str(summary_path)
    # carry through useful extras when present
    for extra in ('weight_schedule', 'num_epochs', 'epochs'):
        if extra in row and pd.notna(row[extra]):
            out[extra] = row[extra]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_name', required=True, choices=['phrasebank', 'fiqa'])
    ap.add_argument('--data', default=None)
    ap.add_argument('--models', default='classical,finbert_zero_shot,finbert_finetuned,agreement_weighted',
                    help='Comma-separated experiment keys.')
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument('--results_dir', default='results')
    ap.add_argument('--figures_dir', default='figures')
    ap.add_argument('--models_dir', default='models')
    ap.add_argument('--finbert_epochs', type=float, default=3)
    ap.add_argument('--finbert_batch_size', type=int, default=16)
    ap.add_argument('--weight_schedule', default='linear',
                    choices=['all_equal', 'linear', 'strong', 'high_only'])
    ap.add_argument('--continue_on_error', action='store_true')
    args = ap.parse_args()

    if not args.data:
        args.data = str(auto_detect_gold_split(args.dataset_name))
        print(f'[AUTO] data = {args.data}')

    # Validate gold split once
    train_df, val_df, test_df = load_gold_split(args.data)
    print(f'[INFO] Gold split rows: train={len(train_df)} val={len(val_df)} test={len(test_df)}')

    experiments = [m.strip() for m in args.models.split(',') if m.strip()]
    unknown = [m for m in experiments if m not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f'Unknown experiments: {unknown}. Allowed: {list(EXPERIMENTS)}')

    results_dir = Path(args.results_dir)
    sweep_ts = timestamp()

    rows: List[Dict] = []
    failures: List[Dict] = []

    for seed in args.seeds:
        for experiment in experiments:
            spec = EXPERIMENTS[experiment]
            cmd = build_command(experiment, args.dataset_name, args.data, seed,
                                args.results_dir, args.figures_dir, args.models_dir,
                                args.finbert_epochs, args.finbert_batch_size,
                                args.weight_schedule)
            print(f'\n[SWEEP] seed={seed} experiment={experiment}')
            print('        cmd:', ' '.join(cmd))
            ts_before = datetime.now(timezone.utc).replace(microsecond=0)
            t0 = time.perf_counter()
            try:
                subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
                elapsed = time.perf_counter() - t0
                summary_path = find_summary_after(results_dir, spec['summary_prefix'], ts_before)
                df = pd.read_csv(summary_path)
                for _, raw in df.iterrows():
                    row = normalize_row(raw.to_dict(), experiment, summary_path, args.dataset_name, seed)
                    row['seconds'] = elapsed
                    rows.append(row)
                print(f'        OK in {elapsed:.1f}s -> {summary_path}')
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                msg = str(exc)
                print(f'        FAIL: {msg}')
                failures.append({'seed': seed, 'experiment': experiment, 'error': msg})
                if not args.continue_on_error:
                    raise

    if not rows:
        raise SystemExit('Seed sweep produced no rows.')

    sweep_df = pd.DataFrame(rows)
    sweep_df = sweep_df[[c for c in REQUIRED_COLUMNS if c in sweep_df.columns]
                        + [c for c in sweep_df.columns if c not in REQUIRED_COLUMNS]]
    summary_path = save_dataframe(sweep_df, args.results_dir, 'seed_sweep_summary', sweep_ts)

    manifest = {
        'timestamp_utc': sweep_ts,
        'dataset_name': args.dataset_name,
        'data_file': str(args.data),
        'experiments': experiments,
        'seeds': list(args.seeds),
        'finbert_epochs': args.finbert_epochs,
        'finbert_batch_size': args.finbert_batch_size,
        'weight_schedule': args.weight_schedule,
        'rows': len(sweep_df),
        'failures': failures,
        'summary_file': str(summary_path),
        'git_commit_sha': git_commit_sha(PROJECT_ROOT),
    }
    save_json(manifest, args.results_dir, 'seed_sweep_manifest', sweep_ts)
    print(f'\n[DONE] Seed sweep summary -> {summary_path}')
    print(f'       Rows: {len(sweep_df)}  Failures: {len(failures)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
