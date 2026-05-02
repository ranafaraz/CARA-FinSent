#!/usr/bin/env python3
"""Phase 5: research-grade multi-seed sweep orchestrator.

Thin wrapper around scripts/19_run_seed_sweep.py that codifies the Phase 5
seed list (13, 21, 42, 87, 101) and a smoke / full mode switch.

  smoke -> classical + finbert_zero_shot only (CPU-friendly, fast).
  full  -> classical + finbert_zero_shot + finbert_finetuned + agreement_weighted.

Usage:
  python scripts/26_run_seed_sweep.py --seeds 13 21 42 87 101 --mode smoke
  python scripts/26_run_seed_sweep.py --seeds 13 21 42 87 101 --mode full
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

DEFAULT_SEEDS = [13, 21, 42, 87, 101]

MODE_MODELS = {
    'smoke': 'classical,finbert_zero_shot',
    'full':  'classical,finbert_zero_shot,finbert_finetuned,agreement_weighted',
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    ap.add_argument('--mode', choices=sorted(MODE_MODELS), default='smoke')
    ap.add_argument('--datasets', default='phrasebank',
                    help='Comma-separated dataset names (phrasebank,fiqa).')
    ap.add_argument('--results_dir', default='results')
    ap.add_argument('--figures_dir', default='figures')
    ap.add_argument('--models_dir', default='models')
    ap.add_argument('--finbert_epochs', type=int, default=3)
    ap.add_argument('--finbert_batch_size', type=int, default=16)
    ap.add_argument('--weight_schedule', default='linear')
    ap.add_argument('--continue_on_error', action='store_true')
    args = ap.parse_args()

    models_csv = MODE_MODELS[args.mode]
    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    if not datasets:
        raise SystemExit('--datasets must list at least one dataset name')

    print(f'[PHASE5] mode={args.mode} models={models_csv}')
    print(f'[PHASE5] datasets={datasets} seeds={args.seeds}')

    inner = PROJECT_ROOT / 'scripts' / '19_run_seed_sweep.py'
    failures = []
    for ds in datasets:
        cmd = [
            PYTHON, str(inner),
            '--dataset_name', ds,
            '--models', models_csv,
            '--seeds', *map(str, args.seeds),
            '--results_dir', args.results_dir,
            '--figures_dir', args.figures_dir,
            '--models_dir', args.models_dir,
            '--finbert_epochs', str(args.finbert_epochs),
            '--finbert_batch_size', str(args.finbert_batch_size),
            '--weight_schedule', args.weight_schedule,
        ]
        if args.continue_on_error:
            cmd.append('--continue_on_error')
        print(f'[PHASE5] -> {ds}')
        rc = subprocess.call(cmd, cwd=str(PROJECT_ROOT))
        if rc != 0:
            print(f'[PHASE5][FAIL] dataset={ds} rc={rc}')
            failures.append(ds)
            if not args.continue_on_error:
                return rc
    if failures:
        print(f'[PHASE5][DONE] with failures: {failures}')
        return 1
    print('[PHASE5][DONE] all datasets ok')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
