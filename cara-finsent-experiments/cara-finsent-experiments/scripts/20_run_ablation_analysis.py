#!/usr/bin/env python3
"""
20_run_ablation_analysis.py
---
Ablation study: run FinBERT with different configurations to quantify
improvement from each component (agreement weighting, uncertainty handling).

Configurations tested:
1. Baseline (no weighting)
2. + Agreement weighting
3. + Uncertainty down-weighting
4. + Both

Usage:
    python scripts/20_run_ablation_analysis.py --seed 42 --epochs 2 --batch_size 8
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


CONFIGS = [
    {'name': 'Baseline', 'args': []},
    {'name': 'With Agreement Weight', 'args': ['--use_agreement_weight']},
    {'name': 'With Uncertainty Down-weight', 'args': ['--down_weight_uncertain']},
    {'name': 'With Both', 'args': ['--use_agreement_weight', '--down_weight_uncertain']},
]


def run_experiment(config: dict, common_args: list[str]) -> dict:
    """Run a single FinBERT experiment and parse results."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / 'scripts' / '11_run_finbert_baseline_improved.py'),
        *common_args,
        *config['args'],
    ]
    
    print(f"\n{'='*70}")
    print(f"Running: {config['name']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    
    if result.returncode != 0:
        print(f"[ERROR] {config['name']} failed with exit code {result.returncode}")
        return None
    
    return {'name': config['name'], 'args': config['args'], 'success': True}


def main():
    parser = argparse.ArgumentParser(description='Run ablation study on FinBERT improvements.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=float, default=2, help='Fewer epochs for faster ablation')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--data', default=None, help='Dataset path')
    parser.add_argument('--max_rows', type=int, default=None)
    args = parser.parse_args()
    
    common_args = [
        '--seed', str(args.seed),
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
    ]
    if args.data:
        common_args.extend(['--data', args.data])
    if args.max_rows:
        common_args.extend(['--max_rows', str(args.max_rows)])
    
    results = []
    for config in CONFIGS:
        result = run_experiment(config, common_args)
        if result:
            results.append(result)
    
    print(f"\n{'='*70}")
    print("ABLATION SUMMARY")
    print(f"{'='*70}")
    for r in results:
        status = "[OK]" if r.get('success') else "[FAIL]"
        print(f"{status} {r['name']}")
    print("\nTo compare results, check results/ folder for *_summary_*.csv files")
    print("Use scripts/17_compare_all_experiments.py to generate comparison table")


if __name__ == '__main__':
    main()
