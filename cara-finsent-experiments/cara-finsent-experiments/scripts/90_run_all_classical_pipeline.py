#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print('\n[PIPELINE] ' + ' '.join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description='Run the classical/reliability pipeline end to end.')
    parser.add_argument('--data', required=True, help='Standardized input CSV.')
    parser.add_argument('--external_corpus_csv', default=None)
    parser.add_argument('--max_rows', type=int, default=None)
    args = parser.parse_args()

    py = sys.executable
    max_rows_args = ['--max_rows', str(args.max_rows)] if args.max_rows else []
    run([py, 'scripts/10_run_classical_baselines.py', '--data', args.data] + max_rows_args)
    run([py, 'scripts/12_run_structured_features_experiment.py', '--data', args.data] + max_rows_args)
    retr = [py, 'scripts/13_run_retrieval_experiment.py', '--data', args.data] + max_rows_args
    if args.external_corpus_csv:
        retr += ['--external_corpus_csv', args.external_corpus_csv]
    run(retr)
    run([py, 'scripts/15_run_calibration_experiment.py', '--data', args.data] + max_rows_args)
    cara = [py, 'scripts/16_run_full_cara_lite_experiment.py', '--data', args.data] + max_rows_args
    if args.external_corpus_csv:
        cara += ['--external_corpus_csv', args.external_corpus_csv]
    run(cara)

    print('\n[DONE] Pipeline finished. Zip or share the results/ folder for analysis.')


if __name__ == '__main__':
    main()
