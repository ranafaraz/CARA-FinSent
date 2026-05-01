#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from cara_finsent.io_utils import save_dataframe, timestamp


def run(cmd, log_dir: Path | None, keep_going: bool) -> int:
    label = Path(cmd[1]).stem if len(cmd) > 1 else 'step'
    print('\n[PIPELINE] ' + ' '.join(cmd))
    log_path = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        log_path = log_dir / f'{label}_{ts}.log'
    try:
        if log_path is not None:
            with log_path.open('w', encoding='utf-8') as fh:
                proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
        else:
            proc = subprocess.run(cmd, check=False)
    except Exception as exc:
        print(f'[PIPELINE][ERROR] {label}: {exc}')
        if not keep_going:
            raise
        return 1
    if proc.returncode != 0:
        print(f'[PIPELINE][FAIL] {label} exited {proc.returncode}. Log: {log_path}')
        if not keep_going:
            raise SystemExit(proc.returncode)
    else:
        print(f'[PIPELINE][OK] {label}' + (f' -> {log_path}' if log_path else ''))
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description='Run the classical/reliability pipeline end to end.')
    parser.add_argument('--data', default=None, help='Standardized input CSV. Auto-detected from data/processed/latest.csv if omitted.')
    parser.add_argument('--external_corpus_csv', default=None)
    parser.add_argument('--max_rows', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--with_finbert', action='store_true', help='Also run FinBERT (script 11). Slow without GPU.')
    parser.add_argument('--with_agreement', action='store_true', help='Also run agreement-aware experiment (script 14). Requires agreement column in data.')
    parser.add_argument('--keep_going', action='store_true', help='Continue running subsequent steps even if one fails.')
    parser.add_argument('--log_dir', default='results/logs', help='Directory for per-step stdout/stderr logs. Empty string disables logging.')
    args = parser.parse_args()

    from cara_finsent.io_utils import date_subdir
    if not args.data:
        from cara_finsent.data_utils import auto_detect_data
        args.data = str(auto_detect_data())
        print(f'[AUTO] data = {args.data}')
    log_dir = (Path(args.log_dir) / date_subdir()) if args.log_dir else None
    py = sys.executable
    common = ['--data', args.data, '--seed', str(args.seed)]
    if args.max_rows:
        common += ['--max_rows', str(args.max_rows)]

    steps: list[list[str]] = [
        [py, 'scripts/10_run_classical_baselines.py', *common],
        [py, 'scripts/12_run_structured_features_experiment.py', *common],
        [py, 'scripts/13_run_retrieval_experiment.py', *common] + (['--external_corpus_csv', args.external_corpus_csv] if args.external_corpus_csv else []),
        [py, 'scripts/15_run_calibration_experiment.py', *common],
        [py, 'scripts/16_run_full_cara_lite_experiment.py', *common] + (['--external_corpus_csv', args.external_corpus_csv] if args.external_corpus_csv else []),
    ]
    if args.with_agreement:
        steps.append([py, 'scripts/14_run_agreement_aware_experiment.py', *common])
    if args.with_finbert:
        steps.append([py, 'scripts/11_run_finbert_baseline.py', *common])

    failures = 0
    step_report_rows = []
    ts = timestamp()
    for cmd in steps:
        step_name = Path(cmd[1]).stem if len(cmd) > 1 else 'step'
        rc = run(cmd, log_dir, args.keep_going)
        status = 'ok' if rc == 0 else 'skipped_failed'
        reason = '' if rc == 0 else f'Command exited with code {rc}'
        step_report_rows.append({
            'step': step_name,
            'status': status,
            'reason': reason,
            'command': ' '.join(cmd),
        })
        if rc != 0:
            failures += 1

    report_df = pd.DataFrame(step_report_rows)
    report_path = save_dataframe(report_df, 'results', 'pipeline_step_report', ts)

    print(f'\n[DONE] Pipeline finished. {len(steps) - failures}/{len(steps)} steps succeeded.')
    print(f'[DONE] Pipeline report -> {report_path}')
    if failures and not args.keep_going:
        raise SystemExit(1)
    print('Zip or share the results/ folder for analysis.')


if __name__ == '__main__':
    main()
