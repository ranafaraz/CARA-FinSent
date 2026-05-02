#!/usr/bin/env python3
"""Phase 4: One-command research-readiness gate for CARA-FinSent.

Runs a sequence of binary checks and emits ``research_gate=PASS|FAIL``.

Checks (each row in the report):
  1. ``compileall`` succeeds for ``src`` and ``scripts``.
  2. Controlled gold splits exist for both PhraseBank and FiQA.
  3. Audit gate passes (``scripts/01_audit_datasets.py``).
  4. FinBERT label-mapping sanity artifact exists.
  5. Main benchmark has at least ``--min_seeds`` seed runs.
  6. No final result has ``text_hash_leakage_count > 0``.
  7. All final result rows include dataset metadata.
  8. Calibration/abstention report exists.
  9. Error analysis report exists.
 10. Final results are not pure smoke tests (epoch budget heuristic).

Outputs
-------
results/<date>/research_gate_report_<ts>.csv
results/<date>/research_gate_manifest_<ts>.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from cara_finsent.io_utils import git_commit_sha, save_dataframe, save_json, timestamp  # noqa: E402

PYTHON = sys.executable

GOLD_FILES = [
    'data/processed/gold/latest_gold_phrasebank_split.csv',
    'data/processed/gold/latest_gold_fiqa_split.csv',
]


def _check(label: str, ok: bool, detail: str) -> Dict[str, object]:
    return {'check': label, 'status': 'PASS' if ok else 'FAIL', 'detail': detail}


def check_compileall() -> Dict[str, object]:
    res = subprocess.run([PYTHON, '-m', 'compileall', '-q', 'src', 'scripts'],
                         cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    return _check('compileall', res.returncode == 0, (res.stdout + res.stderr).strip()[:500] or 'ok')


def check_gold_splits() -> Dict[str, object]:
    missing = [p for p in GOLD_FILES if not (PROJECT_ROOT / p).exists()]
    return _check('controlled_gold_splits', not missing,
                  'all present' if not missing else f'missing: {missing}')


def check_audit_gate() -> Dict[str, object]:
    cmd = [PYTHON, str(PROJECT_ROOT / 'scripts/01_audit_datasets.py'),
           '--inputs', *GOLD_FILES]
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    out = (res.stdout + res.stderr).strip()
    ok = res.returncode == 0 and 'audit_gate=PASS' in out
    tail = out.splitlines()[-3:] if out else ['no output']
    return _check('audit_gate', ok, ' | '.join(tail))


def check_label_sanity() -> Dict[str, object]:
    audit_dir = PROJECT_ROOT / 'data' / 'audit'
    if not audit_dir.exists():
        return _check('finbert_label_sanity', False, 'data/audit/ not found')
    files = list(audit_dir.rglob('finbert_label_mapping_*.json'))
    return _check('finbert_label_sanity', bool(files),
                  f'{len(files)} mapping artifacts' if files else 'no finbert_label_mapping_*.json found')


REQUIRED_EXPERIMENTS = ('classical', 'finbert_zero_shot', 'finbert_finetuned', 'agreement_weighted')


def check_seed_count(min_seeds: int) -> Tuple[Dict[str, object], pd.DataFrame]:
    results_dir = PROJECT_ROOT / 'results'
    files = [p for p in results_dir.rglob('seed_sweep_summary_*.csv') if '_archive' not in p.parts]
    if not files:
        return _check('min_seed_runs', False, 'no seed_sweep_summary_*.csv'), pd.DataFrame()
    frames = [pd.read_csv(p) for p in files]
    df = pd.concat(frames, ignore_index=True)
    counts = df.groupby(['dataset_name', 'experiment'])['seed'].nunique().reset_index(name='n_seeds')
    main = counts[counts['dataset_name'] == 'phrasebank']
    if main.empty:
        return _check('min_seed_runs', False, 'no phrasebank rows in seed sweeps'), df

    present = {r.experiment: int(r.n_seeds) for r in main.itertuples()}
    missing = [e for e in REQUIRED_EXPERIMENTS if e not in present]
    under = [f'{e}={present[e]}' for e in REQUIRED_EXPERIMENTS
             if e in present and present[e] < min_seeds]
    ok = not missing and not under
    parts = []
    for e in REQUIRED_EXPERIMENTS:
        parts.append(f'{e}={present.get(e, 0)}')
    detail = '; '.join(parts)
    if missing:
        detail += f' | missing: {missing}'
    if under:
        detail += f' | under_min_seeds({min_seeds}): {under}'
    return _check('min_seed_runs', ok, detail), df


def check_agreement_weighted_seeds(sweep_df: pd.DataFrame, min_seeds: int) -> Dict[str, object]:
    """Phase 9: agreement_weighted MUST have at least min_seeds unique seeds."""
    if sweep_df.empty:
        return _check('agreement_weighted_seed_runs', False, 'no sweep rows to inspect')
    aw = sweep_df[(sweep_df['experiment'] == 'agreement_weighted')
                  & (sweep_df['dataset_name'] == 'phrasebank')]
    n = int(aw['seed'].nunique()) if not aw.empty else 0
    seeds = sorted({int(s) for s in aw['seed'].unique()}) if not aw.empty else []
    ok = n >= min_seeds
    return _check('agreement_weighted_seed_runs', ok,
                  f'unique_seeds={n} (required {min_seeds}); seeds={seeds}')


def check_no_leakage(sweep_df: pd.DataFrame) -> Dict[str, object]:
    if sweep_df.empty:
        return _check('no_text_hash_leakage', False, 'no sweep rows to inspect')
    bad = sweep_df[pd.to_numeric(sweep_df.get('text_hash_leakage_count', 0), errors='coerce').fillna(0) > 0]
    return _check('no_text_hash_leakage', bad.empty,
                  'all zero' if bad.empty else f'{len(bad)} rows have leakage > 0')


def check_metadata(sweep_df: pd.DataFrame) -> Dict[str, object]:
    required = {'dataset_name', 'benchmark_mode', 'git_commit_sha', 'seed', 'macro_f1'}
    if sweep_df.empty:
        return _check('result_metadata', False, 'no sweep rows to inspect')
    missing = required - set(sweep_df.columns)
    if missing:
        return _check('result_metadata', False, f'missing columns: {sorted(missing)}')
    null_counts = {col: int(sweep_df[col].isna().sum()) for col in required}
    bad = {k: v for k, v in null_counts.items() if v > 0}
    return _check('result_metadata', not bad,
                  'all present' if not bad else f'null counts: {bad}')


def check_calibration_report() -> Dict[str, object]:
    files = list((PROJECT_ROOT / 'results').rglob('calibration_summary_*.csv'))
    files = [p for p in files if '_archive' not in p.parts]
    return _check('calibration_report', bool(files),
                  f'{len(files)} calibration_summary files' if files else 'none found')


def check_error_analysis() -> Dict[str, object]:
    files = list((PROJECT_ROOT / 'results').rglob('error_analysis_summary_*.csv'))
    files = [p for p in files if '_archive' not in p.parts]
    return _check('error_analysis_report', bool(files),
                  f'{len(files)} error_analysis_summary files' if files else 'none found')


def check_not_smoke(sweep_df: pd.DataFrame) -> Dict[str, object]:
    if sweep_df.empty:
        return _check('not_smoke_tests', False, 'no sweep rows to inspect')
    epoch_cols = [c for c in ('epochs', 'num_epochs') if c in sweep_df.columns]
    finbert_rows = sweep_df[sweep_df['experiment'].isin(['finbert_finetuned', 'agreement_weighted'])]
    if finbert_rows.empty:
        return _check('not_smoke_tests', False, 'no finbert/agreement rows in sweeps')
    if not epoch_cols:
        return _check('not_smoke_tests', True, 'no epoch column to test (assumed not smoke)')
    ok = False
    detail_parts = []
    for col in epoch_cols:
        max_epoch = pd.to_numeric(finbert_rows[col], errors='coerce').max()
        detail_parts.append(f'max {col}={max_epoch}')
        if pd.notna(max_epoch) and float(max_epoch) >= 1:
            ok = True
    return _check('not_smoke_tests', ok, '; '.join(detail_parts))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='results')
    ap.add_argument('--min_seeds', type=int, default=5)
    args = ap.parse_args()

    ts = timestamp()
    rows: List[Dict[str, object]] = []

    rows.append(check_compileall())
    rows.append(check_gold_splits())
    rows.append(check_audit_gate())
    rows.append(check_label_sanity())
    seed_check, sweep_df = check_seed_count(args.min_seeds)
    rows.append(seed_check)
    rows.append(check_agreement_weighted_seeds(sweep_df, args.min_seeds))
    rows.append(check_no_leakage(sweep_df))
    rows.append(check_metadata(sweep_df))
    rows.append(check_calibration_report())
    rows.append(check_error_analysis())
    rows.append(check_not_smoke(sweep_df))

    report_df = pd.DataFrame(rows)
    report_df['timestamp_utc'] = ts
    report_path = save_dataframe(report_df, args.results_dir, 'research_gate_report', ts)

    overall_pass = bool((report_df['status'] == 'PASS').all())
    manifest = {
        'timestamp_utc': ts,
        'min_seeds_required': args.min_seeds,
        'checks': rows,
        'overall_pass': overall_pass,
        'failed': [r['check'] for r in rows if r['status'] != 'PASS'],
        'git_commit_sha': git_commit_sha(PROJECT_ROOT),
        'report_csv': str(report_path),
    }
    save_json(manifest, args.results_dir, 'research_gate_manifest', ts)

    print('\n--- Research gate checks ---')
    for r in rows:
        print(f"  [{r['status']}] {r['check']}: {r['detail']}")
    print(f'\n[REPORT] {report_path}')
    final = 'research_gate=PASS' if overall_pass else 'research_gate=FAIL'
    print(final)
    return 0 if overall_pass else 2


if __name__ == '__main__':
    raise SystemExit(main())
