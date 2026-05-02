#!/usr/bin/env python3
"""Phase 5: master orchestrator that runs the full final-research package.

Steps (in order):
  1. compileall src + scripts
  2. dataset audit
  3. FinBERT label mapping check
  4. FiQA semantics audit (best-effort; skipped if no internet)
  5. Phase 5 seed sweep (mode=smoke or full)
  6. Final leaderboard
  7. Calibration / abstention report on the chosen predictions
  8. Error analysis on the chosen predictions
  9. Research gate

Stops at the first failure unless --continue_on_error is passed and writes a
``phase5_run_manifest_<ts>.json`` recording the per-step status.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from cara_finsent.io_utils import save_json, timestamp  # noqa: E402

PYTHON = sys.executable

GOLD_PHRASEBANK = 'data/processed/gold/latest_gold_phrasebank_split.csv'
GOLD_FIQA = 'data/processed/gold/latest_gold_fiqa_split.csv'


def _run(name: str, cmd: List[str], log: list, continue_on_error: bool) -> bool:
    print(f'\n[STEP] {name}')
    print('       cmd:', ' '.join(cmd))
    rc = subprocess.call(cmd, cwd=str(PROJECT_ROOT))
    ok = rc == 0
    log.append({'step': name, 'cmd': cmd, 'rc': rc, 'ok': ok})
    if not ok:
        print(f'[STEP][FAIL] {name} rc={rc}')
        if not continue_on_error:
            return False
    else:
        print(f'[STEP][OK] {name}')
    return True


def _latest_match(root: Path, pattern: str) -> Optional[Path]:
    cands = [p for p in root.rglob(pattern) if '_archive' not in p.parts]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    ap.add_argument('--seeds', type=int, nargs='+', default=[13, 21, 42, 87, 101])
    ap.add_argument('--datasets', default='phrasebank')
    ap.add_argument('--predictions', default=None,
                    help='Optional predictions CSV for calibration + error analysis. '
                         'If omitted, auto-picks newest finbert_baseline_predictions_*.csv.')
    ap.add_argument('--predictions_model_name', default='finbert_zero_shot')
    ap.add_argument('--predictions_dataset_name', default='phrasebank')
    ap.add_argument('--min_seeds', type=int, default=5)
    ap.add_argument('--continue_on_error', action='store_true')
    ap.add_argument('--skip_fiqa_audit', action='store_true',
                    help='Skip the FiQA semantics audit (e.g. when offline).')
    args = ap.parse_args()

    ts = timestamp()
    log: List[dict] = []

    # 1. compileall
    if not _run('compileall', [PYTHON, '-m', 'compileall', '-q', 'src', 'scripts'],
                log, args.continue_on_error):
        return _finalize(ts, log, ok=False)

    # 2. dataset audit
    if not _run('audit_datasets',
                [PYTHON, 'scripts/01_audit_datasets.py',
                 '--inputs', GOLD_PHRASEBANK, GOLD_FIQA],
                log, args.continue_on_error):
        return _finalize(ts, log, ok=False)

    # 3. finbert label mapping check
    if not _run('finbert_label_mapping',
                [PYTHON, 'scripts/11d_check_finbert_label_mapping.py',
                 '--sample_file', GOLD_PHRASEBANK],
                log, args.continue_on_error):
        return _finalize(ts, log, ok=False)

    # 4. fiqa audit (best-effort)
    if not args.skip_fiqa_audit:
        _run('fiqa_semantics_audit',
             [PYTHON, 'scripts/23_fiqa_semantics_audit.py'],
             log, continue_on_error=True)

    # 5. seed sweep
    if not _run('seed_sweep',
                [PYTHON, 'scripts/26_run_seed_sweep.py',
                 '--seeds', *map(str, args.seeds),
                 '--mode', args.mode,
                 '--datasets', args.datasets,
                 *(['--continue_on_error'] if args.continue_on_error else [])],
                log, args.continue_on_error):
        return _finalize(ts, log, ok=False)

    # 6. final leaderboard
    if not _run('final_leaderboard',
                [PYTHON, 'scripts/27_summarize_seed_sweep.py',
                 '--min_seeds', str(args.min_seeds)],
                log, args.continue_on_error):
        return _finalize(ts, log, ok=False)

    # Pick predictions for calibration + error analysis
    pred = args.predictions
    if pred is None:
        latest = _latest_match(PROJECT_ROOT / 'results', 'finbert_baseline_predictions_*.csv')
        if latest is None:
            print('[WARN] no predictions CSV found; skipping calibration + error analysis')
            return _finalize(ts, log, ok=True)
        pred = str(latest.relative_to(PROJECT_ROOT))
        print(f'[AUTO] predictions = {pred}')

    # 7. calibration / abstention
    _run('calibration_report',
         [PYTHON, 'scripts/22_calibration_abstention_report.py',
          '--predictions', pred,
          '--model_name', args.predictions_model_name,
          '--dataset_name', args.predictions_dataset_name],
         log, continue_on_error=True)

    # 8. error analysis (Phase 5 schema)
    _run('error_analysis',
         [PYTHON, 'scripts/28_error_analysis.py',
          '--predictions', pred,
          '--model_name', args.predictions_model_name,
          '--dataset_name', args.predictions_dataset_name],
         log, continue_on_error=True)

    # 9. research gate
    rc_gate = subprocess.call(
        [PYTHON, 'scripts/25_research_gate.py', '--min_seeds', str(args.min_seeds)],
        cwd=str(PROJECT_ROOT),
    )
    log.append({'step': 'research_gate', 'rc': rc_gate, 'ok': rc_gate == 0})
    return _finalize(ts, log, ok=rc_gate == 0)


def _finalize(ts: str, log: list, ok: bool) -> int:
    save_json({'timestamp': ts, 'ok': ok, 'steps': log},
              'results', 'phase5_run_manifest', ts)
    print(f'\n[PHASE5] manifest written. ok={ok}')
    return 0 if ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
