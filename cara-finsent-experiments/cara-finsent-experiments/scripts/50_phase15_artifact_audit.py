#!/usr/bin/env python3
"""Phase 15 Task 2 — verify Phase 14 artefacts and write an audit CSV."""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DOCS = [
    'docs/PHASE14_CORRECTED_VALIDATION_REPORT.md',
    'docs/PHASE14_PAPER_UPDATE_RECOMMENDATION.md',
    'docs/PHASE14_AW_CHECKPOINT_RECOVERY_REPORT.md',
    'docs/PHASE14_PHASE13_AUDIT.md',
]
REQUIRED_SCRIPTS = [
    'scripts/41_generate_val_test_predictions.py',
    'scripts/42_clean_calibration_eval.py',
    'scripts/43_clean_ensemble_eval.py',
    'scripts/44_clean_neutral_mitigation_eval.py',
    'scripts/45_fix_fiqa_polarity_and_external_eval.py',
]
REQUIRED_DIRS = [
    'results/2026-05-02/phase14_predictions',
    'results/2026-05-02/phase14_calibration',
    'results/2026-05-02/phase14_ensemble',
    'results/2026-05-02/phase14_external',
    'results/2026-05-02/phase14_neutral',
    'artifacts/phase14_corrected_validation',
]


def main():
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    rows = []
    missing = 0
    for kind, paths in (('doc', REQUIRED_DOCS), ('script', REQUIRED_SCRIPTS), ('dir', REQUIRED_DIRS)):
        for rel in paths:
            p = ROOT / rel
            exists = p.exists()
            if not exists:
                missing += 1
            n_files = 0
            total_bytes = 0
            if exists and p.is_dir():
                for f in p.rglob('*'):
                    if f.is_file():
                        n_files += 1
                        total_bytes += f.stat().st_size
            elif exists:
                n_files = 1
                total_bytes = p.stat().st_size
            rows.append({
                'kind': kind,
                'path': rel,
                'exists': exists,
                'n_files': n_files,
                'total_bytes': total_bytes,
            })

    out_dir = ROOT / 'results' / '2026-05-02'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'phase15_artifact_audit_{ts}.csv'
    with out_path.open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['kind', 'path', 'exists', 'n_files', 'total_bytes'])
        w.writeheader()
        w.writerows(rows)
    print(f'[OUT] {out_path}')
    print(f'[INFO] missing={missing} of {len(rows)}')
    return 0 if missing == 0 else 2


if __name__ == '__main__':
    sys.exit(main())
