#!/usr/bin/env python3
"""Phase 15 Task 9 — assemble compact paper-ready evidence package + manifest."""
from __future__ import annotations
import json, hashlib, shutil, subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / 'artifacts' / 'phase15_paper_ready_evidence'

# Source -> dest filename
FILES = {
    # Headline numbers
    'results/2026-05-02/phase15_final_clean_leaderboard_20260502_190410.csv': 'phase15_final_clean_leaderboard.csv',
    # Calibration summaries (Phase 14 canonical, val-fit / test-eval)
    'results/2026-05-02/phase14_calibration/clean_calibration_summary_zs_20260502_150835.csv': 'calibration_summary_zs.csv',
    'results/2026-05-02/phase14_calibration/clean_calibration_summary_ft_20260502_151335.csv': 'calibration_summary_ft.csv',
    'results/2026-05-02/phase14_calibration/clean_calibration_summary_aw_20260502_165724.csv': 'calibration_summary_aw.csv',
    # Ensemble + paired comparison
    'results/2026-05-02/phase14_ensemble/clean_ensemble_summary_20260502_165749.csv': 'ensemble_summary.csv',
    'results/2026-05-02/phase14_ensemble/clean_ensemble_paired_comparison_20260502_165749.csv': 'ensemble_paired_comparison.csv',
    # External validation (FiQA polarity-corrected)
    'results/2026-05-02/phase14_external/phase14_external_summary_20260502_165943.csv': 'external_fiqa_summary.csv',
    # Reliability docs
    'docs/PHASE15_FINAL_CLEAN_LEADERBOARD.md': 'PHASE15_FINAL_CLEAN_LEADERBOARD.md',
    'docs/PHASE15_AW_REPRODUCIBILITY_DIAGNOSIS.md': 'PHASE15_AW_REPRODUCIBILITY_DIAGNOSIS.md',
    'docs/PHASE15_EXTERNAL_VALIDATION_CLAIM_BOUNDARIES.md': 'PHASE15_EXTERNAL_VALIDATION_CLAIM_BOUNDARIES.md',
    'docs/PHASE15_REPO_HEAD_AUDIT.md': 'PHASE15_REPO_HEAD_AUDIT.md',
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    PKG.mkdir(parents=True, exist_ok=True)
    git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    entries = []
    for src_rel, dest_name in FILES.items():
        src = ROOT / src_rel
        if not src.exists():
            entries.append({'source': src_rel, 'dest': dest_name, 'status': 'MISSING'})
            continue
        dest = PKG / dest_name
        shutil.copy2(src, dest)
        entries.append({
            'source': src_rel,
            'dest': dest_name,
            'sha256': sha256(dest),
            'size_bytes': dest.stat().st_size,
            'status': 'OK',
        })

    manifest = {
        'phase': 15,
        'generated_at_utc': ts,
        'git_sha': git_sha,
        'description': (
            'Compact paper-ready evidence package for CARA-FinSent. '
            'All numbers come from val-fit / test-eval Phase 14 pipeline; '
            'AW results carry the lower-bound caveat documented in '
            'PHASE15_AW_REPRODUCIBILITY_DIAGNOSIS.md.'
        ),
        'files': entries,
    }
    out = ROOT / 'artifacts' / 'phase15_paper_ready_evidence_manifest.json'
    out.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'[PKG] {PKG}')
    print(f'[OUT] {out}')
    for e in entries:
        print(f"  {e['status']:>7}  {e.get('dest')}")


if __name__ == '__main__':
    main()
