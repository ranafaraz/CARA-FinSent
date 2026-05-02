#!/usr/bin/env python3
"""Phase 15 Task 8 — classify result/artifact files for cleanup; do NOT delete."""
from __future__ import annotations
import csv
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def classify(p: Path) -> tuple[str, str]:
    rel = p.relative_to(ROOT).as_posix()
    name = p.name.lower()
    parts = rel.split('/')

    # Hard rule: never touch raw data, models, source, scripts, docs, configs
    if parts[0] in {'src', 'scripts', 'configs', 'docs', 'data', 'models', 'tests', 'notebooks'}:
        return 'unsafe_to_delete', 'core repo content'

    # Phase 15 / Phase 14 evidence directories — keep
    if 'phase15_' in rel or 'phase14_' in rel:
        return 'keep_main_evidence', 'phase 14/15 final evidence'

    if rel.startswith('artifacts/phase15_'):
        return 'keep_main_evidence', 'phase 15 paper-ready package'

    # Older phases archived — keep as appendix
    if rel.startswith('results/_archive') or rel.startswith('artifacts/phase8') or rel.startswith('artifacts/phase9') or rel.startswith('artifacts/phase10') or rel.startswith('artifacts/phase11') or rel.startswith('artifacts/phase12') or rel.startswith('artifacts/phase13'):
        return 'keep_appendix', 'historical phase artefact (paper appendix only)'

    if rel.startswith('logs/') and ('phase15' in rel or 'phase14' in rel):
        return 'keep_main_evidence', 'phase 14/15 training/eval log'
    if rel.startswith('logs/'):
        return 'keep_appendix', 'historical log'

    # Caches and tempdirs
    if any(seg in {'__pycache__', '.pytest_cache', '.ipynb_checkpoints', '.cache', 'tmp', 'tmp_'} for seg in parts):
        return 'cache_or_temp', 'cache or temporary file'
    if name.endswith('.pyc') or name.endswith('.tmp'):
        return 'cache_or_temp', 'compiled/temp file'

    # Figures keep
    if parts[0] == 'figures':
        return 'keep_main_evidence', 'reporting figure'

    # Default: anything else under results/ or artifacts/ that isn't classified
    if parts[0] in {'results', 'artifacts'}:
        return 'duplicate_candidate', 'review manually before deleting; not referenced by Phase 15 docs'

    return 'unsafe_to_delete', 'unclassified — leave as-is'


def main():
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    rows = []
    for p in ROOT.rglob('*'):
        if not p.is_file():
            continue
        try:
            rel_first = p.relative_to(ROOT).parts[0]
        except ValueError:
            continue
        if rel_first in {'.git', '.venv', 'venv', 'env', 'node_modules'}:
            continue
        if rel_first not in {'results', 'artifacts', 'logs', 'figures'}:
            continue
        cls, reason = classify(p)
        try:
            size = p.stat().st_size
        except OSError:
            size = -1
        rows.append({
            'path': p.relative_to(ROOT).as_posix(),
            'classification': cls,
            'reason': reason,
            'size_bytes': size,
        })

    out_dir = ROOT / 'results' / '2026-05-02'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'phase15_cleanup_manifest_{ts}.csv'
    with out_path.open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['path', 'classification', 'reason', 'size_bytes'])
        w.writeheader()
        w.writerows(rows)

    summary = {}
    for r in rows:
        summary[r['classification']] = summary.get(r['classification'], 0) + 1
    print(f'[OUT] {out_path}')
    for k, v in sorted(summary.items()):
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
