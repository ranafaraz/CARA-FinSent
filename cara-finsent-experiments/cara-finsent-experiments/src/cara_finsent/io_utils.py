from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def timestamp() -> str:
    """UTC timestamp safe for filenames: YYYYMMDD_HHMMSS."""
    return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')


def date_subdir(ts: Optional[str] = None) -> str:
    """Return a YYYY-MM-DD string derived from *ts* (YYYYMMDD_HHMMSS) or from now."""
    if ts and len(ts) >= 8:
        raw = ts[:8]
        return f'{raw[:4]}-{raw[4:6]}-{raw[6:8]}'
    return datetime.now(timezone.utc).strftime('%Y-%m-%d')


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamped_path(output_dir: str | Path, prefix: str, suffix: str = '.csv', ts: Optional[str] = None) -> Path:
    """Return ``output_dir/YYYY-MM-DD/prefix_YYYYMMDD_HHMMSS.suffix``.

    The date subfolder is created automatically so every run's outputs are
    grouped by the date they were produced, avoiding flat-directory clutter.
    """
    ts = ts or timestamp()
    dated_dir = Path(output_dir) / date_subdir(ts)
    ensure_dir(dated_dir)
    if not suffix.startswith('.'):
        suffix = '.' + suffix
    safe_prefix = prefix.replace(' ', '_').replace('/', '_')
    return dated_dir / f'{safe_prefix}_{ts}{suffix}'


def save_dataframe(df: pd.DataFrame, output_dir: str | Path, prefix: str, ts: Optional[str] = None) -> Path:
    path = timestamped_path(output_dir, prefix, '.csv', ts)
    df.to_csv(path, index=False)
    return path


def save_json(obj: Dict[str, Any], output_dir: str | Path, prefix: str, ts: Optional[str] = None) -> Path:
    path = timestamped_path(output_dir, prefix, '.json', ts)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path


def save_skip_report(skipped: list[Dict[str, Any]], output_dir: str | Path, prefix: str, ts: Optional[str] = None) -> Optional[Path]:
    """Save skipped-source records as CSV and return its path.

    Returns None when there are no skipped records.
    """
    if not skipped:
        return None
    df = pd.DataFrame(skipped)
    return save_dataframe(df, output_dir, prefix, ts)


def git_commit_sha(repo_root: str | Path | None = None) -> str:
    """Return the current git commit SHA or ``unknown`` outside a repo."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return 'unknown'


def write_manifest(output_dir: str | Path, run_name: str, files: Dict[str, str], metadata: Optional[Dict[str, Any]] = None, ts: Optional[str] = None) -> Path:
    payload = {
        'run_name': run_name,
        'timestamp_utc': ts or timestamp(),
        'files': files,
        'metadata': metadata or {},
    }
    return save_json(payload, output_dir, f'{run_name}_manifest', ts=payload['timestamp_utc'])
