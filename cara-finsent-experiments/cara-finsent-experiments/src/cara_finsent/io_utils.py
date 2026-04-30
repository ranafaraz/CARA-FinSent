from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def timestamp() -> str:
    """UTC timestamp safe for filenames."""
    return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamped_path(output_dir: str | Path, prefix: str, suffix: str = '.csv', ts: Optional[str] = None) -> Path:
    ensure_dir(output_dir)
    ts = ts or timestamp()
    if not suffix.startswith('.'):
        suffix = '.' + suffix
    safe_prefix = prefix.replace(' ', '_').replace('/', '_')
    return Path(output_dir) / f'{safe_prefix}_{ts}{suffix}'


def save_dataframe(df: pd.DataFrame, output_dir: str | Path, prefix: str, ts: Optional[str] = None) -> Path:
    path = timestamped_path(output_dir, prefix, '.csv', ts)
    df.to_csv(path, index=False)
    return path


def save_json(obj: Dict[str, Any], output_dir: str | Path, prefix: str, ts: Optional[str] = None) -> Path:
    path = timestamped_path(output_dir, prefix, '.json', ts)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path


def write_manifest(output_dir: str | Path, run_name: str, files: Dict[str, str], metadata: Optional[Dict[str, Any]] = None, ts: Optional[str] = None) -> Path:
    payload = {
        'run_name': run_name,
        'timestamp_utc': ts or timestamp(),
        'files': files,
        'metadata': metadata or {},
    }
    return save_json(payload, output_dir, f'{run_name}_manifest', ts=payload['timestamp_utc'])
