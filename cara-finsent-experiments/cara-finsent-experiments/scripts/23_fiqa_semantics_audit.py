#!/usr/bin/env python3
"""Phase 4: Audit FiQA label semantics before treating it as a benchmark.

Inspects the raw HuggingFace FiQA datasets used by ``00_prepare_phrasebank_fiqa.py``
and reports the raw label/score distribution, the normalized label distribution,
50 random sample rows, and a recommendation as to whether FiQA can serve as a
main benchmark or only as an external stress test.

Outputs
-------
data/audit/fiqa_raw_label_distribution_<ts>.csv
data/audit/fiqa_label_semantics_sample_<ts>.csv
data/audit/fiqa_semantics_audit_manifest_<ts>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from cara_finsent.data_utils import normalize_label  # noqa: E402
from cara_finsent.io_utils import save_dataframe, save_json, timestamp  # noqa: E402

CANDIDATE_DATASETS = [
    'ChanceFocus/fiqa-sentiment-classification',
    'pauri32/fiqa-2018',
    'TheFinAI/fiqa-sentiment',
]

TEXT_KEYS = ('sentence', 'text', 'query', 'headline', 'comment')
LABEL_KEYS = ('label', 'sentiment', 'score', 'sentiment_score', 'aspect_sentiment')


def _first(row: Dict[str, Any], keys) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def load_raw_fiqa(dataset_name: str) -> pd.DataFrame:
    from datasets import load_dataset
    ds_dict = load_dataset(dataset_name)
    rows: List[Dict[str, Any]] = []
    for split_name, ds in ds_dict.items():
        for i, row in enumerate(ds):
            text = _first(row, TEXT_KEYS)
            raw_label = _first(row, LABEL_KEYS)
            normalized = normalize_label(raw_label) if raw_label is not None else None
            rows.append({
                'dataset': dataset_name,
                'split': split_name,
                'row_index': i,
                'text': '' if text is None else str(text),
                'raw_label': raw_label,
                'raw_label_type': type(raw_label).__name__,
                'normalized_label': normalized,
                'all_columns': json.dumps({k: (str(v)[:80] if not isinstance(v, (int, float)) else v)
                                            for k, v in row.items()}, default=str),
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=None,
                    help='Override HF dataset id. If omitted, tries the same candidates the prep script uses.')
    ap.add_argument('--n_samples', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output_dir', default='data/audit')
    args = ap.parse_args()

    ts = timestamp()
    candidates = [args.dataset] if args.dataset else CANDIDATE_DATASETS
    raw_df = None
    used_dataset = None
    errors: List[str] = []
    for name in candidates:
        try:
            print(f'[INFO] Trying {name}...')
            raw_df = load_raw_fiqa(name)
            used_dataset = name
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(f'{name}: {exc}')
            print(f'[WARN] {name}: {exc}')
    if raw_df is None or raw_df.empty:
        raise SystemExit('Failed to load any FiQA candidate dataset. Errors:\n' + '\n'.join(errors))

    print(f'[OK] Loaded {len(raw_df)} rows from {used_dataset}')

    # Raw label distribution
    raw_dist_rows = []
    for split_name, sub in raw_df.groupby('split'):
        raw_counts = sub['raw_label'].astype(str).value_counts()
        for value, cnt in raw_counts.items():
            raw_dist_rows.append({
                'dataset': used_dataset,
                'split': split_name,
                'raw_label_value': value,
                'count': int(cnt),
                'pct': float(cnt) / len(sub),
            })
    raw_dist_df = pd.DataFrame(raw_dist_rows)
    raw_dist_path = save_dataframe(raw_dist_df, args.output_dir, 'fiqa_raw_label_distribution', ts)

    # Normalized label distribution
    norm_counts = raw_df['normalized_label'].fillna('UNMAPPED').value_counts(normalize=False)
    norm_pct = raw_df['normalized_label'].fillna('UNMAPPED').value_counts(normalize=True)

    # Sample rows
    sample = raw_df.sample(min(args.n_samples, len(raw_df)), random_state=args.seed).reset_index(drop=True)
    sample_path = save_dataframe(sample, args.output_dir, 'fiqa_label_semantics_sample', ts)

    # Heuristics
    raw_label_types = raw_df['raw_label_type'].value_counts().to_dict()
    raw_value_types_non_string = float(raw_df['raw_label'].apply(lambda v: isinstance(v, (int, float))).mean())
    has_mostly_numeric = raw_value_types_non_string >= 0.5
    if has_mostly_numeric:
        numeric_vals = pd.to_numeric(raw_df['raw_label'], errors='coerce').dropna()
        score_min = float(numeric_vals.min()) if len(numeric_vals) else float('nan')
        score_max = float(numeric_vals.max()) if len(numeric_vals) else float('nan')
    else:
        score_min = score_max = float('nan')

    suspicious_rows = raw_df[raw_df['normalized_label'].isna()]
    n_unmapped = int(len(suspicious_rows))

    # Recommendation
    fiqa_main_benchmark = (
        n_unmapped == 0
        and not has_mostly_numeric
        and norm_pct.get('positive', 0) < 0.85
        and norm_pct.get('negative', 0) < 0.85
        and norm_pct.get('neutral', 0) < 0.85
    )
    fiqa_external_stress_test = True

    manifest = {
        'timestamp_utc': ts,
        'dataset_used': used_dataset,
        'candidate_errors': errors,
        'rows_total': int(len(raw_df)),
        'splits': {k: int(v) for k, v in raw_df['split'].value_counts().to_dict().items()},
        'raw_label_types': {k: int(v) for k, v in raw_label_types.items()},
        'raw_label_mostly_numeric': has_mostly_numeric,
        'raw_score_min': score_min,
        'raw_score_max': score_max,
        'normalized_label_counts': {str(k): int(v) for k, v in norm_counts.items()},
        'normalized_label_pct': {str(k): float(v) for k, v in norm_pct.items()},
        'unmapped_rows': n_unmapped,
        'fiqa_main_benchmark': bool(fiqa_main_benchmark),
        'fiqa_external_stress_test': bool(fiqa_external_stress_test),
        'recommendation_reason': (
            'Labels normalize cleanly to multiple classes; eligible as main benchmark.'
            if fiqa_main_benchmark
            else 'Labels appear numeric/imbalanced or contain unmapped rows; treat as external stress test.'
        ),
        'raw_distribution_csv': str(raw_dist_path),
        'sample_csv': str(sample_path),
    }
    manifest_path = save_json(manifest, args.output_dir, 'fiqa_semantics_audit_manifest', ts)
    print(f'[DONE] Raw distribution -> {raw_dist_path}')
    print(f'[DONE] Sample           -> {sample_path}')
    print(f'[DONE] Manifest         -> {manifest_path}')
    print(f'       fiqa_main_benchmark={fiqa_main_benchmark}, fiqa_external_stress_test={fiqa_external_stress_test}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
