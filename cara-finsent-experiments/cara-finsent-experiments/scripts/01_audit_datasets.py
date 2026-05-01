#!/usr/bin/env python3
"""Phase 1 data audit. Reports row counts, label distributions, duplicates,
text-length distribution, and split leakage. Read-only: never modifies inputs.

Outputs (timestamped):
  data/audit/data_audit_summary_<ts>.csv
  data/audit/label_distribution_<ts>.csv
  data/audit/duplicate_report_<ts>.csv
  data/audit/split_integrity_report_<ts>.csv
  data/audit/text_length_<ts>.csv

Status per check is PASS / WARN / FAIL. The script prints the overall_status
line so a CI/CD step can grep for it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from cara_finsent.io_utils import save_dataframe, save_json, timestamp  # noqa: E402
from cara_finsent.label_mapping import (  # noqa: E402
    VALID_LABELS,
    canonical_label,
    text_hash,
)


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise SystemExit(f'[FAIL] {path} missing required columns (text,label).')
    if 'source_dataset' not in df.columns:
        df['source_dataset'] = path.stem
    if 'text_hash' not in df.columns:
        df['text_hash'] = df['text'].astype(str).map(text_hash)
    return df


def _length_stats(df: pd.DataFrame) -> pd.DataFrame:
    lens = df['text'].astype(str).str.len()
    rows = []
    for src, g in df.groupby('source_dataset'):
        ls = g['text'].astype(str).str.len()
        rows.append({
            'source_dataset': src,
            'rows': int(len(g)),
            'len_min': int(ls.min()),
            'len_max': int(ls.max()),
            'len_mean': float(ls.mean()),
            'len_median': float(ls.median()),
            'len_p95': float(ls.quantile(0.95)),
        })
    rows.append({
        'source_dataset': '__overall__',
        'rows': int(len(df)),
        'len_min': int(lens.min()),
        'len_max': int(lens.max()),
        'len_mean': float(lens.mean()),
        'len_median': float(lens.median()),
        'len_p95': float(lens.quantile(0.95)),
    })
    return pd.DataFrame(rows)


def _duplicate_report(df: pd.DataFrame) -> pd.DataFrame:
    dup = df.groupby('text_hash').agg(
        n=('text_hash', 'size'),
        n_unique_labels=('label', lambda s: s.nunique()),
        labels=('label', lambda s: ','.join(sorted(set(map(str, s))))),
        sources=('source_dataset', lambda s: ','.join(sorted(set(map(str, s))))),
        sample_text=('text', 'first'),
    )
    return dup[dup['n'] > 1].reset_index()


def _split_integrity(df: pd.DataFrame) -> pd.DataFrame:
    if 'split' not in df.columns:
        return pd.DataFrame([{'check': 'split_present', 'status': 'WARN', 'detail': 'no split column'}])
    rows: List[dict] = []
    splits = df['split'].astype(str).str.lower()
    train_h = set(df.loc[splits.eq('train'), 'text_hash'])
    val_h = set(df.loc[splits.isin(['val', 'valid', 'validation']), 'text_hash'])
    test_h = set(df.loc[splits.eq('test'), 'text_hash'])
    pairs = [('train_vs_val', train_h & val_h), ('train_vs_test', train_h & test_h), ('val_vs_test', val_h & test_h)]
    total_leak = 0
    for name, overlap in pairs:
        rows.append({'check': f'leakage_{name}', 'status': 'FAIL' if overlap else 'PASS', 'detail': f'{len(overlap)} overlapping text_hash'})
        total_leak += len(overlap)
    rows.append({'check': 'leakage_total', 'status': 'FAIL' if total_leak else 'PASS', 'detail': f'leakage_count={total_leak}'})
    rows.append({'check': 'split_counts', 'status': 'PASS', 'detail': f'train={len(train_h)}, val={len(val_h)}, test={len(test_h)}'})
    return pd.DataFrame(rows)


def _label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['source_dataset', 'label']
    if 'split' in df.columns:
        cols.insert(1, 'split')
    return df.groupby(cols).size().reset_index(name='rows')


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    n_total = len(df)
    n_missing_text = int(df['text'].astype(str).str.strip().eq('').sum())
    n_missing_label = int(df['label'].isna().sum())
    invalid_mask = ~df['label'].astype(str).map(lambda v: canonical_label(v) in VALID_LABELS)
    n_invalid_label = int(invalid_mask.sum())
    dup = _duplicate_report(df)
    n_dup_groups = int(len(dup))
    n_dup_conflicts = int((dup['n_unique_labels'] > 1).sum()) if len(dup) else 0
    rows += [
        {'check': 'rows_total', 'status': 'PASS', 'detail': str(n_total)},
        {'check': 'missing_text', 'status': 'PASS' if n_missing_text == 0 else 'FAIL', 'detail': str(n_missing_text)},
        {'check': 'missing_label', 'status': 'PASS' if n_missing_label == 0 else 'FAIL', 'detail': str(n_missing_label)},
        {'check': 'invalid_label', 'status': 'PASS' if n_invalid_label == 0 else 'FAIL', 'detail': str(n_invalid_label)},
        {'check': 'duplicate_text_groups', 'status': 'PASS' if n_dup_groups == 0 else 'WARN', 'detail': str(n_dup_groups)},
        {'check': 'duplicate_label_conflicts', 'status': 'PASS' if n_dup_conflicts == 0 else 'FAIL', 'detail': str(n_dup_conflicts)},
    ]
    if 'agreement' in df.columns:
        agg = df['agreement'].dropna().astype(float)
        rows.append({'check': 'agreement_present', 'status': 'PASS', 'detail': f'rows_with_agreement={len(agg)}'})
    overall = 'PASS' if all(r['status'] != 'FAIL' for r in rows) else 'FAIL'
    rows.append({'check': 'overall_status', 'status': overall, 'detail': 'gating value'})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True, help='One or more CSV files to audit.')
    ap.add_argument('--output_dir', default='data/audit')
    args = ap.parse_args()

    ts = timestamp()
    paths = []
    for pat in args.inputs:
        paths.extend([Path(p) for p in sorted(Path().glob(pat))]) if any(c in pat for c in '*?[') else paths.append(Path(pat))
    paths = [p for p in paths if p.exists() and p.is_file()]
    if not paths:
        raise SystemExit('[FAIL] no input files matched.')

    print('[INFO] auditing files:')
    for p in paths:
        print(f'  - {p}')

    frames = []
    for p in paths:
        df = _load(p)
        df['__source_file__'] = p.name
        frames.append(df)
    big = pd.concat(frames, ignore_index=True)

    summary = _summary(big)
    label_dist = _label_distribution(big)
    dup = _duplicate_report(big)
    split = _split_integrity(big)
    lengths = _length_stats(big)

    summary_path = save_dataframe(summary, args.output_dir, 'data_audit_summary', ts)
    label_path = save_dataframe(label_dist, args.output_dir, 'label_distribution', ts)
    dup_path = save_dataframe(dup if len(dup) else pd.DataFrame(columns=['text_hash']), args.output_dir, 'duplicate_report', ts)
    split_path = save_dataframe(split, args.output_dir, 'split_integrity_report', ts)
    len_path = save_dataframe(lengths, args.output_dir, 'text_length', ts)

    overall = summary.loc[summary['check'].eq('overall_status'), 'status'].iloc[0]
    leakage_total_row = split[split['check'].eq('leakage_total')]
    leakage_status = leakage_total_row['status'].iloc[0] if len(leakage_total_row) else 'WARN'

    manifest = {
        'timestamp_utc': ts,
        'inputs': [str(p) for p in paths],
        'outputs': {
            'data_audit_summary': str(summary_path),
            'label_distribution': str(label_path),
            'duplicate_report': str(dup_path),
            'split_integrity_report': str(split_path),
            'text_length': str(len_path),
        },
        'overall_status': overall,
        'leakage_status': leakage_status,
    }
    save_json(manifest, args.output_dir, 'data_audit_manifest', ts)

    print('--- SUMMARY ---')
    print(summary.to_string(index=False))
    print('--- SPLIT INTEGRITY ---')
    print(split.to_string(index=False))
    print(f'overall_status={overall}')
    print(f'leakage_status={leakage_status}')
    audit_gate = 'PASS' if overall == 'PASS' and leakage_status != 'FAIL' else 'FAIL'
    print(f'audit_gate={audit_gate}')
    if audit_gate == 'FAIL':
        sys.exit(2)


if __name__ == '__main__':
    main()
