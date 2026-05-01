#!/usr/bin/env python3
"""Phase 1 controlled splits. Stratified train/val/test split with explicit
duplicate-text guarantees. Never trusts a pre-existing split column.

Outputs:
  data/processed/gold/<dataset>_split_<ts>.csv   (single file with split column)
  data/processed/gold/latest_gold_<dataset>_split.csv   (alias)
  data/audit/split_integrity_report_<ts>.csv
  data/audit/<dataset>_split_manifest_<ts>.json
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from cara_finsent.io_utils import save_dataframe, save_json, timestamp  # noqa: E402
from cara_finsent.label_mapping import VALID_LABELS, text_hash  # noqa: E402


def _stratified_split(df: pd.DataFrame, test_size: float, val_size: float, seed: int):
    """Stratify by label, deduplicate by text_hash before splitting to prevent leakage."""
    df = df.drop_duplicates(subset=['text_hash'], keep='first').reset_index(drop=True)
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df['label']
    )
    rel_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=rel_val, random_state=seed, stratify=train_val['label']
    )
    train['split'] = 'train'
    val['split'] = 'val'
    test['split'] = 'test'
    return pd.concat([train, val, test], ignore_index=True)


def _integrity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    splits = df['split']
    th = {s: set(df.loc[splits == s, 'text_hash']) for s in ['train', 'val', 'test']}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        overlap = th[a] & th[b]
        rows.append({'check': f'leakage_{a}_vs_{b}', 'status': 'FAIL' if overlap else 'PASS', 'detail': f'{len(overlap)} hashes'})
    rows.append({'check': 'split_counts', 'status': 'PASS', 'detail': f'train={len(th["train"])} val={len(th["val"])} test={len(th["test"])}'})
    for s in ['train', 'val', 'test']:
        dist = df[df['split'].eq(s)]['label'].value_counts().to_dict()
        rows.append({'check': f'label_dist_{s}', 'status': 'PASS', 'detail': str(dist)})
    if 'agreement' in df.columns:
        for s in ['train', 'val', 'test']:
            agg = df[df['split'].eq(s)]['agreement'].value_counts().to_dict()
            rows.append({'check': f'agreement_dist_{s}', 'status': 'PASS', 'detail': str(agg)})
    leakage_total = sum(int(r['detail'].split()[0]) for r in rows if r['check'].startswith('leakage_') and r['status'] == 'FAIL')
    rows.append({'check': 'leakage_total', 'status': 'FAIL' if leakage_total else 'PASS', 'detail': f'leakage_count={leakage_total}'})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Clean gold CSV (must have text, label).')
    ap.add_argument('--dataset_name', required=True, help='Logical dataset name (phrasebank, fiqa, ...).')
    ap.add_argument('--test_size', type=float, default=0.20)
    ap.add_argument('--val_size', type=float, default=0.10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output_dir', default='data/processed/gold')
    ap.add_argument('--audit_dir', default='data/audit')
    args = ap.parse_args()

    ts = timestamp()
    src = Path(args.input)
    df = pd.read_csv(src)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise SystemExit(f'[FAIL] {src} missing text/label.')
    df = df[df['label'].isin(VALID_LABELS)].copy()
    if 'text_hash' not in df.columns:
        df['text_hash'] = df['text'].astype(str).map(text_hash)

    split_df = _stratified_split(df, args.test_size, args.val_size, args.seed)
    out_path = save_dataframe(split_df, args.output_dir, f'{args.dataset_name}_split', ts)
    alias = Path(args.output_dir) / f'latest_gold_{args.dataset_name}_split.csv'
    alias.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(out_path), str(alias))

    integrity = _integrity(split_df)
    integrity_path = save_dataframe(integrity, args.audit_dir, f'split_integrity_report_{args.dataset_name}', ts)

    overall = 'FAIL' if (integrity['status'].eq('FAIL').any()) else 'PASS'
    save_json({
        'timestamp_utc': ts,
        'input': str(src),
        'dataset_name': args.dataset_name,
        'seed': args.seed,
        'test_size': args.test_size,
        'val_size': args.val_size,
        'rows_after_dedup': int(len(split_df)),
        'split_counts': split_df['split'].value_counts().to_dict(),
        'output_split_csv': str(out_path),
        'latest_alias': str(alias),
        'integrity_report': str(integrity_path),
        'overall_status': overall,
    }, args.audit_dir, f'{args.dataset_name}_split_manifest', ts)

    print(f'[OK] split saved -> {out_path}')
    print(f'[OK] alias       -> {alias}')
    print('--- INTEGRITY ---')
    print(integrity.to_string(index=False))
    print(f'overall_status={overall}')
    if overall == 'FAIL':
        sys.exit(2)


if __name__ == '__main__':
    main()
