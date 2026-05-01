#!/usr/bin/env python3
"""Phase 4: Build an external retrieval corpus for the future RAG component.

The retrieval corpus must be **external-only**: it must never include the
controlled gold test text or labels. This script aggregates rows from
provided source files (financial news, SEC filings, company descriptions,
sector metadata, RSS exports), enforces the required schema, deduplicates by
``text_hash``, and excludes any rows whose normalized text matches a row in
any controlled gold split.

Usage examples:
  python scripts/24_build_external_retrieval_corpus.py \
      --inputs data/external/news_headlines.csv data/external/sec_descriptions.csv \
      --gold_splits data/processed/gold/latest_gold_phrasebank_split.csv \
                    data/processed/gold/latest_gold_fiqa_split.csv

If no inputs are provided the script writes an empty schema-conformant
``latest_retrieval_corpus.csv`` with zero rows so downstream tooling can detect
the absence of a corpus.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from cara_finsent.io_utils import ensure_dir, save_dataframe, save_json, timestamp  # noqa: E402
from cara_finsent.label_mapping import text_hash  # noqa: E402

REQUIRED_SCHEMA = ['id', 'source', 'text', 'ticker', 'company', 'date', 'url', 'doc_type', 'text_hash']

TEXT_CANDIDATES = ['text', 'sentence', 'headline', 'title', 'body', 'content', 'description', 'summary']
SOURCE_CANDIDATES = ['source', 'source_dataset', 'dataset']


def _detect(df: pd.DataFrame, candidates: List[str], default: Optional[str] = None) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return default


def load_input(path: Path, default_source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    text_col = _detect(df, TEXT_CANDIDATES)
    if text_col is None:
        raise ValueError(f'No text column found in {path}. Tried {TEXT_CANDIDATES}.')

    out = pd.DataFrame()
    out['text'] = df[text_col].astype(str).fillna('').str.strip()
    src_col = _detect(df, SOURCE_CANDIDATES)
    out['source'] = df[src_col].astype(str) if src_col else default_source
    out['ticker'] = df['ticker'] if 'ticker' in df.columns else ''
    out['company'] = df['company'] if 'company' in df.columns else ''
    out['date'] = df['date'] if 'date' in df.columns else ''
    out['url'] = df['url'] if 'url' in df.columns else ''
    out['doc_type'] = df['doc_type'] if 'doc_type' in df.columns else default_source

    out = out[out['text'].str.len() > 0].reset_index(drop=True)
    out['text_hash'] = out['text'].apply(text_hash)
    out['id'] = [f'{path.stem}_{i}' for i in range(len(out))]
    return out[REQUIRED_SCHEMA]


def load_gold_hashes(paths: List[Path]) -> set:
    hashes: set = set()
    for path in paths:
        df = pd.read_csv(path)
        if 'text_hash' in df.columns:
            hashes.update(df['text_hash'].astype(str).tolist())
        elif 'text' in df.columns:
            hashes.update(df['text'].astype(str).apply(text_hash).tolist())
    return hashes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='*', default=[], help='Source CSV files (text + optional metadata).')
    ap.add_argument('--gold_splits', nargs='*', default=[
        'data/processed/gold/latest_gold_phrasebank_split.csv',
        'data/processed/gold/latest_gold_fiqa_split.csv',
    ])
    ap.add_argument('--output_dir', default='data/retrieval_corpus')
    ap.add_argument('--audit_dir', default='data/audit')
    ap.add_argument('--default_source', default='external')
    args = ap.parse_args()

    ts = timestamp()
    out_dir = ensure_dir(Path(args.output_dir))

    gold_paths = [Path(p) for p in args.gold_splits if Path(p).exists()]
    gold_hashes = load_gold_hashes(gold_paths)
    print(f'[INFO] Loaded {len(gold_hashes)} gold-split text hashes from {len(gold_paths)} file(s).')

    frames: List[pd.DataFrame] = []
    skipped: List[dict] = []
    for input_path in args.inputs:
        path = Path(input_path)
        if not path.exists():
            print(f'[WARN] Input not found: {path}')
            skipped.append({'input': str(path), 'reason': 'not_found'})
            continue
        try:
            df = load_input(path, args.default_source)
            print(f'[OK] {path}: {len(df)} rows')
            frames.append(df)
        except Exception as exc:  # noqa: BLE001
            print(f'[FAIL] {path}: {exc}')
            skipped.append({'input': str(path), 'reason': str(exc)})

    if frames:
        corpus = pd.concat(frames, ignore_index=True)
    else:
        corpus = pd.DataFrame(columns=REQUIRED_SCHEMA)

    pre_dedup = len(corpus)
    if pre_dedup:
        corpus = corpus.drop_duplicates(subset=['text_hash']).reset_index(drop=True)

    pre_gold_filter = len(corpus)
    if pre_gold_filter and gold_hashes:
        corpus = corpus[~corpus['text_hash'].isin(gold_hashes)].reset_index(drop=True)
    post_gold_filter = len(corpus)

    timestamped_path = save_dataframe(corpus, args.output_dir, 'retrieval_corpus', ts)
    latest_path = out_dir / 'latest_retrieval_corpus.csv'
    corpus.to_csv(latest_path, index=False)

    audit_rows = [{
        'timestamp_utc': ts,
        'inputs': len(args.inputs),
        'gold_split_files': len(gold_paths),
        'gold_hashes': len(gold_hashes),
        'rows_pre_dedup': pre_dedup,
        'rows_post_dedup': pre_gold_filter,
        'rows_post_gold_filter': post_gold_filter,
        'skipped_inputs': len(skipped),
        'corpus_file': str(timestamped_path),
        'latest_file': str(latest_path),
    }]
    audit_df = pd.DataFrame(audit_rows)
    audit_path = save_dataframe(audit_df, args.audit_dir, 'retrieval_corpus_audit', ts)

    manifest = {
        'timestamp_utc': ts,
        'inputs': args.inputs,
        'gold_splits_used': [str(p) for p in gold_paths],
        'rows_pre_dedup': pre_dedup,
        'rows_post_dedup': pre_gold_filter,
        'rows_post_gold_filter': post_gold_filter,
        'corpus_file': str(timestamped_path),
        'latest_file': str(latest_path),
        'audit_csv': str(audit_path),
        'skipped': skipped,
        'schema': REQUIRED_SCHEMA,
    }
    save_json(manifest, args.audit_dir, 'retrieval_corpus_manifest', ts)
    print(f'[DONE] Corpus: {post_gold_filter} rows -> {timestamped_path}')
    print(f'[DONE] Audit:  {audit_path}')
    if post_gold_filter == 0 and args.inputs:
        print('[WARN] Corpus is empty after gold-split filtering. Provide additional external sources.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
