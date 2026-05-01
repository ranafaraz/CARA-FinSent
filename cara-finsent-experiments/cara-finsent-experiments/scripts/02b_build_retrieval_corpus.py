#!/usr/bin/env python3
"""Phase 1: Build retrieval corpus from SEC 10-K and news RSS data.
No gold sentiment labels required. External-only corpus for RAG context retrieval.

Outputs:
  data/processed/retrieval/retrieval_corpus_<ts>.csv
  data/processed/retrieval/latest_retrieval_corpus.csv (alias)
  results/YYYY-MM-DD/retrieval_corpus_manifest_<ts>.json
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from cara_finsent.io_utils import save_dataframe, save_json, timestamp  # noqa: E402


def _build_sec_retrieval() -> pd.DataFrame:
    """Load SEC 10-K snippets."""
    sec_raw = Path('data/raw/sec_10k')
    if not sec_raw.exists():
        print('[WARN] data/raw/sec_10k not found; skipping SEC corpus.')
        return pd.DataFrame()
    csvs = list(sec_raw.glob('*.csv'))
    if not csvs:
        print('[WARN] no CSVs in data/raw/sec_10k; skipping SEC corpus.')
        return pd.DataFrame()
    frames = [pd.read_csv(f) for f in csvs]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    df['source_type'] = 'sec_10k'
    df['source_name'] = df.get('source_name', 'SEC EDGAR')
    for col in ['doc_id', 'text', 'source_type', 'source_name']:
        if col not in df.columns:
            if col == 'doc_id' and 'id' in df.columns:
                df['doc_id'] = df['id']
            elif col == 'text' and 'snippet' in df.columns:
                df['text'] = df['snippet']
            else:
                df[col] = ''
    return df[['doc_id', 'text', 'source_type', 'source_name']].copy()


def _build_news_retrieval() -> pd.DataFrame:
    """Load news RSS snippets."""
    news_raw = Path('data/raw/financial_news')
    if not news_raw.exists():
        print('[WARN] data/raw/financial_news not found; skipping news corpus.')
        return pd.DataFrame()
    csvs = list(news_raw.glob('*.csv'))
    if not csvs:
        print('[WARN] no CSVs in data/raw/financial_news; skipping news corpus.')
        return pd.DataFrame()
    frames = [pd.read_csv(f) for f in csvs]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    df['source_type'] = 'news'
    df['source_name'] = df.get('source', 'Financial News RSS')
    for col in ['doc_id', 'text', 'source_type', 'source_name']:
        if col not in df.columns:
            if col == 'doc_id' and 'id' in df.columns:
                df['doc_id'] = df['id']
            elif col == 'text' and 'headline' in df.columns:
                df['text'] = df['headline']
            elif col == 'text' and 'title' in df.columns:
                df['text'] = df['title']
            else:
                df[col] = ''
    return df[['doc_id', 'text', 'source_type', 'source_name']].copy()


def _standardize_retrieval(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize schema: doc_id, text, source_type, source_name, ticker, company, published_at, url."""
    out = df.copy()
    for col in ['doc_id', 'text', 'source_type', 'source_name', 'ticker', 'company', 'published_at', 'url']:
        if col not in out.columns:
            out[col] = ''
    out['text'] = out['text'].astype(str).str.strip()
    out = out[out['text'].str.len() > 0].copy()
    out = out[['doc_id', 'text', 'source_type', 'source_name', 'ticker', 'company', 'published_at', 'url']].reset_index(drop=True)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_dir', default='data/processed/retrieval')
    ap.add_argument('--results_dir', default='results')
    args = ap.parse_args()

    ts = timestamp()
    print('[INFO] Building Phase 1 retrieval corpus...')

    sec = _build_sec_retrieval()
    news = _build_news_retrieval()
    frames = [f for f in [sec, news] if not f.empty]

    if not frames:
        print('[WARN] No retrieval data sources found. Creating empty corpus.')
        frames = [pd.DataFrame(columns=['doc_id', 'text', 'source_type', 'source_name', 'ticker', 'company', 'published_at', 'url'])]

    combined = pd.concat(frames, ignore_index=True)
    combined = _standardize_retrieval(combined)
    combined = combined.drop_duplicates(subset=['text']).reset_index(drop=True)

    path = save_dataframe(combined, args.output_dir, 'retrieval_corpus', ts)
    alias = Path(args.output_dir) / 'latest_retrieval_corpus.csv'
    alias.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(str(path), str(alias))

    manifest = {
        'timestamp_utc': ts,
        'source_counts': {
            'sec_10k': int(len(sec)) if not sec.empty else 0,
            'news': int(len(news)) if not news.empty else 0,
        },
        'total_rows': int(len(combined)),
        'output_csv': str(path),
        'latest_alias': str(alias),
    }
    save_json(manifest, args.results_dir, 'retrieval_corpus_manifest', ts)

    print(f'[OK] Retrieval corpus rows: {len(combined)} -> {path}')
    print(f'[OK] Alias -> {alias}')
    print(f'     Source breakdown: SEC={len(sec)}, News={len(news)}')


if __name__ == '__main__':
    main()
