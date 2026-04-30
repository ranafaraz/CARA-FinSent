#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import argparse
import os
import time

import pandas as pd
import requests

from cara_finsent.data_utils import normalize_label
from cara_finsent.io_utils import save_dataframe, timestamp, write_manifest

BASE_URL = 'https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json'


def fetch_symbol(symbol: str, limit: int, token: str | None = None) -> list[dict]:
    params = {'limit': min(limit, 30)}
    if token:
        params['access_token'] = token
    r = requests.get(BASE_URL.format(symbol=symbol.upper()), params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows = []
    for msg in data.get('messages', []):
        sentiment_obj = ((msg.get('entities') or {}).get('sentiment') or {})
        sentiment_raw = sentiment_obj.get('basic')
        rows.append({
            'id': msg.get('id'),
            'symbol_query': symbol.upper(),
            'created_at': msg.get('created_at'),
            'text': msg.get('body'),
            'sentiment_raw': sentiment_raw,
            'label': normalize_label(sentiment_raw) if sentiment_raw else '',
            'source_dataset': 'stocktwits_api',
            'user': (msg.get('user') or {}).get('username'),
            'symbols': ','.join([s.get('symbol', '') for s in (msg.get('symbols') or [])]),
            'source_url': f"https://stocktwits.com/message/{msg.get('id')}" if msg.get('id') else '',
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description='Collect labeled/unlabeled StockTwits messages via the public StockTwits API.')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT', 'TSLA', 'NVDA'], help='Ticker symbols to query.')
    parser.add_argument('--limit_per_symbol', type=int, default=30, help='API usually caps at about 30 messages per request.')
    parser.add_argument('--sleep', type=float, default=2.0, help='Polite delay between requests.')
    parser.add_argument('--output_dir', type=str, default='data/external')
    args = parser.parse_args()

    token = os.getenv('STOCKTWITS_ACCESS_TOKEN')
    ts = timestamp()
    all_rows = []
    for sym in args.symbols:
        try:
            rows = fetch_symbol(sym, args.limit_per_symbol, token=token)
            all_rows.extend(rows)
            print(f'[OK] {sym}: {len(rows)} messages')
        except Exception as exc:
            print(f'[WARN] {sym} failed: {exc}')
        time.sleep(args.sleep)

    df = pd.DataFrame(all_rows)
    path = save_dataframe(df, args.output_dir, 'stocktwits_messages', ts)
    labeled = df[df.get('label', '').astype(str).isin(['positive', 'negative', 'neutral'])] if len(df) else df
    labeled_path = save_dataframe(labeled, args.output_dir, 'stocktwits_labeled_only', ts)
    manifest = write_manifest('results', 'stocktwits_collection', {'all_messages': str(path), 'labeled_only': str(labeled_path)}, {'symbols': args.symbols, 'rows': int(len(df))}, ts)
    print(f'[DONE] Saved {len(df)} rows -> {path}')
    print(f'[DONE] Manifest -> {manifest}')


if __name__ == '__main__':
    main()
