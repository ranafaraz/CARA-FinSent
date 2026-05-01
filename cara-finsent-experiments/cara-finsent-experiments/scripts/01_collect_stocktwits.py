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
from cara_finsent.io_utils import save_dataframe, save_skip_report, timestamp, write_manifest
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


LEGACY_BASE_URL = 'https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json'
GATEWAY_BASE_URL = 'https://api-gw-prd.stocktwits.com/api-middleware/external/api/2/streams/symbol/{symbol}.json'


def fetch_symbol(
    symbol: str,
    limit: int,
    token: str | None = None,
    basic_username: str | None = None,
    basic_password: str | None = None,
) -> list[dict]:
    params = {'limit': min(limit, 30)}
    request_kwargs: dict = {'params': params, 'timeout': 30}

    if basic_username and basic_password:
        url = GATEWAY_BASE_URL.format(symbol=symbol.upper())
        request_kwargs['auth'] = (basic_username, basic_password)
    else:
        url = LEGACY_BASE_URL.format(symbol=symbol.upper())
        if token:
            params['access_token'] = token

    r = requests.get(url, **request_kwargs)
    if r.status_code == 401 and basic_username and basic_password:
        # Some accounts have website credentials but no gateway API entitlement.
        # Fallback to legacy public endpoint so collection can continue.
        legacy_params = {'limit': min(limit, 30)}
        if token:
            legacy_params['access_token'] = token
        r = requests.get(LEGACY_BASE_URL.format(symbol=symbol.upper()), params=legacy_params, timeout=30)
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
    basic_username = os.getenv('STOCKTWITS_API_USERNAME')
    basic_password = os.getenv('STOCKTWITS_API_PASSWORD')

    if basic_username and basic_password:
        print('[INFO] Using StockTwits gateway API with HTTP Basic auth credentials from env.')
    elif token:
        print('[INFO] Using legacy StockTwits API with access token from env.')
    else:
        print('[INFO] No StockTwits credentials found; using legacy public stream endpoint.')

    ts = timestamp()
    all_rows = []
    skipped_sources = []
    for sym in args.symbols:
        try:
            rows = fetch_symbol(
                sym,
                args.limit_per_symbol,
                token=token,
                basic_username=basic_username,
                basic_password=basic_password,
            )
            all_rows.extend(rows)
            print(f'[OK] {sym}: {len(rows)} messages')
        except Exception as exc:
            print(f'[WARN] {sym} failed: {exc}')
            skipped_sources.append({
                'source': 'stocktwits',
                'symbol': sym,
                'reason': str(exc),
                'action': 'skipped',
            })
        time.sleep(args.sleep)

    df = pd.DataFrame(all_rows)
    path = save_dataframe(df, args.output_dir, 'stocktwits_messages', ts)
    labeled = df[df.get('label', '').astype(str).isin(['positive', 'negative', 'neutral'])] if len(df) else df
    labeled_path = save_dataframe(labeled, args.output_dir, 'stocktwits_labeled_only', ts)
    skip_report_path = save_skip_report(skipped_sources, 'results', 'stocktwits_skipped_sources', ts)
    files = {'all_messages': str(path), 'labeled_only': str(labeled_path)}
    if skip_report_path is not None:
        files['skip_report_csv'] = str(skip_report_path)
    manifest = write_manifest(
        'results',
        'stocktwits_collection',
        files,
        {
            'symbols': args.symbols,
            'rows': int(len(df)),
            'skipped_sources_count': len(skipped_sources),
        },
        ts,
    )
    print(f'[DONE] Saved {len(df)} rows -> {path}')
    print(f'[DONE] Manifest -> {manifest}')


if __name__ == '__main__':
    main()
