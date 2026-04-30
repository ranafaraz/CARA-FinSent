#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import argparse
import os
from pathlib import Path
from datetime import datetime

import pandas as pd

from cara_finsent.feature_extractor import FinancialFeatureExtractor, weak_lexicon_label
from cara_finsent.io_utils import save_dataframe, timestamp, write_manifest
from cara_finsent.sec_utils import fetch_filing_text, fetch_submissions, load_ticker_cik_map, rough_extract_item, sleep_polite


def main():
    parser = argparse.ArgumentParser(description='Collect SEC 10-K filing text via official SEC endpoints and create weak sentiment labels.')
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT', 'NVDA', 'TSLA'], help='Ticker symbols.')
    parser.add_argument('--years', type=int, default=3, help='Keep filings from the last N years.')
    parser.add_argument('--max_filings_per_ticker', type=int, default=2)
    parser.add_argument('--sleep', type=float, default=0.25, help='SEC polite delay between requests.')
    parser.add_argument('--output_dir', type=str, default='data/external')
    parser.add_argument('--text_dir', type=str, default='data/external/sec_filings')
    args = parser.parse_args()

    user_agent = os.getenv('SEC_USER_AGENT')
    if not user_agent:
        raise SystemExit('Set SEC_USER_AGENT first. Example: export SEC_USER_AGENT="Rana Research rana@example.com"')

    ts = timestamp()
    ticker_map = load_ticker_cik_map(user_agent)
    text_dir = Path(args.text_dir)
    text_dir.mkdir(parents=True, exist_ok=True)
    cutoff_year = datetime.utcnow().year - args.years
    extractor = FinancialFeatureExtractor()

    rows = []
    for ticker in args.tickers:
        ticker = ticker.upper()
        cik = ticker_map.get(ticker)
        if not cik:
            print(f'[WARN] No CIK found for {ticker}')
            continue
        try:
            sub = fetch_submissions(cik, user_agent)
            recent = sub.get('filings', {}).get('recent', {})
            forms = recent.get('form', [])
            accs = recent.get('accessionNumber', [])
            dates = recent.get('filingDate', [])
            docs = recent.get('primaryDocument', [])
            count = 0
            for form, acc, filing_date, doc in zip(forms, accs, dates, docs):
                if form != '10-K':
                    continue
                year = int(str(filing_date)[:4])
                if year < cutoff_year:
                    continue
                if count >= args.max_filings_per_ticker:
                    break
                try:
                    text, url = fetch_filing_text(cik, acc, doc, user_agent)
                    item_1a = rough_extract_item(text, '1A')
                    item_7 = rough_extract_item(text, '7')
                    core_text = item_1a or item_7 or text[:200000]
                    out_path = text_dir / ticker / f'{acc.replace("-", "")}.txt'
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(core_text, encoding='utf-8')
                    feats = extractor.extract_one(core_text[:50000])
                    rows.append({
                        'id': f'sec10k_{ticker}_{acc}',
                        'ticker': ticker,
                        'cik': cik,
                        'accession': acc,
                        'filing_date': filing_date,
                        'source_url': url,
                        'text_path': str(out_path),
                        'text': core_text[:5000],
                        'source_dataset': 'sec_10k_weak_labeled',
                        'weak_label': weak_lexicon_label(core_text[:50000]),
                        'label': weak_lexicon_label(core_text[:50000]),
                        **feats,
                    })
                    count += 1
                    print(f'[OK] {ticker} {filing_date} {acc}')
                except Exception as exc:
                    print(f'[WARN] {ticker} {acc} failed: {exc}')
                sleep_polite(args.sleep)
        except Exception as exc:
            print(f'[WARN] {ticker} failed: {exc}')

    df = pd.DataFrame(rows)
    path = save_dataframe(df, args.output_dir, 'sec_10k_weak_sentiment', ts)
    manifest = write_manifest('results', 'sec_10k_collection', {'sec_10k_csv': str(path)}, {'rows': int(len(df)), 'tickers': args.tickers, 'years': args.years}, ts)
    print(f'[DONE] Saved {len(df)} rows -> {path}')
    print(f'[DONE] Manifest -> {manifest}')


if __name__ == '__main__':
    main()
