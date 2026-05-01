#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import argparse
import os
from datetime import datetime
from urllib.parse import urlencode

import pandas as pd
import requests

try:
    import feedparser
except ImportError:
    feedparser = None

from cara_finsent.feature_extractor import FinancialFeatureExtractor, weak_lexicon_label
from cara_finsent.io_utils import save_dataframe, save_skip_report, timestamp, write_manifest
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


DEFAULT_RSS = [
    'https://feeds.marketwatch.com/marketwatch/topstories/',
    'https://feeds.a.dj.com/rss/RSSMarketsMain.xml',
    'https://www.cnbc.com/id/100003114/device/rss/rss.html',
    'https://www.investing.com/rss/news_25.rss',
]


def collect_rss(urls):
    rows = []
    skipped = []
    if feedparser is None:
        for url in urls:
            skipped.append({'source': 'rss', 'endpoint': url, 'reason': 'feedparser package is not installed', 'action': 'skipped'})
        print('[WARN] RSS collection skipped: feedparser package is not installed.')
        return rows, skipped
    for url in urls:
        try:
            feed = feedparser.parse(url)
            source = feed.feed.get('title', url)
            for entry in feed.entries:
                title = entry.get('title', '')
                summary = entry.get('summary', '')
                text = f'{title}. {summary}'.strip()
                rows.append({
                    'id': entry.get('id') or entry.get('link') or title,
                    'source_dataset': 'financial_news_rss',
                    'source': source,
                    'published_at': entry.get('published', entry.get('updated', '')),
                    'title': title,
                    'summary': summary,
                    'text': text,
                    'source_url': entry.get('link', ''),
                })
            print(f'[OK] RSS {url}: {len(feed.entries)} entries')
        except Exception as exc:
            print(f'[WARN] RSS {url} failed: {exc}')
            skipped.append({'source': 'rss', 'endpoint': url, 'reason': str(exc), 'action': 'skipped'})
    return rows, skipped


def collect_newsapi(query: str, page_size: int = 100):
    key = os.getenv('NEWSAPI_KEY')
    if not key:
        return [], [{'source': 'newsapi', 'reason': 'NEWSAPI_KEY not set', 'action': 'skipped'}]
    params = {'q': query, 'language': 'en', 'sortBy': 'publishedAt', 'pageSize': page_size, 'apiKey': key}
    url = 'https://newsapi.org/v2/everything?' + urlencode(params)
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        rows = []
        for a in data.get('articles', []):
            title = a.get('title') or ''
            desc = a.get('description') or ''
            rows.append({'id': a.get('url'), 'source_dataset': 'newsapi', 'source': (a.get('source') or {}).get('name'), 'published_at': a.get('publishedAt'), 'title': title, 'summary': desc, 'text': f'{title}. {desc}', 'source_url': a.get('url')})
        print(f'[OK] NewsAPI: {len(rows)} entries')
        return rows, []
    except Exception as exc:
        print(f'[WARN] NewsAPI failed: {exc}')
        return [], [{'source': 'newsapi', 'reason': str(exc), 'action': 'skipped'}]


def collect_finnhub(symbols):
    key = os.getenv('FINNHUB_API_KEY')
    if not key:
        return [], [{'source': 'finnhub', 'reason': 'FINNHUB_API_KEY not set', 'action': 'skipped'}]
    rows = []
    skipped = []
    today = datetime.utcnow().date().isoformat()
    # Uses latest company news for a short window. Adjust manually for broader collection.
    for symbol in symbols:
        url = f'https://finnhub.io/api/v1/company-news?symbol={symbol}&from=2024-01-01&to={today}&token={key}'
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            for a in r.json()[:200]:
                title = a.get('headline') or ''
                summary = a.get('summary') or ''
                rows.append({'id': a.get('id'), 'source_dataset': 'finnhub_company_news', 'source': a.get('source'), 'ticker': symbol, 'published_at': a.get('datetime'), 'title': title, 'summary': summary, 'text': f'{title}. {summary}', 'source_url': a.get('url')})
            print(f'[OK] Finnhub {symbol}')
        except Exception as exc:
            print(f'[WARN] Finnhub {symbol} failed: {exc}')
            skipped.append({'source': 'finnhub', 'symbol': symbol, 'reason': str(exc), 'action': 'skipped'})
    return rows, skipped


def main():
    parser = argparse.ArgumentParser(description='Collect financial news headlines from RSS and optional APIs.')
    parser.add_argument('--rss_urls', nargs='*', default=DEFAULT_RSS)
    parser.add_argument('--query', default='stocks OR earnings OR markets OR finance')
    parser.add_argument('--symbols', nargs='*', default=['AAPL', 'MSFT', 'TSLA', 'NVDA'])
    parser.add_argument('--include_newsapi', action='store_true')
    parser.add_argument('--include_finnhub', action='store_true')
    parser.add_argument('--weak_label', action='store_true', help='Add weak lexicon labels for exploratory training only.')
    parser.add_argument('--output_dir', type=str, default='data/external')
    args = parser.parse_args()

    ts = timestamp()
    rows, skipped_sources = collect_rss(args.rss_urls)
    if args.include_newsapi:
        newsapi_rows, newsapi_skipped = collect_newsapi(args.query)
        rows += newsapi_rows
        skipped_sources += newsapi_skipped
    if args.include_finnhub:
        finnhub_rows, finnhub_skipped = collect_finnhub(args.symbols)
        rows += finnhub_rows
        skipped_sources += finnhub_skipped

    df = pd.DataFrame(rows).drop_duplicates(subset=['source_url', 'title']) if rows else pd.DataFrame()
    if args.weak_label and len(df):
        extractor = FinancialFeatureExtractor()
        feats = extractor.transform(df['text'])
        df = pd.concat([df.reset_index(drop=True), feats], axis=1)
        df['weak_label'] = df['text'].apply(weak_lexicon_label)
        df['label'] = df['weak_label']

    path = save_dataframe(df, args.output_dir, 'financial_news_headlines', ts)
    skip_report_path = save_skip_report(skipped_sources, 'results', 'financial_news_skipped_sources', ts)
    files = {'financial_news_csv': str(path)}
    if skip_report_path is not None:
        files['skip_report_csv'] = str(skip_report_path)
    manifest = write_manifest(
        'results',
        'financial_news_collection',
        files,
        {
            'rows': int(len(df)),
            'weak_label': args.weak_label,
            'skipped_sources_count': len(skipped_sources),
        },
        ts,
    )
    print(f'[DONE] Saved {len(df)} rows -> {path}')
    print(f'[DONE] Manifest -> {manifest}')


if __name__ == '__main__':
    main()
