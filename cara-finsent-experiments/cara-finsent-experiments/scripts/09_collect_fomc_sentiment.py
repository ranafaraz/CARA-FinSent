"""Collect FOMC (Federal Open Market Committee) sentiment-labelled statements.

Tries several public Hugging Face datasets that label FOMC communication
hawkish/dovish/neutral. Maps to {negative=hawkish, positive=dovish,
neutral=neutral} which matches the conventional finance sentiment direction
(dovish = market-positive, hawkish = market-negative).

Output: data/raw/fomc_sentiment/<ts>.csv
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import pandas as pd
from cara_finsent.io_utils import timestamped_path

CANDIDATES = [
    ('gtfintechlab/fomc_communication', None, 'sentence', 'label'),
    ('gtfintechlab/fomc_communication', 'meeting_minutes', 'sentence', 'label'),
    ('gtfintechlab/fomc_communication', 'press_conference', 'sentence', 'label'),
    ('gtfintechlab/fomc_communication', 'speeches', 'sentence', 'label'),
]

LABEL_MAP = {
    0: 'negative',  # hawkish (interest rate hike → bearish for stocks)
    1: 'positive',  # dovish  (interest rate cut → bullish for stocks)
    2: 'neutral',
    'hawkish': 'negative',
    'dovish': 'positive',
    'neutral': 'neutral',
}


def _sha(text: str) -> str:
    return hashlib.sha256((text or '').encode('utf-8', errors='replace')).hexdigest()[:16]


def collect() -> pd.DataFrame:
    from datasets import load_dataset
    rows: list[dict] = []
    now = datetime.now(timezone.utc).isoformat()

    for repo, config, text_field, label_field in CANDIDATES:
        try:
            print(f'[TRY] {repo} (config={config})')
            ds = (load_dataset(repo, config, trust_remote_code=True)
                  if config else load_dataset(repo, trust_remote_code=True))
        except Exception as exc:
            print(f'  [SKIP] {repo}: {exc}')
            continue
        for split_name, split_data in ds.items():
            for item in split_data:
                txt = item.get(text_field) or item.get('text') or item.get('sentence')
                lab_raw = item.get(label_field)
                if not txt or lab_raw is None:
                    continue
                key = lab_raw if isinstance(lab_raw, int) else str(lab_raw).strip().lower()
                label = LABEL_MAP.get(key)
                if label is None:
                    continue
                rows.append({
                    'text': str(txt),
                    'label_raw': lab_raw,
                    'label': label,
                    'source': 'fomc_sentiment',
                    'source_uri': f'hf://{repo}/{config or ""}',
                    'license': 'CC-BY-4.0',
                    'collected_at': now,
                    'content_sha256': _sha(txt),
                    'agreement': pd.NA,
                    'split': str(split_name),
                })
        print(f'  [OK] {repo}/{config}: cumulative rows={len(rows)}')

    return pd.DataFrame(rows)


def main() -> None:
    df = collect()
    if df.empty:
        print('[WARN] No FOMC rows collected — repo may be gated. Skipping.')
        sys.exit(0)
    out_dir = PROJECT_ROOT / 'data' / 'raw' / 'fomc_sentiment'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = timestamped_path(out_dir, 'fomc_sentiment')
    final = out_dir / out_path.name
    df.to_csv(final, index=False)
    print(f'[DONE] Saved {len(df)} rows -> {final}')
    print('Label dist:', df['label'].value_counts().to_dict())


if __name__ == '__main__':
    main()
