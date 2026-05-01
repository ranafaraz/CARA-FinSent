"""Collect SemEval-2017 Task 5 financial sentiment (microblogs + news headlines).

Source: https://huggingface.co/datasets/TimKoornstra/financial-tweets-sentiment
Fallback: https://huggingface.co/datasets/sismetanin/semeval2017_task5

Output: data/raw/semeval2017_task5/<ts>.csv with columns
    text, label_raw, label, source, source_uri, license, collected_at,
    content_sha256, agreement, split

Continuous SemEval scores in [-1, 1] are binarised at +/- 0.15.
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

CANDIDATE_DATASETS = [
    # (repo, config, text_field, score_field)
    ('TimKoornstra/financial-tweets-sentiment', None, 'tweet', 'sentiment'),
    ('sismetanin/semeval2017_task5_subtask_a', None, 'text', 'sentiment'),
    ('sismetanin/semeval2017_task5_subtask_b', None, 'text', 'sentiment'),
]


def _sha(text: str) -> str:
    return hashlib.sha256((text or '').encode('utf-8', errors='replace')).hexdigest()[:16]


def _binarise(score) -> tuple[str | None, bool]:
    """Binarize continuous sentiment score into {negative, neutral, positive}.
    
    Returns (label, is_uncertain) where is_uncertain=True if |score| < 0.25
    (ambiguous region where the signal is weak).
    """
    try:
        s = float(score)
    except (TypeError, ValueError):
        return None, False
    
    is_uncertain = abs(s) < 0.25  # Flag borderline cases
    
    if s > 0.15:
        return 'positive', is_uncertain
    if s < -0.15:
        return 'negative', is_uncertain
    return 'neutral', is_uncertain


def collect() -> pd.DataFrame:
    from datasets import load_dataset
    rows: list[dict] = []
    now = datetime.now(timezone.utc).isoformat()

    for repo, config, text_field, score_field in CANDIDATE_DATASETS:
        try:
            print(f'[TRY] {repo} (config={config})')
            ds = load_dataset(repo, config) if config else load_dataset(repo)
        except Exception as exc:
            print(f'  [SKIP] {repo}: {exc}')
            continue
        for split_name, split_data in ds.items():
            for item in split_data:
                txt = item.get(text_field) or item.get('text') or item.get('sentence')
                if not txt:
                    continue
                raw = item.get(score_field, item.get('score', item.get('sentiment', None)))
                label, is_uncertain = _binarise(raw)
                if label is None and isinstance(raw, str):
                    s = raw.strip().lower()
                    if s in {'positive', 'pos', 'bullish'}:
                        label = 'positive'
                        is_uncertain = False
                    elif s in {'negative', 'neg', 'bearish'}:
                        label = 'negative'
                        is_uncertain = False
                    elif s in {'neutral', 'neu'}:
                        label = 'neutral'
                        is_uncertain = False
                if label is None:
                    continue
                rows.append({
                    'text': str(txt),
                    'label_raw': raw,
                    'label': label,
                    'source': 'semeval2017_task5',
                    'source_uri': f'hf://{repo}',
                    'license': 'CC-BY-4.0',
                    'collected_at': now,
                    'content_sha256': _sha(txt),
                    'agreement': pd.NA,
                    'is_uncertain': is_uncertain,  # Flag borderline cases
                    'split': str(split_name),
                })
        print(f'  [OK] {repo}: cumulative rows={len(rows)}')
        if len(rows) >= 1500:
            break

    return pd.DataFrame(rows)


def main() -> None:
    df = collect()
    if df.empty:
        print('[ERROR] No SemEval rows collected from any source.')
        sys.exit(1)
    out_dir = PROJECT_ROOT / 'data' / 'raw' / 'semeval2017_task5'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = timestamped_path(out_dir, 'semeval2017_task5')
    # write directly to the source-scoped folder to avoid double date dirs
    final = out_dir / out_path.name
    df.to_csv(final, index=False)
    print(f'[DONE] Saved {len(df)} rows -> {final}')
    print('Label dist:', df['label'].value_counts().to_dict())


if __name__ == '__main__':
    main()
