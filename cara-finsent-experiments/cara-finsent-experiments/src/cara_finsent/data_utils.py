from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

STANDARD_LABELS = ['negative', 'neutral', 'positive']
TEXT_CANDIDATES = ['text', 'sentence', 'Sentence', 'headline', 'title', 'body', 'content', 'summary']
LABEL_CANDIDATES = ['label', 'sentiment', 'Sentiment', 'target', 'class', 'weak_label']


def normalize_label(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, np.integer)):
        mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
        return mapping.get(int(value), str(value).lower())
    if isinstance(value, float):
        if value > 0.1:
            return 'positive'
        if value < -0.1:
            return 'negative'
        return 'neutral'
    s = str(value).strip().lower()
    if s in {'pos', 'positive', 'bullish', 'buy', '2'}:
        return 'positive'
    if s in {'neg', 'negative', 'bearish', 'sell', '-1'}:
        return 'negative'
    if s in {'neu', 'neutral', 'hold', '0', '0.0', 'nan', '1'}:
        return 'neutral'
    if 'positive' in s or 'bullish' in s:
        return 'positive'
    if 'negative' in s or 'bearish' in s:
        return 'negative'
    if 'neutral' in s:
        return 'neutral'
    return s


def detect_column(df: pd.DataFrame, candidates, explicit: Optional[str] = None) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f'Column {explicit!r} not found. Available columns: {list(df.columns)}')
        return explicit
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f'Could not detect column from candidates {candidates}. Available columns: {list(df.columns)}')


def load_standardized_csv(path: str | Path, text_col: Optional[str] = None, label_col: Optional[str] = None, allow_unlabeled: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = detect_column(df, TEXT_CANDIDATES, text_col)
    out = df.copy()
    out['text'] = out[tcol].astype(str).fillna('')
    if label_col or any(c in out.columns for c in LABEL_CANDIDATES):
        lcol = detect_column(out, LABEL_CANDIDATES, label_col)
        out['label'] = out[lcol].apply(normalize_label)
    elif not allow_unlabeled:
        raise ValueError('No label column found. Use allow_unlabeled=True for raw corpora.')
    else:
        out['label'] = np.nan
    out = out[out['text'].str.len() > 0].copy()
    if not allow_unlabeled:
        out = out[out['label'].isin(STANDARD_LABELS)].copy()
    out.reset_index(drop=True, inplace=True)
    if 'id' not in out.columns:
        out['id'] = [f'row_{i}' for i in range(len(out))]
    return out


def apply_max_rows(df: pd.DataFrame, max_rows: Optional[int], seed: int = 42) -> pd.DataFrame:
    if max_rows and max_rows > 0 and len(df) > max_rows:
        if 'label' in df.columns and df['label'].notna().all() and df['label'].nunique() > 1:
            parts = []
            for _, group in df.groupby('label'):
                n = max(1, int(max_rows * len(group) / len(df)))
                parts.append(group.sample(min(n, len(group)), random_state=seed))
            return pd.concat(parts).sample(frac=1.0, random_state=seed).head(max_rows).reset_index(drop=True)
        return df.sample(max_rows, random_state=seed).reset_index(drop=True)
    return df.reset_index(drop=True)


def split_dataframe(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if 'split' in df.columns:
        split_values = df['split'].astype(str).str.lower()
        train_df = df[split_values.eq('train')].copy()
        val_df = df[split_values.isin(['val', 'valid', 'validation'])].copy()
        test_df = df[split_values.eq('test')].copy()
        if len(train_df) and len(test_df):
            if len(val_df) == 0:
                train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=seed, stratify=train_df['label'] if train_df['label'].nunique() > 1 else None)
            return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
    stratify = df['label'] if df['label'].nunique() > 1 else None
    try:
        train_val, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify)
    except ValueError:
        train_val, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=None)
    relative_val = val_size / (1.0 - test_size)
    stratify_val = train_val['label'] if train_val['label'].nunique() > 1 else None
    try:
        train_df, val_df = train_test_split(train_val, test_size=relative_val, random_state=seed, stratify=stratify_val)
    except ValueError:
        train_df, val_df = train_test_split(train_val, test_size=relative_val, random_state=seed, stratify=None)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def encode_labels(labels):
    mapping = {label: i for i, label in enumerate(STANDARD_LABELS)}
    return np.array([mapping[x] for x in labels])


def decode_labels(ids):
    return [STANDARD_LABELS[int(i)] for i in ids]
