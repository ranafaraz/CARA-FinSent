from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

STANDARD_LABELS = ['negative', 'neutral', 'positive']


def set_global_seeds(seed: int = 42, enable_deep_learning: bool = False) -> int:
    """Pin Python, NumPy, and optionally deep-learning seeds.

    Safe to call from any script. Returns the seed for logging convenience.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if enable_deep_learning:
        try:
            import torch  # type: ignore
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
        try:
            from transformers import set_seed as _hf_set_seed  # type: ignore
            _hf_set_seed(seed)
        except Exception:
            pass
    elif 'torch' in sys.modules:
        # If torch is already loaded, seed it without forcing a heavyweight import.
        try:
            torch = sys.modules['torch']
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
    return int(seed)


def find_latest_csv(directory: str | Path, pattern: str) -> Optional[Path]:
    """Return the most-recently modified CSV matching glob `pattern` anywhere under `directory`.

    Uses ``rglob`` so it descends into date-stamped subfolders created by
    ``io_utils.timestamped_path``.
    """
    directory = Path(directory)
    if not directory.exists():
        return None
    matches = sorted(directory.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def auto_detect_data(base_dir: str | Path = 'data/processed') -> Path:
    """Auto-detect the latest prepared dataset CSV without user input.

    Priority
    --------
    1. ``data/processed/latest.csv``  — written by script 00 on every run.
    2. Latest ``combined_standardized_*.csv`` anywhere under *base_dir*.

    Raises ``SystemExit`` with a clear actionable message if nothing is found.
    """
    base = Path(base_dir)
    latest = base / 'latest.csv'
    if latest.exists():
        return latest
    if base.exists():
        matches = sorted(base.rglob('combined_standardized_*.csv'),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]
    raise SystemExit(
        f'No prepared dataset found in {base_dir}/. '
        'Run `make prepare` first to download Financial PhraseBank + FiQA from Hugging Face.'
    )


def auto_detect_gold_split(dataset_name: str, base_dir: str | Path = 'data/processed/gold') -> Path:
    """Return the latest controlled gold split for a supported dataset."""
    normalized = str(dataset_name).strip().lower()
    if normalized not in {'phrasebank', 'fiqa'}:
        raise SystemExit(f'Unsupported dataset_name={dataset_name!r}. Use one of: phrasebank, fiqa.')
    path = Path(base_dir) / f'latest_gold_{normalized}_split.csv'
    if path.exists():
        return path
    raise SystemExit(
        f'Missing gold split for {normalized}. '
        'Run scripts/06_create_controlled_splits.py first.'
    )


def infer_dataset_name(path: str | Path | None) -> str:
    """Infer dataset name from a path when possible."""
    lowered = str(path or '').lower()
    for name in ('phrasebank', 'fiqa'):
        if name in lowered:
            return name
    return 'unknown'


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


def text_hash_leakage_count(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> int:
    """Count overlapping text_hash values across train/val/test splits."""
    train_hashes = set(train_df['text_hash'].astype(str))
    val_hashes = set(val_df['text_hash'].astype(str))
    test_hashes = set(test_df['text_hash'].astype(str))
    return int(
        len(train_hashes & val_hashes)
        + len(train_hashes & test_hashes)
        + len(val_hashes & test_hashes)
    )


def split_label_distribution(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Return canonical label counts per split."""
    def _dist(frame: pd.DataFrame) -> dict[str, int]:
        counts = frame['label'].value_counts().to_dict()
        return {label: int(counts.get(label, 0)) for label in STANDARD_LABELS}

    return {
        'train': _dist(train_df),
        'val': _dist(val_df),
        'test': _dist(test_df),
    }


def load_gold_split(path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a controlled gold split file and validate it strictly."""
    source = Path(path)
    if not source.exists():
        raise SystemExit(f'Gold split file not found: {source}')

    df = pd.read_csv(source)
    required = {'text', 'label', 'split', 'text_hash'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing required gold split columns: {sorted(missing)}')

    out = df.copy()
    out['split'] = out['split'].astype(str).str.strip().str.lower()
    invalid_splits = sorted(set(out['split']) - {'train', 'val', 'test'})
    if invalid_splits:
        raise ValueError(f'Invalid split values in {source}: {invalid_splits}')

    out['label'] = out['label'].astype(str).str.strip().str.lower()
    invalid_labels = sorted(set(out['label']) - set(STANDARD_LABELS))
    if invalid_labels:
        raise ValueError(f'Invalid labels in {source}: {invalid_labels}')

    missing_hashes = int(out['text_hash'].isna().sum() + out['text_hash'].astype(str).str.strip().eq('').sum())
    if missing_hashes:
        raise ValueError(f'{source} has {missing_hashes} missing text_hash values.')

    train_df = out[out['split'].eq('train')].reset_index(drop=True)
    val_df = out[out['split'].eq('val')].reset_index(drop=True)
    test_df = out[out['split'].eq('test')].reset_index(drop=True)
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(
            f'Gold split file must contain non-empty train/val/test partitions. '
            f'Got train={len(train_df)}, val={len(val_df)}, test={len(test_df)}.'
        )

    leakage_count = text_hash_leakage_count(train_df, val_df, test_df)
    if leakage_count:
        raise ValueError(f'Gold split leakage detected in {source}: text_hash_leakage_count={leakage_count}')

    return train_df, val_df, test_df


def encode_labels(labels):
    mapping = {label: i for i, label in enumerate(STANDARD_LABELS)}
    return np.array([mapping[x] for x in labels])


def decode_labels(ids):
    return [STANDARD_LABELS[int(i)] for i in ids]
