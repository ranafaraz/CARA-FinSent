"""Quality tests for the canonical CARA-FinSent dataset.

Run with:
    pytest tests/test_dataset_quality.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

DATASET_PATH = PROJECT_ROOT / 'data' / 'processed' / 'latest' / 'dataset.csv'

STANDARD_LABELS = {'negative', 'neutral', 'positive'}


@pytest.fixture(scope='module')
def df() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        pytest.skip(f'canonical dataset not built yet: {DATASET_PATH}')
    return pd.read_csv(DATASET_PATH)


def test_dataset_exists(df):
    assert len(df) > 0


def test_required_columns_present(df):
    required = {'text', 'text_clean', 'text_clean_lower', 'label',
                'source', 'tier', 'split', 'n_tokens', 'qc_flags'}
    missing = required - set(df.columns)
    assert not missing, f'missing required columns: {missing}'


def test_label_values_normalised(df):
    bad = set(df['label'].dropna().unique()) - STANDARD_LABELS
    assert not bad, f'unexpected label values: {bad}'


def test_no_empty_text_clean(df):
    n_empty = df['text_clean'].astype(str).str.len().eq(0).sum()
    assert n_empty == 0, f'{n_empty} rows have empty text_clean'


def test_token_length_in_range(df):
    too_short = (df['n_tokens'] < 3).sum()
    too_long = (df['n_tokens'] > 512).sum()
    assert too_short == 0, f'{too_short} rows < 3 tokens'
    assert too_long == 0, f'{too_long} rows > 512 tokens'


def test_no_url_or_at_user_or_cashtag_in_text_clean(df):
    sample = df.sample(min(2000, len(df)), random_state=0)
    has_url = sample['text_clean'].str.contains(r'https?://', regex=True).sum()
    has_user = sample['text_clean'].str.contains(r'(?<!\[)@\w', regex=True).sum()
    has_cashtag = sample['text_clean'].str.contains(r'\$[A-Za-z]', regex=True).sum()
    assert has_url == 0, f'{has_url} rows still contain raw URLs'
    assert has_user == 0, f'{has_user} rows still contain raw @user'
    assert has_cashtag == 0, f'{has_cashtag} rows still contain raw $TICKER'


def test_test_set_is_substantial(df):
    n_test = (df['split'] == 'test').sum()
    assert n_test >= 500, f'test set too small: {n_test} rows (need ≥500)'


def test_test_set_has_all_three_labels(df):
    test = df[df['split'] == 'test']
    labs = set(test['label'].unique())
    assert labs == STANDARD_LABELS, f'test labels missing: {STANDARD_LABELS - labs}'


def test_test_set_minority_class_size(df):
    test = df[df['split'] == 'test']
    counts = test['label'].value_counts()
    assert counts.min() >= 50, f'minority class in test < 50: {counts.to_dict()}'


def test_train_imbalance_within_threshold(df):
    train = df[df['split'] == 'train']
    counts = train['label'].value_counts()
    if counts.empty:
        pytest.skip('no train rows')
    ratio = counts.max() / counts.min()
    assert ratio <= 2.5, f'train imbalance ratio {ratio:.2f} > 2.5'


def test_no_train_test_text_overlap(df):
    train_texts = set(df[df['split'] == 'train']['text_clean'].astype(str))
    test_texts = set(df[df['split'] == 'test']['text_clean'].astype(str))
    overlap = train_texts & test_texts
    assert not overlap, f'{len(overlap)} text_clean values appear in both train and test'


def test_tier_values_valid(df):
    valid = {'gold', 'silver', 'synthetic', 'bronze'}
    bad = set(df['tier'].dropna().unique()) - valid
    assert not bad, f'unexpected tier values: {bad}'


def test_at_least_one_gold_test_per_source(df):
    test = df[df['split'] == 'test']
    sources_with_test = set(test['source'].dropna().unique())
    assert len(sources_with_test) >= 2, (
        f'test set has too few source diversity: {sources_with_test}'
    )


def test_within_source_no_exact_dupes(df):
    dups = df.groupby('source')['text_clean'].apply(
        lambda s: s.duplicated().sum())
    bad = dups[dups > 0]
    assert bad.empty, f'within-source duplicates found: {bad.to_dict()}'
