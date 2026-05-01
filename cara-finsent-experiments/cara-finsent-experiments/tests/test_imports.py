"""Functional smoke tests for the cara_finsent library."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


def test_imports():
    import cara_finsent  # noqa: F401
    from cara_finsent.feature_extractor import FinancialFeatureExtractor
    assert FinancialFeatureExtractor().extract_one('Revenue rose 20%')['has_numeric_signal'] == 1


def test_normalize_label_round_trip():
    from cara_finsent.data_utils import STANDARD_LABELS, normalize_label
    for canonical in STANDARD_LABELS:
        assert normalize_label(canonical) == canonical
    assert normalize_label('POS') == 'positive'
    assert normalize_label('Bearish') == 'negative'
    assert normalize_label(0) == 'negative'
    assert normalize_label(2) == 'positive'
    assert normalize_label(0.5) == 'positive'
    assert normalize_label(-0.5) == 'negative'
    assert normalize_label(0.0) == 'neutral'


def test_split_dataframe_stratified():
    from cara_finsent.data_utils import split_dataframe
    labels = ['negative'] * 60 + ['neutral'] * 60 + ['positive'] * 60
    df = pd.DataFrame({'text': [f't{i}' for i in range(180)], 'label': labels})
    df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    train, val, test = split_dataframe(df, test_size=0.2, val_size=0.1, seed=42)
    for part in (train, val, test):
        assert set(part['label'].unique()) == {'negative', 'neutral', 'positive'}
    assert len(train) + len(val) + len(test) == len(df)


def test_set_global_seeds_is_deterministic():
    from cara_finsent.data_utils import set_global_seeds
    set_global_seeds(123)
    a = np.random.rand(5)
    set_global_seeds(123)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_expected_calibration_error_perfect_zero():
    from cara_finsent.metrics import expected_calibration_error
    y_true = np.array(['positive', 'negative', 'neutral'])
    y_pred = y_true.copy()
    proba = np.eye(3)[[2, 0, 1]]
    assert expected_calibration_error(y_true, y_pred, proba) == 0.0


def test_tfidf_retriever_topk():
    from cara_finsent.retrieval import TfidfRetriever, augment_with_context
    corpus = [
        'apple reported strong quarterly earnings',
        'tesla shares fell after weak guidance',
        'microsoft revenue exceeded expectations',
        'oil prices declined on demand worries',
    ]
    r = TfidfRetriever(min_df=1).fit(corpus)
    hits = r.retrieve(['apple earnings'], top_k=2)
    assert len(hits) == 1 and len(hits[0]) == 2
    augmented, ctx_df = augment_with_context(['apple earnings'], r, top_k=2)
    assert 'ctxstart' in augmented[0] and 'ctxend' in augmented[0]
    assert len(ctx_df) == 2
