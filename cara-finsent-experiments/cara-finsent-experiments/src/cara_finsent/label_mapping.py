"""Safe label mapping and text-hash utilities for the CARA-FinSent data phase.

Phase 1 rule: never assume a model's label order. Always read
``model.config.id2label`` and remap probabilities/predictions to the canonical
``['negative', 'neutral', 'positive']`` order used throughout this codebase.
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, Iterable, List, Tuple

import numpy as np

CANONICAL_LABELS: List[str] = ['negative', 'neutral', 'positive']
VALID_LABELS = set(CANONICAL_LABELS)
CANONICAL_LABEL_TO_ID: Dict[str, int] = {lbl: i for i, lbl in enumerate(CANONICAL_LABELS)}

_POSITIVE_ALIASES = {'positive', 'pos', 'bullish', 'buy', '2', 'label_2'}
_NEGATIVE_ALIASES = {'negative', 'neg', 'bearish', 'sell', '-1', 'label_0'}
_NEUTRAL_ALIASES = {'neutral', 'neu', 'hold', '0', '1', 'label_1'}


def canonical_label(value) -> str | None:
    """Return one of ``CANONICAL_LABELS`` or ``None`` if value is unrecognised."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in _POSITIVE_ALIASES or 'positive' in s or 'bullish' in s:
        return 'positive'
    if s in _NEGATIVE_ALIASES or 'negative' in s or 'bearish' in s:
        return 'negative'
    if s in _NEUTRAL_ALIASES or 'neutral' in s:
        return 'neutral'
    return None


_WS_RE = re.compile(r'\s+')


def normalize_text_for_hash(text: str) -> str:
    """Lowercase, collapse whitespace, strip — used only for duplicate/leakage detection."""
    if text is None:
        return ''
    return _WS_RE.sub(' ', str(text).lower()).strip()


def text_hash(text: str) -> str:
    """SHA-256 hex digest of normalized text."""
    return hashlib.sha256(normalize_text_for_hash(text).encode('utf-8')).hexdigest()


def model_label_remap(id2label: Dict[int, str]) -> List[int]:
    """Return a column index list that reorders model output columns to canonical order.

    Example: HF ``ProsusAI/finbert`` ships with id2label = {0:positive, 1:negative, 2:neutral}.
    Calling ``probs[:, model_label_remap(config.id2label)]`` produces columns ordered
    as ``[negative, neutral, positive]``.
    """
    name_to_idx: Dict[str, int] = {}
    for idx, name in id2label.items():
        c = canonical_label(name)
        if c is None:
            raise ValueError(f"Cannot map model label {name!r} (id={idx}) to canonical labels.")
        if c in name_to_idx:
            raise ValueError(f"Duplicate canonical label {c!r} from model labels {id2label!r}.")
        name_to_idx[c] = int(idx)
    missing = [lbl for lbl in CANONICAL_LABELS if lbl not in name_to_idx]
    if missing:
        raise ValueError(f"Model id2label is missing canonical labels: {missing}. Got: {id2label!r}")
    return [name_to_idx[lbl] for lbl in CANONICAL_LABELS]


def remap_probs(probs: np.ndarray, id2label: Dict[int, str]) -> np.ndarray:
    """Reorder a (N, 3) probability array from a model's native label order to canonical."""
    cols = model_label_remap(id2label)
    return probs[:, cols]


def validate_labels(labels: Iterable) -> Tuple[int, List[str]]:
    """Return (invalid_count, list_of_invalid_values)."""
    bad: List[str] = []
    for v in labels:
        if canonical_label(v) is None:
            bad.append(str(v))
    return len(bad), bad
