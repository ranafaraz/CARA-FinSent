"""Deduplication & contamination utilities.

Two checks:
1. Within-source / cross-source near-duplicate detection via MinHash LSH.
   When duplicates are found, keep the row from the highest tier
   (gold > silver > synthetic).
2. Train/test contamination check: assert no test row has a near-duplicate
   in the train pool above a Jaccard threshold.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Set, Tuple

import pandas as pd
from datasketch import MinHash, MinHashLSH


TIER_RANK = {'gold': 3, 'silver': 2, 'synthetic': 1, 'bronze': 0}

_RE_TOKEN = re.compile(r"\w+")


def _shingles(text: str, k: int = 5) -> Set[str]:
    """Word-level k-shingles. Returns at least 1 shingle for non-empty text."""
    tokens = _RE_TOKEN.findall((text or '').lower())
    if len(tokens) < k:
        return {' '.join(tokens)} if tokens else set()
    return {' '.join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}


def _build_minhash(text: str, num_perm: int = 128, k: int = 5) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for sh in _shingles(text, k=k):
        m.update(sh.encode('utf-8'))
    return m


@dataclass
class DedupReport:
    n_input: int = 0
    n_kept: int = 0
    n_dropped_exact: int = 0
    n_dropped_near: int = 0
    near_dup_pairs_sample: List[Tuple[str, str]] = field(default_factory=list)


def dedup_dataframe(
    df: pd.DataFrame,
    text_col: str = 'text_clean',
    tier_col: str = 'tier',
    threshold: float = 0.85,
    num_perm: int = 128,
) -> tuple[pd.DataFrame, DedupReport]:
    """Drop exact + near-duplicate rows. Tier-aware: keep highest-tier copy.

    Stable: row order is preserved among kept rows.
    """
    rep = DedupReport(n_input=len(df))
    work = df.copy().reset_index(drop=True)

    # ── exact dedup ────────────────────────────────────────────────────────
    work['_norm'] = work[text_col].astype(str).str.lower().str.strip()
    work['_rank'] = work[tier_col].map(TIER_RANK).fillna(0).astype(int)
    work = work.sort_values(['_norm', '_rank'], ascending=[True, False])
    before = len(work)
    work = work.drop_duplicates(subset='_norm', keep='first')
    rep.n_dropped_exact = before - len(work)
    work = work.drop(columns=['_norm']).sort_index().reset_index(drop=True)

    # ── near dedup via MinHash LSH ─────────────────────────────────────────
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: dict[int, MinHash] = {}
    drop_ids: Set[int] = set()
    sample_pairs: List[Tuple[str, str]] = []

    for idx, row in work.iterrows():
        m = _build_minhash(row[text_col], num_perm=num_perm)
        # candidates already inserted with higher or equal rank
        candidates = lsh.query(m)
        kept_against = None
        for cand in candidates:
            cand_id = int(cand)
            if cand_id in drop_ids:
                continue
            cand_rank = work.at[cand_id, '_rank']
            this_rank = row['_rank']
            if this_rank > cand_rank:
                # current row is higher tier → drop the candidate
                drop_ids.add(cand_id)
            else:
                # candidate is higher (or equal earlier-added) → drop current
                kept_against = cand_id
                break
        if kept_against is not None:
            drop_ids.add(idx)
            if len(sample_pairs) < 20:
                sample_pairs.append(
                    (str(row[text_col])[:80], str(work.at[kept_against, text_col])[:80])
                )
        else:
            lsh.insert(str(idx), m)
            minhashes[idx] = m

    rep.n_dropped_near = len(drop_ids)
    rep.near_dup_pairs_sample = sample_pairs
    out = work.drop(index=list(drop_ids)).drop(columns=['_rank']).reset_index(drop=True)
    rep.n_kept = len(out)
    return out, rep


def contamination_check(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str = 'text_clean',
    threshold: float = 0.5,
    num_perm: int = 128,
) -> pd.DataFrame:
    """Return test-row indices whose near-duplicate appears in train above `threshold`.

    Returns dataframe of contaminating pairs: test_idx, train_idx, sim.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    train_minhashes: dict[int, MinHash] = {}
    for idx, row in train_df.reset_index(drop=True).iterrows():
        m = _build_minhash(row[text_col], num_perm=num_perm)
        lsh.insert(str(idx), m)
        train_minhashes[idx] = m

    hits: List[dict] = []
    for tidx, row in test_df.reset_index(drop=True).iterrows():
        mt = _build_minhash(row[text_col], num_perm=num_perm)
        for cand in lsh.query(mt):
            cand_id = int(cand)
            sim = mt.jaccard(train_minhashes[cand_id])
            hits.append({
                'test_idx': tidx,
                'train_idx': cand_id,
                'sim': float(sim),
                'test_text': str(row[text_col])[:100],
                'train_text': str(train_df.iloc[cand_id][text_col])[:100],
            })
    return pd.DataFrame(hits)
