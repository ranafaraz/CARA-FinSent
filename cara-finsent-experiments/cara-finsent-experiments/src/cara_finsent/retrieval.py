from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class TfidfRetriever:
    max_features: int = 50000
    ngram_range: tuple = (1, 2)
    min_df: int = 1
    vectorizer: TfidfVectorizer | None = None
    corpus_texts: List[str] | None = None
    matrix: object | None = None

    def fit(self, corpus_texts: Iterable[str]):
        self.corpus_texts = [str(x) for x in corpus_texts]
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, ngram_range=self.ngram_range, min_df=self.min_df, stop_words='english')
        self.matrix = self.vectorizer.fit_transform(self.corpus_texts)
        return self

    def retrieve(self, queries: Iterable[str], top_k: int = 3) -> List[List[Tuple[int, float, str]]]:
        if self.vectorizer is None or self.matrix is None or self.corpus_texts is None:
            raise RuntimeError('Retriever must be fitted before retrieve().')
        q_mat = self.vectorizer.transform([str(q) for q in queries])
        sim = cosine_similarity(q_mat, self.matrix)
        results = []
        for row in sim:
            idx = np.argsort(-row)[:top_k]
            results.append([(int(i), float(row[i]), self.corpus_texts[int(i)]) for i in idx])
        return results


def augment_with_context(texts: Iterable[str], retriever: TfidfRetriever, top_k: int = 3, separator: str = ' ctxstart ', terminator: str = ' ctxend ', drop_self_match: bool = True) -> tuple[list[str], pd.DataFrame]:
    """Augment each input text with retrieved contexts.

    The default sentinel tokens ``ctxstart`` / ``ctxend`` are alphabetic so they
    survive the default ``TfidfVectorizer`` tokenizer (the previous ``[CTX]``
    separator was stripped because of bracket characters and stop-word handling).
    When ``drop_self_match`` is True, retrieved contexts whose text exactly equals
    the query string are skipped (prevents trivial label leakage in the
    self-retrieval setting where the query also appears in the corpus).
    """
    texts = [str(x) for x in texts]
    # Over-fetch by 1 so we still have top_k results after dropping self-matches.
    fetch_k = top_k + 1 if drop_self_match else top_k
    retrieved = retriever.retrieve(texts, top_k=fetch_k)
    augmented = []
    rows = []
    for qid, (text, hits) in enumerate(zip(texts, retrieved)):
        kept = []
        for idx, score, ctx_text in hits:
            if drop_self_match and ctx_text.strip() == text.strip():
                continue
            kept.append((idx, score, ctx_text))
            if len(kept) >= top_k:
                break
        ctx_blob = ' '.join([h[2] for h in kept])
        augmented.append(f'{text}{separator}{ctx_blob}{terminator}' if ctx_blob else text)
        for rank, (idx, score, ctx_text) in enumerate(kept, start=1):
            rows.append({'query_id': qid, 'rank': rank, 'corpus_index': idx, 'score': score, 'query_text': text, 'context_text': ctx_text})
    return augmented, pd.DataFrame(rows)
