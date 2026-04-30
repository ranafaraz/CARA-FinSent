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


def augment_with_context(texts: Iterable[str], retriever: TfidfRetriever, top_k: int = 3, separator: str = ' [CTX] ') -> tuple[list[str], pd.DataFrame]:
    texts = [str(x) for x in texts]
    retrieved = retriever.retrieve(texts, top_k=top_k)
    augmented = []
    rows = []
    for qid, (text, hits) in enumerate(zip(texts, retrieved)):
        ctx = ' '.join([h[2] for h in hits])
        augmented.append(text + separator + ctx if ctx else text)
        for rank, (idx, score, ctx_text) in enumerate(hits, start=1):
            rows.append({'query_id': qid, 'rank': rank, 'corpus_index': idx, 'score': score, 'query_text': text, 'context_text': ctx_text})
    return augmented, pd.DataFrame(rows)
