from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List

import pandas as pd

POSITIVE_TERMS = {'gain','gains','growth','grow','grew','increase','increased','rises','rose','rise','rising','beat','beats','beating','profit','profits','profitable','surplus','strong','strength','outperform','outperformed','upgrade','upgraded','bullish','record','improved','improvement','higher','expansion','recover','recovered','recovery','positive','optimistic','accelerate'}
NEGATIVE_TERMS = {'loss','losses','decline','declined','declining','fall','fell','falling','drop','dropped','decrease','decreased','weak','weakness','miss','missed','downgrade','downgraded','bearish','risk','risks','lawsuit','probe','fraud','default','bankrupt','bankruptcy','negative','recession','slowdown','underperform','underperformed','volatile','volatility','cut','cuts'}
UP_TERMS = {'up','rise','rises','rose','higher','increase','increased','jump','jumps','gain','gains'}
DOWN_TERMS = {'down','fall','falls','fell','lower','decline','declined','drop','drops','loss','losses'}
UNCERTAINTY_TERMS = {'may','might','could','would','expected','expects','forecast','forecasted','guidance','uncertain','possibly','likely'}
NEGATION_TERMS = {'not','no','never','neither','without','less','lack','lacks','failed','fails'}

TOKEN_RE = re.compile(r"[A-Za-z$][A-Za-z0-9$'\-]*")
MONEY_RE = re.compile(r'(\$|USD|EUR|GBP)\s?\d+(?:\.\d+)?\s?(?:m|mn|million|b|bn|billion)?', re.I)
PERCENT_RE = re.compile(r'[-+]?\d+(?:\.\d+)?\s?%')
NUMBER_RE = re.compile(r'[-+]?\d+(?:\.\d+)?')
TICKER_RE = re.compile(r'\$[A-Z]{1,6}\b|\b[A-Z]{2,5}\b')


def tokenize(text: str) -> List[str]:
    return [x.lower() for x in TOKEN_RE.findall(str(text))]


@dataclass
class FinancialFeatureExtractor:
    positive_terms: set = field(default_factory=lambda: set(POSITIVE_TERMS))
    negative_terms: set = field(default_factory=lambda: set(NEGATIVE_TERMS))
    up_terms: set = field(default_factory=lambda: set(UP_TERMS))
    down_terms: set = field(default_factory=lambda: set(DOWN_TERMS))
    uncertainty_terms: set = field(default_factory=lambda: set(UNCERTAINTY_TERMS))
    negation_terms: set = field(default_factory=lambda: set(NEGATION_TERMS))

    def extract_one(self, text: str) -> dict:
        raw = str(text)
        toks = tokenize(raw)
        n = max(len(toks), 1)
        pos = sum(t in self.positive_terms for t in toks)
        neg = sum(t in self.negative_terms for t in toks)
        up = sum(t in self.up_terms for t in toks)
        down = sum(t in self.down_terms for t in toks)
        uncertain = sum(t in self.uncertainty_terms for t in toks)
        negations = sum(t in self.negation_terms for t in toks)
        money = len(MONEY_RE.findall(raw))
        perc = len(PERCENT_RE.findall(raw))
        nums = len(NUMBER_RE.findall(raw))
        tickers = len(TICKER_RE.findall(raw))
        all_caps = sum(1 for tok in raw.split() if tok.isupper() and len(tok) > 1)
        return {'char_len': len(raw), 'token_len': len(toks), 'positive_term_count': pos, 'negative_term_count': neg, 'positive_term_rate': pos / n, 'negative_term_rate': neg / n, 'lexicon_net_score': (pos - neg) / n, 'up_term_count': up, 'down_term_count': down, 'direction_net_score': (up - down) / n, 'uncertainty_count': uncertain, 'negation_count': negations, 'money_count': money, 'percent_count': perc, 'number_count': nums, 'ticker_like_count': tickers, 'all_caps_count': all_caps, 'has_numeric_signal': int(nums > 0 or money > 0 or perc > 0)}

    def transform(self, texts: Iterable[str]) -> pd.DataFrame:
        return pd.DataFrame([self.extract_one(t) for t in texts])


def weak_lexicon_label(text: str, pos_threshold: float = 0.02, neg_threshold: float = -0.02) -> str:
    score = FinancialFeatureExtractor().extract_one(text)['lexicon_net_score']
    if score >= pos_threshold:
        return 'positive'
    if score <= neg_threshold:
        return 'negative'
    return 'neutral'
