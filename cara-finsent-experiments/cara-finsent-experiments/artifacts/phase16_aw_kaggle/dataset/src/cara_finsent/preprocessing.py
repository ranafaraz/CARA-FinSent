"""Universal text preprocessing for financial sentiment data.

Cleans text for two parallel pipelines:
- transformer (FinBERT): case-preserved, normalised, masked entities
- classical (TF-IDF):    same as above but additionally lowercased

The cleaners are deterministic and idempotent.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import ftfy
import emoji
import pandas as pd

try:
    from langdetect import detect, DetectorFactory, LangDetectException
    DetectorFactory.seed = 42
except Exception:  # pragma: no cover - langdetect is required
    detect = None
    LangDetectException = Exception


# ─── regex patterns ────────────────────────────────────────────────────────
_RE_URL = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
_RE_USER = re.compile(r'@[A-Za-z0-9_]{1,30}')
# Match common cashtag forms like $AAPL, $SPX500, $ES1, $safegalaxy.
# We require a leading letter to avoid prices like $100.
_RE_CASHTAG = re.compile(r'\$([A-Za-z][A-Za-z0-9]{0,14})')
_RE_HTML = re.compile(r'<[^>]+>')
_RE_WS = re.compile(r'\s+')
_RE_CTRL = re.compile(r'[\u0000-\u001f\u007f-\u009f\u200b-\u200f\ufeff]')


def _extract_tickers(text: str) -> List[str]:
    return list({m.group(1).upper() for m in _RE_CASHTAG.finditer(text or '')})


def clean_text(text: str) -> str:
    """Apply universal cleaning. Returns case-preserved cleaned text."""
    if text is None or not isinstance(text, str):
        return ''
    s = ftfy.fix_text(text)
    s = unicodedata.normalize('NFKC', s)
    s = _RE_CTRL.sub(' ', s)
    s = _RE_HTML.sub(' ', s)
    s = _RE_URL.sub(' [URL] ', s)
    s = _RE_USER.sub(' [USER] ', s)
    s = _RE_CASHTAG.sub(' [TICKER] ', s)
    s = emoji.demojize(s, delimiters=(' :', ': '))
    s = _RE_WS.sub(' ', s).strip()
    return s


def detect_language(text: str) -> Optional[str]:
    if not text or detect is None:
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None


def token_count(text: str) -> int:
    return len((text or '').split())


@dataclass
class PreprocessReport:
    n_input: int = 0
    n_empty_after_clean: int = 0
    n_non_english: int = 0
    n_too_short: int = 0
    n_too_long: int = 0
    n_kept: int = 0
    flags_summary: dict = field(default_factory=dict)


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = 'text',
    min_tokens: int = 3,
    max_tokens: int = 512,
    require_english: bool = True,
    langdetect_min_tokens: int = 20,
    progress_every: int = 5000,
) -> tuple[pd.DataFrame, PreprocessReport]:
    """Add text_clean / text_clean_lower / language / n_tokens / qc_flags / tickers columns.

    Performance: langdetect is only invoked on rows with at least
    `langdetect_min_tokens` tokens. Shorter rows are tagged 'en' by default
    because (a) langdetect is unreliable on short inputs and (b) every
    upstream financial source we use is English-by-construction.
    """
    rep = PreprocessReport(n_input=len(df))
    if text_col not in df.columns:
        raise KeyError(f'preprocess_dataframe: column {text_col!r} not in dataframe')

    work = df.copy().reset_index(drop=True)
    n_total = len(work)

    text_clean: List[str] = []
    languages: List[Optional[str]] = []
    token_counts: List[int] = []
    qc_flags: List[str] = []
    tickers: List[str] = []
    encoding_repaired_count = 0

    for i, raw in enumerate(work[text_col].astype(str).tolist()):
        cleaned = clean_text(raw)
        text_clean.append(cleaned)
        flags: List[str] = []
        repaired = ftfy.fix_text(raw) != raw
        if repaired:
            flags.append('encoding_repaired')
            encoding_repaired_count += 1
        if '[URL]' in cleaned:
            flags.append('had_url')
        if '[TICKER]' in cleaned:
            flags.append('had_ticker')
        if '[USER]' in cleaned:
            flags.append('had_mention')
        n_tok = token_count(cleaned)
        token_counts.append(n_tok)
        if n_tok >= langdetect_min_tokens:
            languages.append(detect_language(cleaned))
        else:
            languages.append('en')  # trust upstream
        tk = _extract_tickers(raw)
        tickers.append(','.join(tk))
        qc_flags.append(','.join(flags))
        if progress_every and (i + 1) % progress_every == 0:
            print(f'  preprocess: {i + 1}/{n_total} rows', flush=True)

    work['text_clean'] = text_clean
    work['text_clean_lower'] = [t.lower() for t in text_clean]
    work['language'] = languages
    work['n_tokens'] = token_counts
    work['qc_flags'] = qc_flags
    work['tickers'] = tickers

    # ── filter ─────────────────────────────────────────────────────────────
    keep = pd.Series(True, index=work.index)

    empty_mask = work['text_clean'].str.len().eq(0)
    rep.n_empty_after_clean = int(empty_mask.sum())
    keep &= ~empty_mask

    short_mask = work['n_tokens'] < min_tokens
    rep.n_too_short = int(short_mask.sum())
    keep &= ~short_mask

    long_mask = work['n_tokens'] > max_tokens
    rep.n_too_long = int(long_mask.sum())
    keep &= ~long_mask

    if require_english:
        non_en = (work['language'].notna()) & (work['language'] != 'en')
        rep.n_non_english = int(non_en.sum())
        keep &= ~non_en

    out = work[keep].reset_index(drop=True)
    rep.n_kept = len(out)
    rep.flags_summary = {
        'encoding_repaired': encoding_repaired_count,
        'had_url': int(out['qc_flags'].str.contains('had_url').sum()),
        'had_ticker': int(out['qc_flags'].str.contains('had_ticker').sum()),
        'had_mention': int(out['qc_flags'].str.contains('had_mention').sum()),
    }
    return out, rep
