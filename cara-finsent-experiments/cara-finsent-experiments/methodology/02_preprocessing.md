# 02 · Preprocessing

All cleaning happens in
[`src/cara_finsent/preprocessing.py`](../src/cara_finsent/preprocessing.py)
and is invoked by step 2 of [`scripts/05_build_dataset.py`](../scripts/05_build_dataset.py).
The function of interest is `preprocess_dataframe(df, text_col='text', …)`.

## Pipeline (per row)

```
raw text
   │
   ▼ ftfy.fix_text             repair mojibake / smart-quotes / "Â" garbage
   ▼ unicodedata.normalize NFKC compose width / ligatures
   ▼ strip control & HTML       drop \x00..\x1f / "<...>" tags
   ▼ regex masking
        │  https?://… or www.…  ⇒  [URL]
        │  @user                 ⇒  [USER]
        │  $TICKER               ⇒  [TICKER]   (ticker also captured separately)
   ▼ emoji.demojize             "🚀" ⇒ ":rocket:"
   ▼ collapse whitespace
   ▼ language detect            langdetect, only if n_tokens ≥ 20 (else 'en')
   ▼ length filter              keep iff 3 ≤ n_tokens ≤ 512
```

The result is appended to the DataFrame as the following columns:

| Column | Type | Description |
|---|---|---|
| `text_clean` | str | Cleaned, masked, demojised text — used by FinBERT and as the canonical text |
| `text_clean_lower` | str | Lower-cased version, used by TF-IDF baselines |
| `language` | str | langdetect ISO-639-1 (`'en'` for short rows we trust) |
| `n_tokens` | int | Whitespace token count (after cleaning) |
| `qc_flags` | str | Comma-separated tags: `encoding_repaired`, `had_url`, `had_ticker`, `had_mention` |
| `tickers` | str | Comma-separated original `$TICKER` symbols extracted before masking |

## Why mask instead of strip?

Stripping URLs / tickers / mentions destroys evidence the classifier may
need — e.g. a tweet that is *just* a `$TSLA` and an emoji becomes empty.
Masking preserves the **slot** (so word counts stay sensible) while
removing the **identity** (so the model can't memorise URLs as a label
proxy and so we don't leak ticker prevalence into the test set).

The original tickers are kept in their own column so downstream features
(structured-features baseline, retrieval) can still use them.

## Why `langdetect` only on long rows?

`langdetect` is unreliable below ~20 tokens and is the slowest step
(~10 µs per call ≈ 30 s on 60k rows). Every upstream source we ingest is
English-by-construction, so for short rows we **trust** the source label
of `'en'`. For long rows (where a stray Finnish PhraseBank entry might
sneak in) we run the detector. In practice this drops 194 rows from the
final corpus — about 0.3 %.

## Filter outcomes for build `20260430_212903`

```
input               66,493
empty after clean        1
non-English            194
too short              518   (< 3 tokens after masking)
too long                 2   (> 512 tokens)
kept                65,779
```

The flag histogram tells us how dirty the corpus was before masking:

```
encoding_repaired   6,354   (rows ftfy actually fixed)
had_url            22,189
had_ticker         22,710
had_mention         7,086
```

≈ 1 in 10 rows had to be repaired — a meaningful corruption rate that
classical baselines silently learn around if you don't fix it.

## Reproducing

```python
import pandas as pd
from cara_finsent.preprocessing import preprocess_dataframe

df = pd.read_csv('data/raw/financial_phrasebank/financial_phrasebank_<ts>.csv')
clean, report = preprocess_dataframe(df, text_col='text')
print(report)
clean.head()
```

## Sanity tests covering this stage

* `tests/test_dataset_quality.py::test_no_url_or_at_user_or_cashtag_in_text_clean`
* `tests/test_dataset_quality.py::test_no_empty_text_clean`
* `tests/test_dataset_quality.py::test_token_length_in_range`
