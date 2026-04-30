# Data Source Guide

## Primary benchmark: Financial PhraseBank

Use `scripts/00_prepare_phrasebank_fiqa.py`. It tries Hugging Face configs:

- `sentences_50agree`
- `sentences_66agree`
- `sentences_75agree`
- `sentences_allagree`

The script deduplicates sentences and keeps the highest agreement level as the `agreement` column.

## External validation: FiQA

The preparation script tries known Hugging Face FiQA-style datasets. If automatic loading fails, download a FiQA CSV manually and run:

```bash
python scripts/00_prepare_phrasebank_fiqa.py --fiqa_csv data/raw/fiqa.csv
```

## StockTwits

Script:

```bash
python scripts/01_collect_stocktwits.py --symbols AAPL MSFT TSLA NVDA
```

Notes:

- StockTwits messages may contain `Bullish` or `Bearish` labels.
- API behavior and access may change.
- Use only according to StockTwits terms.

## SEC 10-K filings

Script:

```bash
export SEC_USER_AGENT="Your Name your_email@example.com"
python scripts/02_collect_sec_10k.py --tickers AAPL MSFT NVDA TSLA --years 3
```

Notes:

- Uses official SEC endpoints.
- Produces weak lexicon labels. These are not gold labels.
- Best use: retrieval corpus, qualitative evidence, or external robustness checks.

## Financial news headlines

Script:

```bash
python scripts/03_collect_financial_news.py --weak_label
```

Optional APIs:

```bash
export NEWSAPI_KEY="..."
export FINNHUB_API_KEY="..."
python scripts/03_collect_financial_news.py --include_newsapi --include_finnhub --weak_label
```

Notes:

- RSS items are usually unlabeled.
- `--weak_label` creates exploratory labels using a small financial lexicon.
- Use weak labels cautiously in the paper.
