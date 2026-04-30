# CARA-FinSent Execution Guide

## Objective

Generate experimental evidence for a publication-grade financial sentiment analysis paper. The target is not another accuracy-only benchmark. The target is a decision-grade system that adds retrieval, structured financial signals, agreement-aware training, calibration, abstention, and external validation.

## Expected final evidence package

After running the scripts, the `results/` folder should contain:

- classical baseline summary
- structured feature ablation summary
- retrieval experiment summary
- agreement-aware training summary
- calibration summary
- CARA-lite summary
- prediction-level CSV files
- confusion matrices
- reliability bins
- abstention curves

The `figures/` folder should contain confusion matrix and reliability diagrams.

## Step 0 - Setup

Local setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python -m compileall -q src scripts
```

Docker setup:

```bash
docker build -t cara-finsent .
docker run -it --rm -v "$PWD:/app" cara-finsent bash
```

Colab setup:

```python
!git clone <your-repo-url>
%cd cara-finsent-experiments
!pip install -r requirements.txt
```

## Step 1 - Prepare primary and validation datasets

Run:

```bash
python scripts/00_prepare_phrasebank_fiqa.py
```

If automatic downloads fail, place your dataset CSV in `data/raw/` and run:

```bash
python scripts/00_prepare_phrasebank_fiqa.py --local_csv data/raw/your_dataset.csv --skip_hf
```

Required CSV columns:

```csv
text,label
```

Optional columns:

```csv
id,source_dataset,agreement,split
```

## Step 2 - Run classical baselines

```bash
python scripts/10_run_classical_baselines.py --data data/processed/combined_standardized_<timestamp>.csv
```

Goal: establish the baseline performance of majority classifier, Logistic Regression, Linear SVM, SGD, Naive Bayes, Random Forest, and optional XGBoost.

## Step 3 - Run FinBERT baseline

Use GPU if possible.

```bash
python scripts/11_run_finbert_baseline.py \
  --data data/processed/combined_standardized_<timestamp>.csv \
  --model_name ProsusAI/finbert \
  --epochs 3 \
  --batch_size 8
```

For a smoke test:

```bash
python scripts/11_run_finbert_baseline.py --data data/processed/combined_standardized_<timestamp>.csv --max_rows 1000 --epochs 1
```

Goal: compare your system against the canonical finance-specific transformer baseline.

## Step 4 - Run structured financial signal ablation

```bash
python scripts/12_run_structured_features_experiment.py --data data/processed/combined_standardized_<timestamp>.csv
```

Goal: test whether numeric, lexical, negation, uncertainty, direction, and ticker-like signals improve performance.

## Step 5 - Collect external corpora

### StockTwits

```bash
python scripts/01_collect_stocktwits.py --symbols AAPL MSFT TSLA NVDA --limit_per_symbol 30
```

Optional:

```bash
export STOCKTWITS_ACCESS_TOKEN="..."
```

### SEC 10-K

```bash
export SEC_USER_AGENT="Your Name your_email@example.com"
python scripts/02_collect_sec_10k.py --tickers AAPL MSFT NVDA TSLA --years 3 --max_filings_per_ticker 2
```

### Financial news headlines

```bash
python scripts/03_collect_financial_news.py --weak_label
```

Optional APIs:

```bash
export NEWSAPI_KEY="..."
export FINNHUB_API_KEY="..."
python scripts/03_collect_financial_news.py --include_newsapi --include_finnhub --weak_label
```

## Step 6 - Run retrieval experiment

Basic retrieval using training corpus:

```bash
python scripts/13_run_retrieval_experiment.py --data data/processed/combined_standardized_<timestamp>.csv
```

With external news corpus:

```bash
python scripts/13_run_retrieval_experiment.py \
  --data data/processed/combined_standardized_<timestamp>.csv \
  --external_corpus_csv data/external/financial_news_headlines_<timestamp>.csv
```

Goal: determine whether extra context improves short and ambiguous financial text.

## Step 7 - Run agreement-aware training

Use the PhraseBank prepared file, because it contains agreement values.

```bash
python scripts/14_run_agreement_aware_experiment.py --data data/processed/phrasebank_standardized_<timestamp>.csv
```

Goal: test whether high-agreement examples should receive stronger supervision and whether this improves low-agreement/ambiguous performance.

## Step 8 - Run calibration and abstention

```bash
python scripts/15_run_calibration_experiment.py \
  --data data/processed/combined_standardized_<timestamp>.csv \
  --base_model linear_svm \
  --method sigmoid
```

Goal: produce confidence validity evidence using ECE, Brier score, reliability bins, and abstention curves.

## Step 9 - Run CARA-lite integrated pipeline

```bash
python scripts/16_run_full_cara_lite_experiment.py --data data/processed/combined_standardized_<timestamp>.csv
```

With external corpus:

```bash
python scripts/16_run_full_cara_lite_experiment.py \
  --data data/processed/combined_standardized_<timestamp>.csv \
  --external_corpus_csv data/external/financial_news_headlines_<timestamp>.csv
```

Goal: run the first integrated version of CARA-FinSent: retrieval + structured features + agreement weighting if available + calibration.

## Step 10 - Run the one-command classical pipeline

```bash
python scripts/90_run_all_classical_pipeline.py --data data/processed/combined_standardized_<timestamp>.csv
```

## Step 11 - Share results for comparison

Zip and share:

```text
results/
figures/
```

Minimum files needed:

```text
classical_baseline_summary_*.csv
structured_features_summary_*.csv
retrieval_experiment_summary_*.csv
agreement_aware_summary_*.csv
calibration_summary_*.csv
cara_lite_summary_*.csv
```

## How results will be judged

The paper should not only ask: which model has the highest F1?

It should ask:

1. Does retrieval improve short-text and neutral-class performance?
2. Do structured financial signals add useful evidence?
3. Does agreement-aware weighting improve ambiguous cases?
4. Does calibration reduce overconfidence?
5. Does abstention improve reliability when confidence is low?
6. Is the pipeline practical in latency and cost?

## Scientific warnings

- Do not claim stock-market prediction from PhraseBank alone.
- Treat StockTwits, SEC 10-K, and news weak labels as noisy unless manually validated.
- Do not make SOTA claims until the complete results folder proves them.
- The likely contribution is reliability and decision utility, not only raw accuracy.
