# CARA-FinSent Experimental Codebase

This repository contains executable scripts for building the experimental evidence needed for the proposed paper:

**CARA-FinSent: Confidence-Aware Retrieval-Augmented Financial Sentiment Analysis with Agreement-Aware Training and Numerical-Event Grounding**

The research goal is to move financial sentiment analysis beyond simple label classification. The system should produce sentiment plus confidence and evidence, while evaluating calibration, abstention, retrieval benefit, structured financial signals, external validation, latency, and cost.

## What this codebase does

It provides scripts for:

1. Preparing Financial PhraseBank and FiQA into a standard CSV format.
2. Collecting additional financial corpora from SEC 10-K filings and financial news RSS/API sources.
3. Running classical ML baselines.
4. Running FinBERT fine-tuning/evaluation.
5. Running structured-feature ablations.
6. Running retrieval-context experiments.
7. Running agreement-aware training experiments.
8. Running calibration and abstention experiments.
9. Running a first full **CARA-lite** pipeline combining retrieval, structured signals, agreement weighting, and calibration.

Every script saves timestamped CSV files into `results/` to avoid overwriting previous runs.

## Standard dataset schema

Most experiment scripts expect a CSV with at least:

```csv
text,label
"Company profit increased by 20%",positive
"Revenue remained unchanged",neutral
"Shares declined after weak guidance",negative
```

Optional columns:

```csv
id,source_dataset,agreement,split
```

Accepted labels are:

```text
negative, neutral, positive
```

## Quick local setup

```bash
git clone <your-repo-url>
cd cara-finsent-experiments
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Check syntax:

```bash
python -m compileall -q src scripts
```

## Quick Colab setup

```python
!git clone <your-repo-url>
%cd cara-finsent-experiments
!pip install -r requirements.txt
```

For GPU FinBERT experiments, use Colab GPU runtime.

## Step 1: Prepare datasets

Try Hugging Face download:

```bash
python scripts/00_prepare_phrasebank_fiqa.py
```

Or use a local CSV:

```bash
python scripts/00_prepare_phrasebank_fiqa.py --local_csv data/raw/your_dataset.csv --skip_hf
```

Output examples:

```text
data/processed/phrasebank_standardized_YYYYMMDD_HHMMSS.csv
data/processed/fiqa_standardized_YYYYMMDD_HHMMSS.csv
data/processed/combined_standardized_YYYYMMDD_HHMMSS.csv
results/dataset_inventory_YYYYMMDD_HHMMSS.csv
```

## Step 2: Run classical baselines

```bash
python scripts/10_run_classical_baselines.py --data data/processed/combined_standardized_<timestamp>.csv
```

Outputs:

```text
results/classical_baseline_summary_<timestamp>.csv
results/classical_baseline_predictions_<timestamp>.csv
results/*_confusion_matrix_<timestamp>.csv
figures/*_confusion_matrix_<timestamp>.png
```

## Step 3: Run FinBERT baseline

```bash
python scripts/11_run_finbert_baseline.py \
  --data data/processed/combined_standardized_<timestamp>.csv \
  --model_name ProsusAI/finbert \
  --epochs 3 \
  --batch_size 8
```

Use `--max_rows 1000` for a cheap smoke test.

## Step 4: Run ablations and reliability experiments

```bash
python scripts/12_run_structured_features_experiment.py --data data/processed/combined_standardized_<timestamp>.csv
python scripts/13_run_retrieval_experiment.py --data data/processed/combined_standardized_<timestamp>.csv
python scripts/14_run_agreement_aware_experiment.py --data data/processed/phrasebank_standardized_<timestamp>.csv
python scripts/15_run_calibration_experiment.py --data data/processed/combined_standardized_<timestamp>.csv
python scripts/16_run_full_cara_lite_experiment.py --data data/processed/combined_standardized_<timestamp>.csv
```

## Step 5: Run the classical/reliability pipeline

```bash
python scripts/90_run_all_classical_pipeline.py --data data/processed/combined_standardized_<timestamp>.csv
```

## External corpus collection

### SEC 10-K filings

SEC requires a real user-agent with contact email:

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

## What to send back for analysis

After running scripts, zip and send:

```text
results/
figures/
```

At minimum, send:

```text
classical_baseline_summary_*.csv
structured_features_summary_*.csv
retrieval_experiment_summary_*.csv
agreement_aware_summary_*.csv
calibration_summary_*.csv
cara_lite_summary_*.csv
```

## Important scientific warning

Do not claim stock prediction superiority from Financial PhraseBank alone. PhraseBank is not time-aligned with returns. Use SEC filings and news only as external validation or weakly labeled retrieval corpus unless you later build a time-aligned return dataset.
