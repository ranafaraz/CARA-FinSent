# Experiment Guide

> **Canonical dataset**: `data/processed/latest.csv` (12,605 rows; train=8,036 / val=893 / test=3,676 with seed 42).
> All experiments below must run on this dataset so results are directly comparable.

## E0 - Dataset preparation

```bash
python scripts/00_prepare_phrasebank_fiqa.py
python scripts/05_build_dataset.py   # merges all sources into latest.csv
```

Purpose: create standardized datasets with `text,label` and optional `agreement`.

## E1 - Classical baselines

```bash
python scripts/10_run_classical_baselines.py --seed 42
```

Purpose: establish majority, Logistic Regression, Linear SVM, SGD, Naive Bayes, Random Forest, and optional XGBoost baselines.

## E2 - FinBERT baseline

```bash
python scripts/11_run_finbert_improved_v2.py --seed 42 --epochs 3 --batch_size 8
```

**Note**: use `11_run_finbert_improved_v2.py`, not the original `11_run_finbert_baseline.py`.
The improved script includes `ignore_mismatched_sizes=True` to correctly reinitialize the
classification head when the label order differs from the pretrained ProsusAI/finbert weights.
Without this fix the model collapses to constant "neutral" predictions.

For zero-shot evaluation of the base model (no fine-tuning):

```bash
python scripts/11c_eval_finbert_zero_shot.py
```

## E3 - Structured signals

```bash
python scripts/12_run_structured_features_experiment.py --seed 42
```

Purpose: test whether numerical, lexical, directional, negation, uncertainty, and ticker-like features improve performance.

## E4 - Retrieval context

**Key finding**: TF-IDF self-retrieval from the training corpus *degrades* macro-F1 by ~0.06.
Short financial sentences on similar topics have mixed sentiments; retrieving them
contaminates the input signal. Retrieval only helps when an **external** domain corpus
(SEC 10-K filings, news headlines) is used as the knowledge base.

```bash
# Retrieval ablation (self-retrieval from training corpus, confirmed to degrade performance)
python scripts/13_run_retrieval_experiment.py --seed 42

# Retrieval with external corpus (recommended for CARA)
python scripts/13_run_retrieval_experiment.py --seed 42 \
  --external_corpus_csv data/external/financial_news_headlines_<timestamp>.csv
```

## E5 - Agreement-aware training

```bash
python scripts/14_run_agreement_aware_experiment.py --seed 42
```

Purpose: use PhraseBank agreement levels as supervision confidence weights.
Marginal improvement (~+0.001 macro-F1 over unweighted baseline); confirms label quality
correlates with annotation confidence.

## E6 - Calibration and abstention

```bash
python scripts/15_run_calibration_experiment.py --seed 42 --method sigmoid
```

Purpose: produce ECE, Brier score, reliability bins, and abstention curves.

## E7 - CARA-lite pilot

```bash
python scripts/16_run_full_cara_lite_experiment.py --seed 42
```

Default configuration: LinearSVC base, structured features on, agreement weighting on,
sigmoid calibration on, self-retrieval **disabled** (external corpus required for retrieval).

With external retrieval corpus:

```bash
python scripts/16_run_full_cara_lite_experiment.py --seed 42 \
  --external_corpus_csv data/external/financial_news_headlines_<timestamp>.csv
```

Purpose: integrated pipeline combining structured signals, agreement weighting, and calibration.
Retrieval is only activated when an external corpus is provided.

## E8 - One-command classical pipeline

```bash
python scripts/90_run_all_classical_pipeline.py
```

Purpose: run baselines + structured + retrieval + calibration + CARA-lite.

