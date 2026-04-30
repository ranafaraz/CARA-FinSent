# Experiment Guide

## E0 - Dataset preparation

```bash
python scripts/00_prepare_phrasebank_fiqa.py
```

Purpose: create standardized datasets with `text,label` and optional `agreement`.

## E1 - Classical baselines

```bash
python scripts/10_run_classical_baselines.py --data data/processed/combined_standardized_<timestamp>.csv
```

Purpose: establish majority, Logistic Regression, Linear SVM, SGD, Naive Bayes, Random Forest, and optional XGBoost baselines.

## E2 - FinBERT baseline

```bash
python scripts/11_run_finbert_baseline.py --data data/processed/combined_standardized_<timestamp>.csv --epochs 3 --batch_size 8
```

Purpose: compare against a finance-specific transformer baseline.

## E3 - Structured signals

```bash
python scripts/12_run_structured_features_experiment.py --data data/processed/combined_standardized_<timestamp>.csv
```

Purpose: test whether numerical, lexical, directional, negation, uncertainty, and ticker-like features improve performance.

## E4 - Retrieval context

```bash
python scripts/13_run_retrieval_experiment.py --data data/processed/combined_standardized_<timestamp>.csv
```

With external news corpus:

```bash
python scripts/13_run_retrieval_experiment.py --data data/processed/combined_standardized_<timestamp>.csv --external_corpus_csv data/external/financial_news_headlines_<timestamp>.csv
```

Purpose: test whether retrieved context helps short/ambiguous financial text.

## E5 - Agreement-aware training

```bash
python scripts/14_run_agreement_aware_experiment.py --data data/processed/phrasebank_standardized_<timestamp>.csv
```

Purpose: use PhraseBank agreement levels as supervision confidence.

## E6 - Calibration and abstention

```bash
python scripts/15_run_calibration_experiment.py --data data/processed/combined_standardized_<timestamp>.csv --base_model linear_svm --method sigmoid
```

Purpose: produce ECE, Brier score, reliability bins, and abstention curves.

## E7 - CARA-lite pilot

```bash
python scripts/16_run_full_cara_lite_experiment.py --data data/processed/combined_standardized_<timestamp>.csv
```

Purpose: first integrated version combining retrieval, structured features, agreement weighting if available, and calibration.

## E8 - One-command classical pipeline

```bash
python scripts/90_run_all_classical_pipeline.py --data data/processed/combined_standardized_<timestamp>.csv
```

Purpose: run baselines + structured + retrieval + calibration + CARA-lite.
