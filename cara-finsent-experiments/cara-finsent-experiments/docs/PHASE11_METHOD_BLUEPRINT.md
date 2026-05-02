# Phase 11 — Methodology Blueprint

Date: 2026-05-02
Scope: Method-section blueprint for the CARA-FinSent paper. This file is the
single source of truth that the methodology section of the IEEE draft must
follow.

## Research questions

- **RQ1.** Does agreement-aware fine-tuning improve financial sentiment
  classification compared with zero-shot and vanilla fine-tuned FinBERT?
- **RQ2.** Does agreement-aware fine-tuning improve seed stability and
  reliability?
- **RQ3.** How do calibration and abstention change the practical usefulness of
  financial sentiment models?
- **RQ4.** Can low-cost classical baselines remain competitive enough for
  resource-constrained environments?

## Hypotheses

- **H1.** AW improves mean macro-F1 modestly over both zero-shot and vanilla
  fine-tuned FinBERT under leakage-controlled, multi-seed evaluation.
- **H2.** AW reduces seed-to-seed macro-F1 variance relative to vanilla
  fine-tuning.
- **H3.** Zero-shot FinBERT remains better calibrated (lower ECE@10) than
  trained FinBERT variants.
- **H4.** Classical TF-IDF baselines remain materially below FinBERT-class
  models on macro-F1 but retain a deployment-cost advantage.

## Dataset description

- Source: Financial PhraseBank with ≥75% annotator agreement.
- Labels: {negative, neutral, positive}, normalized exactly to those strings.
- Token granularity: sentence-level.
- Split sizes (per [paper_assets/tables/table1_dataset_summary.csv](../paper_assets/tables/table1_dataset_summary.csv)):
  train=3,353; val=479; test=959.

## Split strategy

- A controlled gold split is generated once and stored as
  `data/processed/gold/latest_gold_phrasebank_split.csv`.
- The same split is used by every model and every seed to enable strictly
  paired comparisons.

## Leakage prevention

- Per-instance text hashes are recorded for every split.
- A `no_text_hash_leakage` audit check runs on every gate invocation and is
  required to be PASS for the run to be considered research-grade.
- The current evidence shows zero text-hash overlap between train/val/test for
  every model artifact.

## Models evaluated

| family | variants |
|---|---|
| classical | TF-IDF + {linear SVM, logistic regression, SGD log-loss, XGBoost, random forest, multinomial NB}, plus a majority-class baseline |
| FinBERT zero-shot | `ProsusAI/finbert` with native head, evaluated without fine-tuning |
| FinBERT fine-tuned | `ProsusAI/finbert` with supervised cross-entropy fine-tuning |
| FinBERT agreement-weighted | `ProsusAI/finbert` fine-tuned with per-instance loss weights derived from annotator agreement |

## Agreement-weighting strategy

- A per-instance weight `w_i` is derived from the PhraseBank annotator-agreement
  level (e.g. 75%, 100%) and rescaled with a `linear` schedule.
- The training loss becomes `L = mean_i(w_i * CE(logits_i, y_i))`.
- Weights are added at the dataset level via a `batched=True` map call so that
  per-batch weights match per-batch labels.

## Calibration metrics

- Expected Calibration Error with 10 equal-width bins (ECE@10).
- Brier score (multiclass, sum of squared probability errors averaged over
  instances).
- Mean predictive confidence and reliability bin counts are also recorded for
  visual diagnostics.

## Abstention method

- Confidence threshold `t ∈ {0.0, 0.05, ..., 0.95}` applied to the max softmax
  probability.
- For each `t`, we report coverage, abstention rate, accuracy on kept,
  macro-F1 on kept, and counts of kept/abstained instances.

## Statistical validation method

- Per-model bootstrap CI: percentile bootstrap on per-seed macro-F1 values
  (n_bootstrap = 2000), plus a within-test bootstrap CI on prediction-level
  outputs for one representative seed.
- Paired bootstrap: matched-instance resampling of per-prediction outputs for
  three pairs (AW vs zero-shot, AW vs fine-tuned, zero-shot vs fine-tuned),
  reporting Δ macro-F1, 95% CI, and a two-sided empirical p-value.
- McNemar exact test on discordant correctness counts for each pair.
- Effect size: Cohen's d-like ratio of across-seed macro-F1 means to pooled
  std.

## Evaluation metrics

| metric | role |
|---|---|
| accuracy | sanity baseline |
| macro-F1 | primary headline metric (class-imbalanced) |
| weighted-F1 | secondary |
| MCC | rank-aware secondary |
| ECE@10 | calibration |
| Brier score | calibration / sharpness |
| coverage @ t | abstention |
| accuracy @ t / macro-F1 @ t | abstention quality |

## Reproducibility controls

- Fixed random seeds: {13, 21, 42, 87, 101} for FinBERT variants; classical
  baselines use the deterministic-by-design pipeline plus the same seed pool.
- All scripts log `git_commit_sha` into every summary CSV.
- All output files are timestamped to prevent overwrite.
- A single research-gate script (`scripts/25_research_gate.py`) enforces
  artifact integrity and minimum-seed-coverage rules before any leaderboard or
  paper asset is treated as final.
