# 06 · Classical baselines

The non-transformer arm of the comparison. Five scripts feed five (or more)
scikit-learn / XGBoost models, all trained on the same canonical
`text_clean_lower` text and the same `train` split.

| Script | Models / variants |
|---|---|
| [10_run_classical_baselines.py](../scripts/10_run_classical_baselines.py) | majority, TF-IDF + Logistic Regression / LinearSVC / SGD-log-loss / Random Forest |
| [12_run_structured_features_experiment.py](../scripts/12_run_structured_features_experiment.py) | TF-IDF (LR / SVM) **with and without** structured numeric features |
| [13_run_retrieval_experiment.py](../scripts/13_run_retrieval_experiment.py) | retrieval-augmented LR — k-NN over a TF-IDF index, soft labels appended |
| [14_run_agreement_aware_experiment.py](../scripts/14_run_agreement_aware_experiment.py) | LR/SVM with `sample_weight` from `agreement` and `tier` |
| [15_run_calibration_experiment.py](../scripts/15_run_calibration_experiment.py) | Platt + isotonic calibration on the LR baseline |
| [16_run_full_cara_lite_experiment.py](../scripts/16_run_full_cara_lite_experiment.py) | full stack (structured + retrieval + agreement + calibration) |
| [90_run_all_classical_pipeline.py](../scripts/90_run_all_classical_pipeline.py) | runs 10/12/13/14/15/16 in order |

## Vectoriser

A single shared TF-IDF configuration is used everywhere:

```python
TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    stop_words='english',
)
```

* **Lowercased input** — we use `text_clean_lower`, so the vectoriser
  never has to lowercase again.
* **No min_df / max_df** — at ~13k rows the vocabulary is small enough
  that `max_features=50,000` is the binding constraint.
* **English stop-words** — fine because we filter to English in step 2.

## Models and key hyperparameters

| Model | Key args |
|---|---|
| `majority_baseline` | `DummyClassifier(strategy='most_frequent')` |
| `tfidf_logistic_regression` | `LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=-1)` |
| `tfidf_linear_svm` | `LinearSVC(class_weight='balanced')` |
| `tfidf_sgd_log_loss` | `SGDClassifier(loss='log_loss', class_weight='balanced')` |
| `tfidf_random_forest` | `RandomForestClassifier(n_estimators=300, class_weight='balanced', n_jobs=-1)` |
| `tfidf_xgboost` (optional) | `XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1)` |

All models pass `random_state=42` (or `seed=42`).

## Structured features (script 12)

Built by [`feature_extractor.py`](../src/cara_finsent/feature_extractor.py)
and concatenated to the TF-IDF matrix with `scipy.sparse.hstack`:

* `n_tokens`
* `n_chars`
* number of `[URL]` / `[USER]` / `[TICKER]` placeholders
* per-row count of finance-positive / finance-negative lexicon hits
* `agreement` (when present)

Net effect on macro-F1 is small but consistent (+0.5–1.0 pt).

## Retrieval baseline (script 13)

Implemented in [`retrieval.py`](../src/cara_finsent/retrieval.py).

* Build a k-NN index over the train TF-IDF matrix (`NearestNeighbors`,
  cosine, `k=8`).
* For each test row, fetch the 8 nearest train rows, average their
  one-hot label vectors → 3-dim soft-label feature.
* Concatenate that 3-dim feature to the TF-IDF matrix and train an LR.

## Agreement-aware loss (script 14)

`sample_weight` per training row:

```
weight = tier_weight[tier] * (agreement / 100 if agreement is not NaN else 1.0)
tier_weight = {'gold': 1.0, 'silver': 0.5, 'synthetic': 0.3, 'bronze': 0.0}
```

The bronze rows get weight 0, i.e. they are present for vocabulary but
contribute no gradient.

## Calibration (script 15)

Two passes of `CalibratedClassifierCV` on top of the LR baseline:

* Platt scaling (`method='sigmoid'`)
* Isotonic regression (`method='isotonic'`)

Reported alongside Brier score and Expected Calibration Error (ECE) — see
[`metrics.py`](../src/cara_finsent/metrics.py).

## Outputs

Every script writes one or more CSVs into
`results/YYYY-MM-DD/<prefix>_YYYYMMDD_HHMMSS.csv` with at minimum:

```
model, split, accuracy, macro_f1, macro_precision, macro_recall,
weighted_f1, n_test, dataset_path, runtime_seconds
```

Per-class breakdowns and confusion matrices are written to sibling files
with `_perclass.csv` and `_confusion.json` suffixes.
