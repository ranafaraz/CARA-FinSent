# 04 · Dataset assembly

This is what [`scripts/05_build_dataset.py`](../scripts/05_build_dataset.py)
does end-to-end. It composes the modules from sections 02 and 03 into a
single deterministic pipeline whose outputs go to
[`data/processed/latest/`](../data/processed/latest).

## Seven steps

```
1. Ingest        legacy latest.csv  +  every data/raw/<source>/*.csv
                  → tag with `source` and `tier` columns
2. Preprocess    preprocess_dataframe(df, text_col='text')
3. Dedup         dedup_dataframe(df, threshold=0.85)
4. Carve test    carve_test_set(df, target_size=1450, seed=42)
5. Balance       balance_classes(train_pool, max_imbalance=2.0, seed=42)
6. Decontaminate contamination_check(train, test, threshold=0.5) → drop train hits
7. Write         dataset.csv + manifest.json + data_card.md
                 + immutable archive snapshot
                 + backwards-compat data/processed/latest.csv
```

## Step 1 — ingest

* `ingest_legacy_latest()` reads any pre-existing
  `data/processed/latest.csv` so we never silently lose rows that were
  already labelled.
* `ingest_raw_dirs()` walks `data/raw/*/` and picks the **latest CSV per
  source** (`find_latest_csv()`), so re-running a collector deterministically
  upgrades that source on the next build without touching the others.
* Every row is given a `source` (folder name) and a `tier` via `_tier(source)`.

## Step 2 — preprocess

Calls into [`preprocessing.py`](../src/cara_finsent/preprocessing.py); see
[02_preprocessing](02_preprocessing.md).

## Step 3 — dedup

Calls into [`dedup.py`](../src/cara_finsent/dedup.py); see
[03_deduplication_and_quality](03_deduplication_and_quality.md).

## Step 4 — test-set carving

Function: `carve_test_set(df, target_size=1450, seed=42)`.

Rules, in priority order:

1. Any row that already had `split == 'test'` from a prior build is
   **kept in test** (so re-builds don't shuffle the holdout).
2. From the remaining gold-tier rows, sample stratified by `source` with
   per-source caps so no single source dominates the test set. Within a
   source, prefer rows with the highest `agreement` score (PhraseBank
   has 50 %–100 % annotator agreement bands; we take the top first).
3. Stop when we hit `target_size` (1,450 rows) or run out of gold.
4. Silver and bronze are **never** selected for test.

Build `20260430_212903` ended up at **1,205 test rows** (target was 1,450
but capped by available gold across sources):

| Source | Test rows |
|---|---:|
| fomc_sentiment | 638 |
| fiqa | 292 |
| financial_phrasebank | 165 |
| semeval2017_task5 | 110 |

Final test label distribution: `{neutral: 483, positive: 378, negative: 344}`
— all three classes well above the 50-row minority floor required by the
quality test suite.

## Step 5 — class balancing

Function: `balance_classes(df, max_imbalance=2.0, seed=42)`.

Strategy: **silver-first majority undersampling**. We never touch the
minority class (negatives are scarce and precious). For each majority
class we cap its row count at `max_imbalance × |minority|`. When dropping
rows we drop **silver before gold** so we preserve as much human-labelled
data as possible.

Build `20260430_212903`:

```
before: positive=30,519  neutral=16,234  negative=2,546
target_max = 2 × 2,546 = 5,092
after:  positive=5,092   neutral=5,092   negative=2,546
```

The final train ratio is exactly 2.00 — within the 2.5× envelope enforced
by `tests/test_dataset_quality.py::test_train_imbalance_within_threshold`.

> Paraphrasing-based oversampling is implemented in
> [`scripts/07_balance_classes.py`](../scripts/07_balance_classes.py)
> behind a `--paraphrase` flag (Vamsi/T5_Paraphrase_Paws). It is **off**
> in the canonical build because it would create synthetic-tier rows we
> haven't validated. Turn it on if you want to push negatives up
> artificially.

## Step 6 — contamination check

See [03_deduplication_and_quality](03_deduplication_and_quality.md). 20
train rows dropped for build `20260430_212903`.

## Step 7 — write artefacts

`data/processed/latest/`
* `dataset.csv` — canonical, with full schema (text, text_clean,
  text_clean_lower, label, source, tier, split, agreement, n_tokens,
  qc_flags, tickers, language, license, build_id).
* `manifest.json` — full provenance (sources, counts, preprocess report,
  dedup report, balance report, contamination count, sample near-dup
  pairs).
* `data_card.md` — auto-generated HF-style data card with YAML
  frontmatter, source table, methodology one-pager, schema.
* `contamination_report.csv` — only written when there were leaks.

`data/processed/archive/dataset_<build_id>.csv` — immutable snapshot.

`data/processed/latest.csv` — backwards-compat copy with
`text_clean_lower → text_lower` rename so legacy scripts (10–17, 90)
keep working without code changes.

## CLI

```bash
# Dry run — runs everything but only prints the manifest, no files written.
python scripts/05_build_dataset.py --dry-run

# Real build with defaults (seed=42, max_imbalance=2.0, target_test=1450).
python scripts/05_build_dataset.py

# Custom imbalance / test size / seed.
python scripts/05_build_dataset.py --max-imbalance 1.5 --target-test-size 1000 --seed 7
```
