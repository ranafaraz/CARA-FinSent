# CARA-FinSent Methodology

This folder is the **single source of truth** for how the CARA-FinSent
financial-sentiment dataset and experiment pipeline are built, validated and
evaluated. It is written so that any contributor (or downstream agent) can
reproduce every artefact in [`results/`](../results) and
[`data/processed/latest/`](../data/processed/latest) from a clean checkout.

## Reading order

| # | Document | Purpose |
|---|---|---|
| 00 | [overview](00_overview.md) | Research question, system diagram, deliverables |
| 01 | [dataset_collection](01_dataset_collection.md) | Every source, license, collector script |
| 02 | [preprocessing](02_preprocessing.md) | Text cleaning, language filter, tokenisation |
| 03 | [deduplication_and_quality](03_deduplication_and_quality.md) | Exact + MinHash near-dup, contamination |
| 04 | [dataset_assembly](04_dataset_assembly.md) | Tier system, test-set carving, class balancing |
| 05 | [validation](05_validation.md) | The 14-test pytest suite |
| 06 | [classical_baselines](06_classical_baselines.md) | TF-IDF + LR / SVM / SGD / RF, structured features |
| 07 | [finbert_finetuning](07_finbert_finetuning.md) | FinBERT fine-tune setup and CPU run notes |
| 08 | [evaluation_and_comparison](08_evaluation_and_comparison.md) | Metrics, calibration, leaderboard |
| 09 | [reproduction](09_reproduction.md) | End-to-end command sequence |
| 10 | [design_decisions](10_design_decisions.md) | Why we chose what we chose |

## At-a-glance pipeline

```
raw HF datasets ──► scripts/00,04,08,09  ──► data/raw/<source>/<ts>.csv
                                                       │
                                                       ▼
                                         scripts/05_build_dataset.py
                                                       │
            ┌───────────────────┬───────────────────┼─────────────────────┐
            ▼                   ▼                   ▼                     ▼
       preprocessing        deduplication       test carving         class balance
       (preprocessing.py)   (dedup.py)          (gold stratified)    (max imbalance 2.0)
                                                       │
                                                       ▼
                                  data/processed/latest/{dataset.csv, manifest.json, data_card.md}
                                                       │
                                                       ▼
                            tests/test_dataset_quality.py (14 checks)
                                                       │
                                                       ▼
                       scripts/10–17 (classical, FinBERT, retrieval, calibration)
                                                       │
                                                       ▼
                          results/YYYY-MM-DD/ + figures/ + leaderboard
```

## Canonical artefacts produced

* [data/processed/latest/dataset.csv](../data/processed/latest/dataset.csv) — 13,916 rows × ~14 columns
* [data/processed/latest/manifest.json](../data/processed/latest/manifest.json) — full provenance
* [data/processed/latest/data_card.md](../data/processed/latest/data_card.md) — HF-style data card
* [data/processed/latest.csv](../data/processed/latest.csv) — backwards-compat snapshot for legacy scripts
* [data/processed/archive/dataset_<ts>.csv](../data/processed/archive) — immutable build snapshots

## Conventions used throughout

* All scripts are **deterministic** with `seed=42` unless documented otherwise.
* All result files follow `results/YYYY-MM-DD/<prefix>_YYYYMMDD_HHMMSS.csv`.
* Standard label set: `negative`, `neutral`, `positive` (mapped via [`normalize_label`](../src/cara_finsent/data_utils.py)).
* Quality tier set: `gold > silver > synthetic > bronze`. Sample weights: `1.0 / 0.5 / 0.3 / 0.0`.
* Working directory for every command is the repo root.
