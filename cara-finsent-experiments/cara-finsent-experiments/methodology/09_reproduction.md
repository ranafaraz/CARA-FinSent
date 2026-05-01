# 09 · Reproduction guide

End-to-end commands to rebuild every artefact from a clean clone. Tested
on Windows + PowerShell 7. For bash just swap `$env:VAR=…` for
`export VAR=…`.

## 0. One-time setup

```powershell
# from the repo root
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# add a Hugging Face token (needed by collectors)
"HF_TOKEN=<your_token>" | Out-File -FilePath .env -Encoding ascii
```

(or, in PowerShell for the current session only:
`$env:HF_TOKEN="<your_token>"`)

## 1. Collect the raw data layer

```powershell
python scripts/00_prepare_phrasebank_fiqa.py
python scripts/04_collect_extra_datasets.py
python scripts/08_collect_semeval2017.py
python scripts/09_collect_fomc_sentiment.py
```

Each writes one CSV to `data/raw/<source>/`. They are idempotent in the
sense that re-running creates a new timestamped file; the build always
takes the latest per source.

## 2. Build the canonical dataset

Dry run first (cheap, prints manifest, writes nothing):

```powershell
python scripts/05_build_dataset.py --dry-run
```

Then for real:

```powershell
python scripts/05_build_dataset.py
```

Outputs:

```
data/processed/latest/dataset.csv
data/processed/latest/manifest.json
data/processed/latest/data_card.md
data/processed/latest/contamination_report.csv   # only if leaks found
data/processed/archive/dataset_<build_id>.csv
data/processed/latest.csv                         # backwards-compat
```

## 3. Validate

```powershell
python -m pytest tests/test_dataset_quality.py -v
```

All 14 tests must pass. If they don't, fix the dataset before training
anything — see [`05_validation.md`](05_validation.md) for what each test
catches.

## 4. Run the classical baseline suite

Either piecewise:

```powershell
python scripts/10_run_classical_baselines.py
python scripts/12_run_structured_features_experiment.py
python scripts/13_run_retrieval_experiment.py
python scripts/14_run_agreement_aware_experiment.py
python scripts/15_run_calibration_experiment.py
python scripts/16_run_full_cara_lite_experiment.py
```

Or in one shot:

```powershell
python scripts/90_run_all_classical_pipeline.py
```

Wall-clock for the whole thing on the reference machine: 2–4 min.

## 5. Fine-tune FinBERT (long)

```powershell
python scripts/11_run_finbert_baseline.py
```

CPU runtime: 8–10 h for 3 epochs at batch size 8. Don't run anything else
heavy on the same machine.

## 6. Compare and chart

```powershell
python scripts/17_compare_finbert_vs_classical.py
```

Produces:

```
results/<date>/comparison_<ts>.csv
results/<date>/comparison_perclass_<ts>.csv
figures/leaderboard_<ts>.png
figures/runtime_vs_f1_<ts>.png
figures/per_source_heatmap_<ts>.png
```

## 7. Quick "is everything still wired up?" smoke test

```powershell
python -m pytest tests/test_imports.py -v          # module imports
python -m pytest tests/test_dataset_quality.py -v  # dataset gates
python scripts/05_build_dataset.py --dry-run       # build pipeline
python scripts/10_run_classical_baselines.py       # < 1 min
```

Four green commands and you know the codebase is healthy without
spending a CPU-day on FinBERT.

## Common pitfalls

* **`HF_TOKEN` not set** — collectors will hang on private/gated repos.
  Always export it.
* **Old `latest.csv` kept around** — script 05 *appends* it during
  ingest. If you want a clean rebuild from raw, delete
  `data/processed/latest.csv` *and* `data/processed/latest/dataset.csv`
  before running step 2.
* **`langdetect` non-determinism** — the module sets `DetectorFactory.seed = 42`
  inside `preprocessing.py`. If you see flakey language tags across builds,
  check that this seed line wasn't lost in a refactor.
* **Unicode `→` arrows on Windows** — `print('→')` crashes under cp1252.
  Use ASCII arrows `->` in any new print statements.
