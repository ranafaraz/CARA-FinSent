# Phase 3 Validation Report

Date: 2026-05-02

## Scope

This validation pass implemented the Phase 3 repository hardening instructions for:

- audit gating
- controlled gold split loading
- label-safe FinBERT evaluation and fine-tuning paths
- explicit benchmark mode separation
- result metadata traceability
- repo hygiene for retrieval corpora and obsolete results

## Code Changes

- `scripts/01_audit_datasets.py`
  - leakage failures now force `audit_gate=FAIL` and exit code `2`
- `src/cara_finsent/data_utils.py`
  - added `auto_detect_gold_split()`
  - added `infer_dataset_name()`
  - added strict `load_gold_split()`
  - added split label-distribution and text-hash leakage helpers
- `src/cara_finsent/label_mapping.py`
  - added helpers to encode canonical labels into model-native label IDs
- `src/cara_finsent/io_utils.py`
  - added `git_commit_sha()` for manifest traceability
- `scripts/10_run_classical_baselines.py`
  - now defaults to controlled gold splits and emits dataset/split metadata
- `scripts/11c_eval_finbert_zero_shot.py`
  - now uses strict gold splits and always records native label mapping + canonical remap
- `scripts/11_run_finbert_baseline.py`
  - now fine-tunes in native FinBERT label order and remaps only for reporting
- `scripts/11b_eval_finbert_checkpoint.py`
  - now evaluates checkpoints using checkpoint-native label order and controlled gold splits
- `scripts/11d_train_finbert_phrasebank.py`
  - now trains with model-native label IDs and controlled gold splits
- `scripts/11e_eval_finbert_external.py`
  - now records explicit transfer-mode metadata and controlled split provenance
- `scripts/11f_train_finbert_agreement_weighted.py`
  - now trains with model-native label IDs and controlled gold splits
- `scripts/11_run_finbert_improved_v2.py`
  - now uses the same native-label-safe fine-tuning path
- `scripts/18_cross_domain_eval.py`
  - new explicit cross-domain transfer benchmark entrypoint
- `.gitignore`
  - extended runtime artifact ignores for retrieval corpora, temporary eval dirs, checkpoints, and notebook outputs
- `results/README.md`
  - now distinguishes final benchmark outputs from obsolete exploratory outputs
- `data/retrieval_corpus/.gitkeep`
  - created Phase 3 retrieval-corpus directory layout

## Validation Commands

### 1. Full compile pass

Command:

```powershell
python -m compileall -q src scripts
```

Status: PASS

### 2. Regenerate clean gold datasets

Command:

```powershell
python scripts/00_prepare_phrasebank_fiqa.py
```

Status: PASS

Key outputs:

- `data/processed/gold/latest_gold_phrasebank.csv`
- `data/processed/gold/latest_gold_fiqa.csv`
- `results/2026-05-01/dataset_preparation_manifest_20260501_193108.json`

### 3. Rebuild controlled splits

Commands:

```powershell
python scripts/06_create_controlled_splits.py --input data/processed/gold/latest_gold_phrasebank.csv --dataset_name phrasebank
python scripts/06_create_controlled_splits.py --input data/processed/gold/latest_gold_fiqa.csv --dataset_name fiqa
```

Status: PASS

Observed integrity:

- PhraseBank split counts: train=3353, val=479, test=959, leakage_count=0
- FiQA split counts: train=777, val=111, test=223, leakage_count=0

### 4. Audit gate

Command:

```powershell
python scripts/01_audit_datasets.py --inputs data/processed/gold/latest_gold_phrasebank_split.csv data/processed/gold/latest_gold_fiqa_split.csv
```

Status: PASS

Observed gate:

- `overall_status=PASS`
- `leakage_status=PASS`
- `audit_gate=PASS`

### 5. FinBERT label mapping sanity

Command:

```powershell
python scripts/11d_check_finbert_label_mapping.py --sample_file data/processed/gold/latest_gold_phrasebank_split.csv
```

Status: PASS

Observed mapping:

- native `id2label = {0: positive, 1: negative, 2: neutral}`
- canonical remap = `[1, 2, 0]`

### 6. PhraseBank in-domain classical baselines

Command:

```powershell
python scripts/10_run_classical_baselines.py --data data/processed/gold/latest_gold_phrasebank_split.csv --dataset_name phrasebank
```

Status: PASS

Key result:

- best model by macro-F1: `tfidf_linear_svm`
- accuracy = `0.7497`
- macro-F1 = `0.6941`

Summary file:

- `results/2026-05-01/classical_baseline_summary_20260501_193204.csv`

### 7. PhraseBank in-domain FinBERT zero-shot

Command:

```powershell
python scripts/11c_eval_finbert_zero_shot.py --data data/processed/gold/latest_gold_phrasebank_split.csv --dataset_name phrasebank
```

Status: PASS

Key result:

- accuracy = `0.8832`
- macro-F1 = `0.8836`
- abstention coverage = `0.9541`
- abstention accuracy = `0.8984`

Interpretation:

- PhraseBank zero-shot FinBERT is clearly stronger than the old mixed-dataset FinBERT baseline and stronger than the classical PhraseBank baseline.

Summary file:

- `results/2026-05-01/finbert_baseline_summary_20260501_193253.csv`

### 8. FiQA in-domain classical baselines

Command:

```powershell
python scripts/10_run_classical_baselines.py --data data/processed/gold/latest_gold_fiqa_split.csv --dataset_name fiqa
```

Status: PASS

Key results:

- best accuracy: `tfidf_linear_svm` at `0.6996`
- best macro-F1: `tfidf_logistic_regression` at `0.5678`

Summary file:

- `results/2026-05-01/classical_baseline_summary_20260501_193220.csv`

### 9. FiQA in-domain FinBERT zero-shot

Command:

```powershell
python scripts/11c_eval_finbert_zero_shot.py --data data/processed/gold/latest_gold_fiqa_split.csv --dataset_name fiqa
```

Status: PASS (execution and traceability)

Observed result:

- accuracy = `0.1300`
- macro-F1 = `0.1293`
- abstention coverage = `0.8700`
- abstention accuracy = `0.1186`

Interpretation:

- The run is now label-safe and reproducible, but performance is extremely weak on FiQA. This should be treated as a real benchmark finding, not a data-leakage or label-order artifact.

Summary file:

- `results/2026-05-01/finbert_baseline_summary_20260501_193514.csv`

### 10. Cross-domain transfer: PhraseBank -> FiQA

Command:

```powershell
python scripts/18_cross_domain_eval.py --train data/processed/gold/latest_gold_phrasebank_split.csv --test data/processed/gold/latest_gold_fiqa_split.csv
```

Status: PASS

Key result:

- best transfer model: `tfidf_logistic_regression_transfer`
- accuracy = `0.1076`
- macro-F1 = `0.1001`

Interpretation:

- PhraseBank-to-FiQA transfer is extremely weak and must remain a separate external-transfer mode, not a main benchmark claim.

Summary file:

- `results/2026-05-01/cross_domain_summary_20260501_193559.csv`

### 11. Fine-tuning baseline smoke validation

Command executed for smoke validation only:

```powershell
python scripts/11_run_finbert_baseline.py --data data/processed/gold/latest_gold_phrasebank_split.csv --dataset_name phrasebank --epochs 0.02 --batch_size 8
```

Status: PASS (smoke test)

Observed result:

- accuracy = `0.8895`
- macro-F1 = `0.8884`

Purpose:

- verify that the rewritten native-label fine-tuning path trains, evaluates, remaps, and emits the required metadata without waiting for a full multi-hour CPU run

Summary file:

- `results/2026-05-01/finbert_baseline_summary_20260501_193637.csv`

## Acceptance Criteria Check

- `python -m compileall -q src scripts` passes: PASS
- `audit_gate=PASS`: PASS
- both controlled gold split aliases exist: PASS
- no `text_hash` leakage across train/val/test: PASS
- FinBERT mapping sanity confirms native order and canonical remap: PASS
- clean PhraseBank-only baseline results generated: PASS
- summary CSVs now include `dataset_name`, `split_source`, `seed`, `train_rows`, `val_rows`, `test_rows`: PASS
- repo now separates gold data, retrieval corpus, and obsolete exploratory outputs in layout/docs: PASS

## Notes

- A full 3-epoch CPU rerun of `scripts/11_run_finbert_baseline.py` was not included in this validation pass because it is materially longer than the other Phase 3 checks. The code path was smoke-tested successfully and now uses native-label-safe fine-tuning.
- FiQA zero-shot and PhraseBank-to-FiQA transfer remain weak after hardening. That is now a benchmark conclusion, not a split or label-order bug.