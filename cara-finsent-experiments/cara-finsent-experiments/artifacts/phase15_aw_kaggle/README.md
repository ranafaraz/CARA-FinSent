# Phase 15 — Kaggle GPU AW seed=13 rerun (PREPARE-ONLY)

**Status:** Prepared, not submitted (per Rana's instruction `prepare_only`).

## What this directory contains

- `kernel-metadata.json` — Kaggle kernel manifest. Replace `USERNAME` with your Kaggle handle before pushing.
- `phase15_aw_seed13_kaggle.ipynb` — single-cell notebook that:
  1. Installs `transformers==4.49.0` and `datasets`.
  2. Uploads / fetches the controlled gold split (Phase 14 canonical CSVs).
  3. Re-runs `scripts/11f_train_finbert_agreement_weighted.py --seed 13 --weight_schedule linear --num_epochs 3 --batch_size 16` on a Kaggle T4 GPU.
  4. Evaluates the val-best checkpoint on the canonical PhraseBank test split.
  5. Prints test macro-F1 and accuracy and writes `phase15_aw_gpu_repro_<timestamp>.csv`.

## Inputs the notebook expects

You must attach (as Kaggle dataset) the following files from this repo:
- `data/processed/gold/latest_gold_train.csv`
- `data/processed/gold/latest_gold_val.csv`
- `data/processed/gold/latest_gold_test.csv`
- `scripts/11f_train_finbert_agreement_weighted.py`

A pre-bundled tarball is sufficient.

## Submission steps (manual)

```bash
# 1. Edit kernel-metadata.json — set "id" to "<your-kaggle-username>/cara-finsent-phase15-aw-seed13"
# 2. From this directory:
kaggle kernels push -p .
# 3. Wait for completion (T4, ~25 min). Then:
kaggle kernels output <your-kaggle-username>/cara-finsent-phase15-aw-seed13 -p ../../results/2026-05-02/phase15_aw_gpu_repro/
```

## Pass / fail criterion

- **PASS:** test macro-F1 ≥ 0.8817 (lower bound of Phase 11 envelope `0.8863 ± 0.0023`).
  Action: keep Phase 11 multi-seed AW number as paper headline; CPU result stays as appendix footnote.
- **FAIL:** test macro-F1 < 0.8817.
  Action: **STOP** and notify Rana before any further paper revision; do not auto-fall-back to RunPod.
