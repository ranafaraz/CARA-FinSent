# Phase 16 — Kaggle AW Package Audit

**Audit timestamp (UTC):** 2026-05-03

## Phase 15 prepare-only package — review

Inspected:

- `artifacts/phase15_aw_kaggle/kernel-metadata.json`
- `artifacts/phase15_aw_kaggle/phase15_aw_seed13_kaggle.ipynb`
- `artifacts/phase15_aw_kaggle/README.md`
- `results/2026-05-02/phase15_aw_gpu_repro/STATUS.yml`

The Phase 15 notebook was minimal: it called `scripts/11f_train_finbert_agreement_weighted.py` and dumped the resulting summary CSV. **It did not save val predictions, did not run the script-41 path with the AW-specific tokenizer fallback, and did not emit a Phase-16-grade manifest with split SHA-256 / GPU name / pass-gate flag.**

Phase 16 therefore replaces the Phase 15 stub with a **stronger** package staged at `artifacts/phase16_aw_kaggle/`:

```
artifacts/phase16_aw_kaggle/
  dataset/                                       # uploaded as Kaggle dataset
    dataset-metadata.json                        # ranafarazahmed/cara-finsent-phase16-aw-bundle
    data/processed/gold/latest_gold_phrasebank_split.csv      (1.32 MB, controlled gold split)
    data/processed/gold/latest_gold_fiqa_split_polarity_corrected.csv
    src/cara_finsent/*.py                        (11 modules — same code that produced Phase 11/13/14)
    scripts/11f_train_finbert_agreement_weighted.py
    scripts/41_generate_val_test_predictions.py
  kernel/
    kernel-metadata.json                         # ranafarazahmed/cara-finsent-phase16-aw-seed13, GPU on, internet on, dataset attached
    phase16_aw_seed13.ipynb                      # 9-cell notebook (see audit answers below)
```

## Audit answers (per Phase 16 spec §Task 2)

| # | Question | Answer |
|---|---|---|
| 1 | Does the notebook/script exist? | ✅ `artifacts/phase16_aw_kaggle/kernel/phase16_aw_seed13.ipynb` |
| 2 | Does it train only AW seed 13? | ✅ Cell 4 invokes `11f_train_finbert_agreement_weighted.py --seed 13 --weight_schedule linear --num_epochs 3 --batch_size 16` only — no other seed / no other variant |
| 3 | Does it use the controlled PhraseBank train/val/test split? | ✅ `data/processed/gold/latest_gold_phrasebank_split.csv` (3353 / 479 / 959, byte-identical to Phase 11/13/14 — verified by SHA-256 in the manifest) |
| 4 | Does it use the same label remap as Phase 11/13/14? | ✅ `11f` reads native `id2label = {0:positive, 1:negative, 2:neutral}` and applies canonical remap `[1, 2, 0]` → `[negative, neutral, positive]` (logged in manifest) |
| 5 | Does it use the same hyperparameters? | ✅ epochs=3, batch=16, LR=2e-5, warmup=100, max_len=128, weight_schedule=linear, base=ProsusAI/finbert, `load_best_model_at_end=True` on val macro_f1 |
| 6 | Does it save val/test predictions? | ✅ Cell 5 picks val-best checkpoint via `trainer_state.json`; cell 6 invokes `41_generate_val_test_predictions.py` which writes `*_val_predictions_*.csv` and `*_test_predictions_*.csv` (with proba columns); cell 7 copies them out as `phase16_aw_seed13_{val,test}_predictions.csv` |
| 7 | Does it save manifest and package versions? | ✅ Cell 3 writes `phase16_aw_seed13_environment.txt` (Python / torch / cuda / GPU name + full `pip freeze`); cell 9 writes `phase16_aw_seed13_manifest.json` containing all spec fields (kernel slug, GPU name, seed, base model, split SHA-256, label remap, hyperparams, val/test macro-F1, gate flag, timestamp) |
| 8 | Does it avoid touching test data for fitting calibration / threshold / ensemble weights? | ✅ Phase 16 deliberately does **no** calibration, threshold tuning, or ensemble weighting. The notebook only trains AW (with val-driven checkpoint selection on val macro-F1) and emits raw val + test predictions for downstream val-fit / test-eval consumption back on the laptop. |

## Verdict

The Phase 16 Kaggle package is reviewer-defensible and meets every Task-2 requirement. Ready to submit (Task 3).
