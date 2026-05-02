# Phase 7 — Final Experiment Validation Report

**Date:** 2026-05-02
**Repo:** `ranafaraz/CARA-FinSent`
**Git commit SHA:** `295188fceca763f3efa7f6fb23246e8525575f82`
**Branch:** `main`
**Status:** **PARTIAL — paper-grade evidence pending GPU run.**

---

## 0. Executive Summary

Phase 7 is an **execution phase**: the brief explicitly states *"The next phase
should not add many new features. It should produce final evidence."* and
mandates a **GPU machine** with `--mode full --seeds 13 21 42 87 101`, real
fine-tuning epochs for `finbert_finetuned` and `agreement_weighted`.

This session was executed on the same **CPU-only Windows box** as Phases 4–6
(no CUDA, no online HF dataset audit). The Phase 7 acceptance checklist
therefore **cannot be satisfied here**. What was done:

1. Re-ran every preflight check the brief requires (Steps 0, 1, 2 entry
   points) — all PASS.
2. Verified that the orchestrator (`scripts/29_run_final_research_package.py`)
   and every downstream script are wired correctly and run end-to-end (proven
   in Phase 6's CPU smoke run; nothing has changed since).
3. Documented the exact GPU command, expected artifacts, and acceptance
   criteria so the paper-grade run is a one-line invocation.
4. Stated honestly which Phase 7 acceptance items are *deferred* vs *blocked
   by hardware* vs *truly missing*. Nothing is silently missing.

This report follows the Step-9 structure mandated by the brief.

---

## 1. Environment

| Item | Value |
| --- | --- |
| OS | Windows 11, PowerShell 7 |
| Python | 3.11 (project venv `.venv`) |
| Hardware | CPU only — no CUDA accelerator detected |
| RAM | sufficient (no OOMs in any prior run) |
| Network | restricted (no online HF dataset audit) |
| Repo branch | `main` |
| Git commit SHA | `295188fceca763f3efa7f6fb23246e8525575f82` |

The brief mandates Python 3.10/3.11, CUDA-enabled GPU, ≥16 GB RAM, and
internet access. **Only the Python version condition is met here.**

---

## 2. Preflight (Brief: "Before running experiments")

Run on 2026-05-02:

```text
python -m compileall -q src scripts            -> exit 0       PASS
python scripts/01_audit_datasets.py ...        -> audit_gate=PASS
                                                  rows_total=5902
                                                  leakage_total=0
                                                  splits 4130/590/1182
python scripts/11d_check_finbert_label_mapping.py
                                               -> id2label {0:'positive',
                                                            1:'negative',
                                                            2:'neutral'}
                                                  canonical remap [1, 2, 0]
                                                  sanity sample written to
                                                  data/audit/2026-05-02/
                                                  finbert_label_sanity_
                                                  20260502_053257.csv
```

All three preflight gates PASS. The repo is shovel-ready for the GPU run.

---

## 3. Step 1 — FiQA Semantics Audit

**Status: DEFERRED (no internet on this host).**

The decision in [FIQA_BENCHMARK_DECISION.md](FIQA_BENCHMARK_DECISION.md) is
unchanged: **FiQA is treated as an external stress test only**. To revisit:

```bash
python scripts/23_fiqa_semantics_audit.py
```

then update [FIQA_BENCHMARK_DECISION.md](FIQA_BENCHMARK_DECISION.md) per
brief's decision rule. No paper claim depends on this resolving favorably.

---

## 4. Step 2 — Final Five-Seed PhraseBank Sweep

**Status: BLOCKED (requires GPU). Command is queued and validated.**

Brief command:

```bash
python scripts/29_run_final_research_package.py \
    --mode full \
    --seeds 13 21 42 87 101 \
    --datasets phrasebank \
    --min_seeds 5 \
    --skip_fiqa_audit
```

`--mode full` resolves to:
`classical,finbert_zero_shot,finbert_finetuned,agreement_weighted`
(see [scripts/26_run_seed_sweep.py](../scripts/26_run_seed_sweep.py) line 28).

Default fine-tuning hyper-parameters:
- `--finbert_epochs 3`
- `--finbert_batch_size 16`
- `--weight_schedule linear`

Estimated wall clock on a single L4/A10 GPU: 60–120 minutes.
Estimated wall clock on this CPU box: 8–12+ hours, with high risk of partial
failure, so this run is **not attempted here**.

**No `finbert_finetuned*.csv` or `finbert_agreement_weighted_summary*.csv`
artifacts exist in `results/` yet.** This is the central remaining gap for
paper readiness.

---

## 5. Steps 3–4 — Current Seed-Sweep & Leaderboard (zero-shot only)

**Status: PARTIAL.** A 2-seed (13, 21) zero-shot + classical sweep was
executed in Phase 6. It is **not paper-grade** but proves every aggregator
works.

Latest leaderboard:
[results/2026-05-02/final_leaderboard_mean_std_20260502_045741.csv](../results/2026-05-02/final_leaderboard_mean_std_20260502_045741.csv)

| experiment | model | n_seeds | mean_acc | mean_F1 | mean_ECE | mean_Brier | research_grade |
|---|---|---:|---:|---:|---:|---:|:---:|
| finbert_zero_shot | finbert_base_zero_shot | 2 | 0.883 | 0.884 | 0.024 | 0.176 | True |
| classical | tfidf_linear_svm | 4 | 0.750 | 0.694 | – | – | True |
| classical | tfidf_logistic_regression | 4 | 0.740 | 0.684 | 0.173 | 0.425 | True |
| classical | tfidf_sgd_log_loss | 4 | 0.747 | 0.677 | 0.098 | 0.378 | True |
| classical | tfidf_xgboost | 4 | 0.721 | 0.622 | 0.032 | 0.393 | True |
| classical | tfidf_random_forest | 4 | 0.717 | 0.589 | 0.043 | 0.395 | True |
| classical | tfidf_multinomial_nb | 4 | 0.652 | 0.394 | 0.106 | 0.469 | True |
| classical | majority_baseline | 4 | 0.599 | 0.250 | 0.401 | 0.803 | True |

Brief required-column audit on the latest seed sweep summary
([results/2026-05-02/seed_sweep_summary_20260502_043209.csv](../results/2026-05-02/seed_sweep_summary_20260502_043209.csv)):

| Required column | Present |
| --- | :---: |
| model, experiment, dataset_name, benchmark_mode, seed | ✅ |
| accuracy, macro_f1, weighted_f1, mcc | ✅ |
| ece_10_bins, brier_score, mean_confidence | ✅ |
| train_rows, val_rows, test_rows | ✅ |
| text_hash_leakage_count | ✅ (all = 0) |
| git_commit_sha | ✅ |
| summary_file | ✅ |
| 5 unique seeds per experiment | ❌ (only 2 today) |
| `finbert_finetuned`, `agreement_weighted` rows | ❌ (not yet run) |
| `epochs / num_epochs` indicating real training | n/a until 4. runs |

---

## 6. Steps 5–7 — Best-Model Selection, Calibration, Error Analysis

**Status: PROVISIONAL on the only multi-seed model available
(`finbert_zero_shot`).** All three scripts ran cleanly in Phase 6:

* [results/2026-05-02/calibration_summary_20260502_045753.csv](../results/2026-05-02/calibration_summary_20260502_045753.csv) — ECE=0.0236, Brier=0.1762, mean_conf=0.8746
* [results/2026-05-02/error_analysis_summary_20260502_045809.csv](../results/2026-05-02/error_analysis_summary_20260502_045809.csv) + [error_examples_20260502_045809.csv](../results/2026-05-02/error_examples_20260502_045809.csv)
* Phase 6 also ran [scripts/30_apply_posthoc_calibration.py](../scripts/30_apply_posthoc_calibration.py) (T=0.8785, ECE 0.0384→0.0279) and [scripts/31_agreement_subset_eval.py](../scripts/31_agreement_subset_eval.py) (graceful degradation 0.977→0.726 macro-F1 as agreement drops 1.00→0.50).

After the GPU run, repeat each of these on the actual best model:

```bash
python scripts/22_calibration_abstention_report.py \
    --predictions <best_predictions.csv> \
    --model_name <best_model_name> --dataset_name phrasebank

python scripts/28_error_analysis.py \
    --predictions <best_predictions.csv> \
    --model_name <best_model_name> --dataset_name phrasebank
```

Best-model selection rule (per brief Step 5): highest mean macro-F1, then
lower ECE, then lower Brier, then lower overconfident-wrong, then simpler
model. Pick the seed closest to the mean for the prediction CSV passed to the
two scripts above.

---

## 7. Step 8 — Research Gate

**Latest result (2026-05-02):** `research_gate=FAIL`

[results/2026-05-02/research_gate_report_20260502_045816.csv](../results/2026-05-02/research_gate_report_20260502_045816.csv)

| Check | Status |
| --- | :---: |
| compileall | PASS |
| controlled_gold_splits | PASS |
| audit_gate | PASS |
| finbert_label_sanity | PASS |
| min_seed_runs (≥2 today) | PASS |
| no_text_hash_leakage | PASS |
| result_metadata | PASS |
| calibration_report | PASS |
| error_analysis_report | PASS |
| **not_smoke_tests** | **FAIL** — no `finbert_finetuned` / `agreement_weighted` rows in any sweep |

A single failing check, exactly the one Phase 7 was created to close. After
the GPU run with `--mode full --seeds 13 21 42 87 101` and `--min_seeds 5`,
this check flips to PASS automatically because the sweep summary will then
contain non-zero-shot, non-smoke fine-tuned rows.

---

## 8. Step 9 — Honest Claim Boundaries

### What we **can** claim now

* The CARA-FinSent pipeline is reproducible, leakage-safe, and clears every
  data and label sanity gate (`audit_gate=PASS`, leakage=0, FinBERT remap
  verified end-to-end).
* On PhraseBank in-domain, FinBERT zero-shot beats every classical baseline
  by ~19 macro-F1 points (0.884 vs ≤0.694) — **single-pair seed average,
  not yet 5-seed**.
* Confidence and abstention expose risk that accuracy hides: overconfident
  wrong predictions drop from 19 → 2 as PhraseBank annotator agreement rises
  from 0.50 → 1.00 (Phase 6, [scripts/31_agreement_subset_eval.py](../scripts/31_agreement_subset_eval.py)).
* Post-hoc temperature scaling reduces ECE ~27 % with no F1 loss — qualitative
  evidence; final claim still requires the GPU multi-seed run.
* FiQA cross-domain transfer is poor and is framed as a negative finding
  (stress test, not benchmark).

### What we **cannot** claim yet

* Any number for fine-tuned FinBERT (`finbert_finetuned`) — never trained.
* Any number for agreement-weighted FinBERT (`agreement_weighted`) — never trained.
* Any "best model" identity beyond `finbert_zero_shot` — fine-tunes have not been observed.
* 5-seed mean ± std for any FinBERT variant.
* Retrieval / RAG contributions (corpus is empty; brief Step 0 still applies).
* "State of the art" framing.
* Stock-prediction or trading-related claims.

### What remains future work (post-GPU run)

* Full 5-seed `--mode full` PhraseBank sweep (Step 2).
* FiQA online semantics audit + benchmark-tier decision update (Step 1).
* Real-validation post-hoc calibration of best model (Phase 6 script 30 with
  a true `--val_predictions` instead of the 70/30 internal split).
* Build non-empty external retrieval corpus before any retrieval claim.

---

## 9. Step 2 — GPU Hand-off (one-shot)

The repo is configured so that on a GPU box the entire Phase 7 run is:

```bash
git pull
python -m pip install -r requirements.txt
python -m compileall -q src scripts
python scripts/01_audit_datasets.py \
    --inputs data/processed/gold/latest_gold_phrasebank_split.csv \
             data/processed/gold/latest_gold_fiqa_split.csv
python scripts/11d_check_finbert_label_mapping.py \
    --sample_file data/processed/gold/latest_gold_phrasebank_split.csv

# (optional, if internet) the FiQA decision:
python scripts/23_fiqa_semantics_audit.py

# the headline run:
python scripts/29_run_final_research_package.py \
    --mode full \
    --seeds 13 21 42 87 101 \
    --datasets phrasebank \
    --min_seeds 5 \
    --skip_fiqa_audit

# best-model post-hoc calibration on the new artifacts:
python scripts/30_apply_posthoc_calibration.py \
    --val_predictions <best_val_predictions.csv> \
    --test_predictions <best_test_predictions.csv> \
    --method temperature \
    --model_name <best_model_name> --dataset_name phrasebank

python scripts/31_agreement_subset_eval.py \
    --predictions <best_test_predictions.csv> \
    --gold_split data/processed/gold/latest_gold_phrasebank_split.csv \
    --model_name <best_model_name> --dataset_name phrasebank

# verify:
python scripts/25_research_gate.py --min_seeds 5
```

Target: `research_gate=PASS`.

---

## 10. Phase 7 Acceptance Checklist (current state)

- [x] `python -m compileall -q src scripts` passes.
- [x] `audit_gate=PASS`.
- [x] FinBERT label mapping sanity passes.
- [ ] FiQA audit has been run or its omission is documented. *(documented; deferred)*
- [ ] `seed_sweep_summary_*.csv` contains at least 5 seeds for final PhraseBank experiments. *(2 seeds today)*
- [x] `final_leaderboard_mean_std_*.csv` exists.
- [x] Final leaderboard has `research_grade=True` rows.
- [ ] Fine-tuned FinBERT is not smoke-only. *(no fine-tuned rows yet; GPU pending)*
- [ ] Agreement-weighted FinBERT is not smoke-only. *(no rows yet; GPU pending)*
- [x] Calibration summary exists for current best model (zero-shot).
- [x] Abstention curve exists for current best model.
- [x] Error analysis exists for current best model.
- [ ] `research_gate=PASS`. *(FAILs on `not_smoke_tests` only)*
- [x] `docs/PHASE7_FINAL_EXPERIMENT_VALIDATION_REPORT.md` exists.

**Final state:** 8 of 13 acceptance items satisfied; the remaining 5 all
collapse into a single GPU run of Step 4.

---

## 11. Suggested Commit Message

```text
phase7: preflight validation + Phase 7 evidence report (GPU run pending)
```

Files to add:
* `docs/PHASE7_FINAL_EXPERIMENT_VALIDATION_REPORT.md` (this file)
* `data/audit/2026-05-02/finbert_label_sanity_20260502_053257.csv`

(No model checkpoints, `.bin`, `.pt`, `.safetensors`, raw HF caches, or
proprietary data are produced or staged.)
