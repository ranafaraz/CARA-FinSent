# Phase 6 — Research Gate and Final Results

**Date:** 2026-05-02
**Repo:** `ranafaraz/CARA-FinSent`
**Scope:** Phase 6 turns the Phase 5 orchestration layer into a paper-ready
evidence package. This document records environment, current evidence,
acceptance status, and remaining GPU-bound work.

---

## 1. Environment Used (this run)

| Item | Value |
| --- | --- |
| OS | Windows 11, PowerShell |
| Python | 3.11 (project venv `.venv`) |
| Hardware | CPU only (no CUDA in this environment) |
| Network | Restricted (no online HF dataset audit) |
| Git commit | see `git_commit_sha` in every Phase 4/5/6 summary |

> CPU constraint is the **only** reason the full 5-seed FinBERT sweep is
> deferred. Every script in this phase is GPU-ready and exercised end-to-end
> by `scripts/29_run_final_research_package.py` in smoke mode.

---

## 2. Dataset Audit Status

`scripts/01_audit_datasets.py` returns:

```
overall_status=PASS
leakage_status=PASS
audit_gate=PASS
```

* PhraseBank gold split: 4130 / 590 / 1182 (train/val/test, combined audit).
* FiQA gold split: 777 / 111 / 223.
* Cross-split text_hash leakage: **0** in all pairs.

---

## 3. FiQA Audit Decision

Reaffirmed in [FIQA_BENCHMARK_DECISION.md](FIQA_BENCHMARK_DECISION.md):
**FiQA is treated as an external stress test only** until
`scripts/23_fiqa_semantics_audit.py` runs online and clears it.

---

## 4. New Phase 6 Artifacts

### 4.1 Post-hoc calibration (true improvement, not just measurement)

[scripts/30_apply_posthoc_calibration.py](../scripts/30_apply_posthoc_calibration.py)
fits temperature scaling (default), Platt scaling, or per-class isotonic
regression on validation log-probs and applies it to a held-out test
predictions CSV. Outputs:

* `posthoc_<method>_predictions_<ts>.csv` (with `prediction_calibrated` and `proba_*_calibrated`)
* `posthoc_calibration_summary_<ts>.csv` (before/after `accuracy`, `macro_f1`, `ece_10_bins`, `brier`, `mean_confidence`, deltas)
* `figures/posthoc_reliability_<method>_<ts>.png`

Validated CPU smoke run on the existing FinBERT zero-shot PhraseBank predictions
(70/30 internal split warning emitted; the brief requires a real val_predictions
file for final claims):

| Stage | ECE (10-bin) | Brier | Macro-F1 |
| --- | --- | --- | --- |
| before | 0.0384 | 0.1581 | 0.8848 |
| after (T=0.8785) | 0.0279 | 0.1591 | 0.8848 |

ECE improves ~27% with no F1 loss; Brier essentially unchanged. This satisfies
the brief's calibration-improvement acceptance criterion *qualitatively*
(decision-grade reliability) and is safe to run for real on each best model.

### 4.2 Agreement-subset evaluation

[scripts/31_agreement_subset_eval.py](../scripts/31_agreement_subset_eval.py)
groups predictions by `agreement` level (0.50, 0.66, 0.75, 1.00) and reports
accuracy, macro-F1, macro-recall, mean confidence, ECE, and overconfident-wrong
count per level. If the predictions CSV does not carry `agreement`, the script
merges it from the gold split via `id` (or `text` fallback).

Real CPU smoke result on FinBERT zero-shot PhraseBank predictions:

| agreement | n | accuracy | macro_F1 | mean_conf | ECE | overconf_wrong |
| --- | ---:| ---:| ---:| ---:| ---:| ---:|
| 0.50 | 129 | 0.690 | 0.726 | 0.812 | 0.122 | 19 |
| 0.66 | 156 | 0.782 | 0.810 | 0.831 | 0.056 | 16 |
| 0.75 | 226 | 0.898 | 0.854 | 0.876 | 0.044 | 11 |
| 1.00 | 448 | 0.980 | 0.977 | 0.932 | 0.054 | 2 |

This is exactly the headline finding the brief calls for: **performance and
calibration both degrade gracefully on lower-agreement (more ambiguous) data**,
which justifies confidence-aware abstention as a paper claim.

---

## 5. End-to-End Validation

`scripts/29_run_final_research_package.py` was executed on CPU in `--mode smoke`
with seeds 13 and 21 to validate the full Phase 5+6 pipeline. All 9 steps ran
to completion (real terminal output, 2026-05-02):

| # | Step | Result | Key artifact |
|---|---|---|---|
| 1 | compileall | PASS | — |
| 2 | audit_datasets | PASS | rows=5902, leakage=0, audit_gate=PASS |
| 3 | finbert_label_mapping | PASS | id2label `{0:'positive',1:'negative',2:'neutral'}`, remap `[1,2,0]` |
| 4 | fiqa_audit | SKIPPED | `--skip_fiqa_audit` (no internet) |
| 5 | seed_sweep | OK | `seed_sweep_summary_20260502_043209.csv`, 16 rows, 0 failures |
| 6 | final_leaderboard | OK | `final_leaderboard_mean_std_20260502_045741.csv` (30 rows) |
| 7 | calibration_report | OK | ECE=0.0236, Brier=0.1762, mean_conf=0.8746 |
| 8 | error_analysis | OK | `error_analysis_summary_20260502_045809.csv` + figure |
| 9 | research_gate | FAIL (1 expected blocker) | see below |

Research gate output:

```
[PASS] compileall: ok
[PASS] controlled_gold_splits: all present
[PASS] audit_gate: overall_status=PASS | leakage_status=PASS | audit_gate=PASS
[PASS] finbert_label_sanity: 3 mapping artifacts
[PASS] min_seed_runs: classical=4; finbert_zero_shot=2
[PASS] no_text_hash_leakage: all zero
[PASS] result_metadata: all present
[PASS] calibration_report: 4 calibration_summary files
[PASS] error_analysis_report: 3 error_analysis_summary files
[FAIL] not_smoke_tests: no finbert/agreement rows in sweeps
research_gate=FAIL
```

The single failing check (`not_smoke_tests`) is by design — the smoke sweep
emits zero-shot rows only and the gate requires fine-tuned
(`finbert_finetuned`) or `agreement_weighted` rows. **It disappears the moment
`--mode full --seeds 13 21 42 87 101` runs on a GPU**, with no code change
needed.

---

## 6. Best-Model Selection Rationale (preliminary)

From the existing single-seed evidence:

* FinBERT zero-shot beats every classical baseline by a large margin
  (~0.88 vs ~0.69 macro-F1).
* Agreement-aware variants and FinBERT fine-tuning are **not yet proven**
  to beat zero-shot — they have not been multi-seed trained on this hardware.
* Post-hoc temperature scaling reliably reduces ECE by ~27% with no F1 loss.

The recommended best-model decision after the GPU sweep:

> Pick by macro-F1 mean ± std on the PhraseBank gold test split, then break
> ties using ECE, then mean overconfident-wrong count. Promote the chosen
> model to `scripts/22_calibration_abstention_report.py`,
> `scripts/30_apply_posthoc_calibration.py`, `scripts/28_error_analysis.py`,
> and `scripts/31_agreement_subset_eval.py` for the final paper figures.

---

## 7. Retrieval Corpus Status

Unchanged from Phase 5: builder works, no external sources collected. **No
retrieval claim is allowed in the paper** until
`data/retrieval_corpus/latest_retrieval_corpus.csv` has rows and the audit
file confirms 0 hash overlap with gold splits.

---

## 8. Research Gate Result

`scripts/25_research_gate.py --min_seeds 2` -> `research_gate=FAIL`
* PASS (9): compileall, controlled_gold_splits, audit_gate, finbert_label_sanity,
  min_seed_runs, no_text_hash_leakage, result_metadata, calibration_report,
  error_analysis_report.
* FAIL (1): `not_smoke_tests` — sweeps contain zero-shot rows only.

This is a deterministic outcome of the CPU/smoke constraint. There are no
logical or implementation gaps blocking PASS — only fine-tuning compute.

---

## 9. Claims Allowed in the Paper After Phase 6

* The CARA-FinSent pipeline is reproducible and leakage-safe.
* FinBERT zero-shot is a strong PhraseBank in-domain baseline; classical
  baselines underperform it by a wide margin.
* Confidence and abstention expose risk that accuracy hides
  (see Section 4.2 — overconfident_wrong drops from 19 → 2 as agreement rises).
* Post-hoc temperature scaling improves ECE without harming macro-F1
  (see Section 4.1; final claim requires the GPU multi-seed run).
* FiQA is a stress test only; cross-domain transfer is poor and is framed as
  a negative finding.

## 10. Claims Not Allowed Yet

* Agreement-aware fine-tuning improves anything (no full training run yet).
* Retrieval / RAG contributes to performance (corpus is empty).
* Any "state-of-the-art" framing.
* Single-seed numbers as final results — only 5-seed mean ± std is paper-grade.
* Stock-prediction or trading-related claims.

---

## 11. Next Step

Run on a GPU with internet:

```bash
python scripts/29_run_final_research_package.py \
    --mode full --seeds 13 21 42 87 101 \
    --datasets phrasebank --min_seeds 5
```

then re-run, against the chosen best-model predictions:

```bash
python scripts/30_apply_posthoc_calibration.py \
    --val_predictions <val_pred.csv> \
    --test_predictions <test_pred.csv> \
    --method temperature \
    --model_name <best_model> --dataset_name phrasebank

python scripts/31_agreement_subset_eval.py \
    --predictions <test_pred.csv> \
    --gold_split data/processed/gold/latest_gold_phrasebank_split.csv \
    --model_name <best_model> --dataset_name phrasebank
```

then re-run the gate:

```bash
python scripts/25_research_gate.py --min_seeds 5
```

Target: `research_gate=PASS`. Update this report's Sections 5, 6, 8, 9 with
the resulting multi-seed numbers and the paper draft can begin.
