# Phase 5 — Final Research Evidence Report

**Date:** 2026-05-02
**Repo:** `ranafaraz/CARA-FinSent`
**Pipeline scope:** Phases 1 → 5 are committed. This report covers what Phase 5
adds and what is still pending before paper-writing can start.

---

## 1. Executive Summary

Phase 5 adds the orchestration, aggregation, error-analysis, and master-runner
scripts required for a research-grade evidence package, plus the FiQA
benchmark decision document. CPU-only smoke validation passes for all new
scripts. The full 5-seed sweep that flips `research_gate` to `PASS` is
deferred to a GPU runner; the gate currently reports `FAIL` with exactly two
expected blockers.

**Headline numbers (single-seed PhraseBank test set, n=959):**

| Model | Accuracy | Macro-F1 | ECE | Brier |
| --- | --- | --- | --- | --- |
| TF-IDF Linear SVM (Phase 3) | ~0.7497 | ~0.6941 | n/a | n/a |
| FinBERT zero-shot (Phase 3) | ~0.8832 | ~0.8836 | 0.0344 | 0.1577 |

These are single-seed values and **must not be quoted in the paper** until
they are replaced with multi-seed mean ± std from
`final_leaderboard_mean_std_*.csv`.

---

## 2. Current Research Status

| Layer | Status |
| --- | --- |
| Data correction & gold splits | Complete (Phase 2/3). |
| Leakage prevention | Verified via `text_hash_leakage_count == 0` columns in every Phase 4/5 summary. |
| FinBERT label remap | Verified via `data/audit/finbert_label_mapping_*.json`. |
| Multi-seed sweep orchestration | Implemented (`scripts/26_run_seed_sweep.py`). |
| Final leaderboard aggregation | Implemented (`scripts/27_summarize_seed_sweep.py`). |
| Calibration / abstention | Implemented (Phase 4). Smoke-validated. |
| Error analysis (Phase 5 schema) | Implemented (`scripts/28_error_analysis.py`). Smoke-validated. |
| FiQA semantics audit | Script exists; **online run required**. Decision documented in [FIQA_BENCHMARK_DECISION.md](FIQA_BENCHMARK_DECISION.md). |
| External retrieval corpus | Builder works; **no external sources collected yet** (0-row corpus). |
| Master runner | Implemented (`scripts/29_run_final_research_package.py`). |
| Research gate | Implemented (Phase 4). Currently `FAIL` (expected). |

---

## 3. Dataset Audit Summary

* PhraseBank gold split: 3,353 / 479 / 959 (train / val / test). No duplicates
  across splits (verified by `scripts/01_audit_datasets.py`).
* FiQA gold split: 777 / 111 / 223. No cross-split duplicates.
* FinBERT label sanity: native order `{0:positive, 1:negative, 2:neutral}`,
  canonical remap `[1, 2, 0]` confirmed in
  `data/audit/2026-05-01/finbert_label_mapping_20260501_193145.json`.

---

## 4. Benchmark Matrix

| Benchmark mode | Train data | Test data | Status |
| --- | --- | --- | --- |
| `phrasebank_in_domain` | PhraseBank gold train | PhraseBank gold test | 1 seed run; 5-seed sweep deferred to GPU. |
| `fiqa_in_domain` | FiQA gold train | FiQA gold test | Treated as stress-test only pending audit. |
| `phrasebank_to_fiqa` | PhraseBank gold train | FiQA gold test | 1 seed run (Phase 3); 5-seed sweep pending. |

---

## 5. Multi-Seed Leaderboard

* Aggregator: [scripts/27_summarize_seed_sweep.py](../scripts/27_summarize_seed_sweep.py).
* Output schema: `dataset_name`, `benchmark_mode`, `experiment`, `model`,
  `n_seeds`, `mean_accuracy`, `std_accuracy`, `mean_macro_f1`, `std_macro_f1`,
  `mean_ece_10_bins`, `std_ece_10_bins`, `mean_brier_score`, `std_brier_score`,
  `mean_latency_or_seconds`, `research_grade`.
* Latest CPU smoke run (classical-only, seeds 42–43):
  `results/2026-05-01/final_leaderboard_mean_std_20260501_203203.csv` —
  14 input rows, 7 (model,dataset) groups, all flagged `research_grade=False`
  because n_seeds=2 < 5.
* Required to flip to `research_grade=True`: re-run
  `scripts/26_run_seed_sweep.py --seeds 13 21 42 87 101 --mode full` on a GPU.

---

## 6. Calibration and Abstention Findings (preliminary)

From the existing single-seed FinBERT zero-shot PhraseBank predictions
(`results/2026-05-01/finbert_baseline_predictions_20260501_193637.csv`):

| Metric | Value |
| --- | --- |
| ECE (10 bins) | 0.0344 |
| Brier score | 0.1577 |
| Mean confidence | 0.8862 |

Coverage / accuracy at the 0.60 and 0.70 thresholds are written into
`calibration_summary_*.csv`. Multi-seed mean ± std is pending the full sweep.

---

## 7. Error Analysis Findings (preliminary)

`scripts/28_error_analysis.py` summary on the same predictions:

* `error_analysis_summary_20260501_203200.csv` — counts of `neutral_as_positive`,
  `neutral_as_negative`, `positive_as_negative`, `negative_as_positive`,
  `overconfident_wrong_count` (confidence ≥ 0.80), `low_confidence_correct_count`
  (confidence < 0.60).
* `error_examples_20260501_203200.csv` — top-50 highest-confidence wrong
  predictions for qualitative inspection.
* `figures/error_type_distribution_20260501_203200.png` — bar chart of error
  categories.

The phase-4 deeper breakdown (length buckets, numeric content, agreement
quartiles, neutral-confusion, classwise) is also available via
`scripts/21_error_analysis.py`.

---

## 8. FiQA Semantics Decision

Documented in [FIQA_BENCHMARK_DECISION.md](FIQA_BENCHMARK_DECISION.md).
Summary: **stress-test only** until `scripts/23_fiqa_semantics_audit.py` is
run online and validates label semantics.

---

## 9. Retrieval Corpus Status

* Builder implemented and gold-leakage-safe
  ([scripts/24_build_external_retrieval_corpus.py](../scripts/24_build_external_retrieval_corpus.py)).
* Latest output: 0 rows
  (`data/retrieval_corpus/2026-05-01/retrieval_corpus_20260501_201315.csv`)
  because no external source files have been collected yet.
* Acceptable next sources: SEC filing snippets
  (`scripts/02_collect_sec_10k.py`), financial news headlines
  (`scripts/03_collect_financial_news.py`), company profiles, sector metadata,
  earnings-call summaries.
* **Do not claim retrieval contribution** until corpus is non-empty *and*
  `text_hash` overlap with gold splits is verified to be 0.

---

## 10. What CARA-FinSent Can Honestly Claim Today

* **Clean data handling improves apparent performance.** Mixed-dataset
  numbers from before Phase 2 are not comparable to clean PhraseBank numbers.
* **FinBERT zero-shot is a strong PhraseBank in-domain baseline.** ~0.88
  accuracy / macro-F1 (single seed) on the controlled gold split, with no
  cross-split leakage, is reproducible from `scripts/11c_eval_finbert_zero_shot.py`.
* **The pipeline is reproducible and auditable.** Every Phase 4/5 summary
  carries `git_commit_sha`, `dataset_name`, `benchmark_mode`, and a leakage
  count column.

---

## 11. What CARA-FinSent Cannot Claim Yet

* **Multi-seed mean ± std numbers** — only single-seed values exist on CPU.
* **Calibration / abstention improvement vs FinBERT vanilla** — requires
  multi-seed comparison.
* **Agreement-aware training improvement** — script is fixed
  ([scripts/11f_train_finbert_agreement_weighted.py](../scripts/11f_train_finbert_agreement_weighted.py))
  but no full training run has been performed in this environment.
* **Retrieval contribution** — corpus is empty.
* **FiQA in-domain claims** — pending semantics audit.
* **Cross-domain robustness as a positive result** — only as a "honest stress
  test" framing.

---

## 12. Next Step Toward Paper Writing

Run, on a GPU box with internet, the following single command:

```bash
python scripts/29_run_final_research_package.py --mode full \
    --seeds 13 21 42 87 101 --datasets phrasebank --min_seeds 5
```

That will produce:

* `seed_sweep_summary_*.csv` covering classical, FinBERT zero-shot, FinBERT
  fine-tuned, and agreement-weighted FinBERT on PhraseBank.
* `final_leaderboard_mean_std_*.csv` with `research_grade=True` rows.
* `calibration_summary_*.csv`, `abstention_curve_*.csv`,
  `error_analysis_summary_*.csv` for the chosen best predictions file.
* `research_gate_report_*.csv` ideally printing `research_gate=PASS`.

Once the gate flips to `PASS`, this report should be updated with the
multi-seed numbers and the manuscript can be drafted from
`final_leaderboard_mean_std_*.csv`, `calibration_summary_*.csv`, and
`error_analysis_summary_*.csv`.

---

## 13. Acceptance Criteria Status

| # | Criterion | Status |
| --- | --- | --- |
| 1 | `python -m compileall -q src scripts` passes | PASS |
| 2 | `01_audit_datasets.py` returns `audit_gate=PASS` | PASS |
| 3 | `11d_check_finbert_label_mapping.py` passes | PASS |
| 4 | `seed_sweep_summary_*.csv` exists with ≥5 seeds | PENDING (GPU) |
| 5 | `final_leaderboard_mean_std_*.csv` exists | PASS (currently 2-seed) |
| 6 | `calibration_summary_*.csv` exists | PASS |
| 7 | `abstention_curve_*.csv` exists | PASS |
| 8 | `error_analysis_summary_*.csv` exists | PASS |
| 9 | `docs/FIQA_BENCHMARK_DECISION.md` exists | PASS |
| 10 | `docs/PHASE5_FINAL_RESEARCH_EVIDENCE_REPORT.md` exists | PASS (this file) |
| 11 | `25_research_gate.py --min_seeds 5` returns PASS | FAIL — blockers documented (`min_seed_runs`, `not_smoke_tests`) |
