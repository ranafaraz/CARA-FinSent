# Phase 4 — Research-Grade Experiment Plan

This document describes the research-grade experimental layer added in Phase 4
on top of the corrections from Phases 1 and 3. It covers (a) the fixes applied
to the agreement-aware fine-tuning script, (b) the new orchestration / reporting
scripts, and (c) the deliverables required for paper-quality evidence.

> Phase 4 does **not** introduce new modeling ideas. Its purpose is to make the
> existing pipeline reproducible, statistically defensible, and reliability-aware.

---

## 1. Fixes applied to existing scripts

### 1.1 `scripts/11f_train_finbert_agreement_weighted.py`

Three correctness bugs were addressed:

| Issue | Fix |
| --- | --- |
| `high_only` filter compared decimal agreement scores against `75` and silently dropped every training row. | New `high_agreement_threshold(...)` helper inspects the score range and returns `0.75` (decimal scale) or `75` (percent scale). The filter aborts with a clear error if it would produce 0 rows. |
| Training and evaluation both happened on the test split, so model selection was test-driven. | The script now loads the validation split, evaluates each epoch, uses `metric_for_best_model='macro_f1'`, `load_best_model_at_end=True`, `EarlyStoppingCallback(patience=2)`, and `save_strategy='epoch'`. The test split is used **only** in the final `trainer.predict(test_ds)` call. |
| `WeightedTrainer.compute_loss(...)` rejected the `num_items_in_batch` keyword passed by newer `transformers` releases and re-wrapped the existing weight tensor with `torch.tensor(...)`. | Signature is now `compute_loss(self, model, inputs, return_outputs=False, **kwargs)`. The sample weight is moved with `sample_weight.to(loss.device).float()` and is only wrapped via `torch.as_tensor(...)` when it is not already a tensor. |

---

## 2. New scripts

| Script | Purpose | Key outputs |
| --- | --- | --- |
| `scripts/19_run_seed_sweep.py` | Run any subset of `classical`, `finbert_zero_shot`, `finbert_finetuned`, `agreement_weighted` across multiple seeds via subprocesses. Loads gold split for validation only — never resplits. | `seed_sweep_summary_<ts>.csv` with the unified column set required by Phase 4. |
| `scripts/20_aggregate_research_results.py` | Aggregate every `seed_sweep_summary_*.csv` under `results/` into mean/std and 95 % CI leaderboards. Flags `(model, dataset_name)` rows as `research_grade=True` when `n_seeds >= --min_seeds` (default 5). | `research_leaderboard_mean_std_<ts>.csv`, `research_leaderboard_ci95_<ts>.csv`, `research_best_model_summary_<ts>.csv`, `figures/research_macro_f1_mean_std_<ts>.png`, `figures/research_accuracy_mean_std_<ts>.png`. |
| `scripts/21_error_analysis.py` | Class-wise + neutral-confusion + worst-N error tables and heatmaps from a predictions CSV. Patterns by length bucket, numeric/percentage presence, and (if available) PhraseBank agreement quartile. | `error_analysis_summary_<ts>.csv`, `error_examples_by_class_<ts>.csv`, `neutral_confusion_analysis_<ts>.csv`, `figures/error_confusion_heatmap_<ts>.png`, `figures/neutral_error_breakdown_<ts>.png`. |
| `scripts/22_calibration_abstention_report.py` | Reliability bins, abstention curve over `0.00..0.95` thresholds, and summary including `coverage_at_0_60`/`accuracy_at_0_60`/`macro_f1_at_0_60` plus the same trio at 0.70. | `calibration_reliability_bins_<ts>.csv`, `abstention_curve_<ts>.csv`, `calibration_summary_<ts>.csv`, `figures/reliability_diagram_<ts>.png`, `figures/abstention_curve_<ts>.png`, `figures/confidence_histogram_<ts>.png`. |
| `scripts/23_fiqa_semantics_audit.py` | Loads the upstream FiQA HF dataset(s), reports the raw label/score distribution, the normalized distribution, 50 random samples, unmapped-row count, and a `fiqa_main_benchmark / fiqa_external_stress_test` recommendation. | `data/audit/fiqa_raw_label_distribution_<ts>.csv`, `data/audit/fiqa_label_semantics_sample_<ts>.csv`, `data/audit/fiqa_semantics_audit_manifest_<ts>.json`. |
| `scripts/24_build_external_retrieval_corpus.py` | Aggregates external sources into the required schema (`id, source, text, ticker, company, date, url, doc_type, text_hash`), deduplicates by `text_hash`, and **explicitly excludes any row whose hash matches a controlled gold split**. | `data/retrieval_corpus/retrieval_corpus_<ts>.csv`, `data/retrieval_corpus/latest_retrieval_corpus.csv`, `data/audit/retrieval_corpus_audit_<ts>.csv`. |
| `scripts/25_research_gate.py` | Single-command research-readiness gate. Runs the 10 checks listed in the brief and prints `research_gate=PASS|FAIL`. | `results/<date>/research_gate_report_<ts>.csv`, `results/<date>/research_gate_manifest_<ts>.json`. |

---

## 3. Execution plan (mirrors the brief)

```bash
# 1. Compile + audit
python -m compileall -q src scripts
python scripts/01_audit_datasets.py --inputs \
    data/processed/gold/latest_gold_phrasebank_split.csv \
    data/processed/gold/latest_gold_fiqa_split.csv
python scripts/11d_check_finbert_label_mapping.py \
    --sample_file data/processed/gold/latest_gold_phrasebank_split.csv

# 2. PhraseBank main benchmark — 5 seeds
python scripts/19_run_seed_sweep.py \
    --dataset_name phrasebank \
    --models classical,finbert_zero_shot,finbert_finetuned,agreement_weighted \
    --seeds 42 43 44 45 46

# 3. FiQA stress benchmark — audit first, then classical + zero-shot only
python scripts/23_fiqa_semantics_audit.py
python scripts/19_run_seed_sweep.py \
    --dataset_name fiqa \
    --models classical,finbert_zero_shot \
    --seeds 42 43 44 45 46

# 4. Cross-domain transfer (both directions, 5 seeds each)
foreach ($s in 42,43,44,45,46) {
    python scripts/18_cross_domain_eval.py --train data/processed/gold/latest_gold_phrasebank_split.csv \
        --test data/processed/gold/latest_gold_fiqa_split.csv --seed $s
    python scripts/18_cross_domain_eval.py --train data/processed/gold/latest_gold_fiqa_split.csv \
        --test data/processed/gold/latest_gold_phrasebank_split.csv --seed $s
}

# 5. Calibration / abstention (run on best PhraseBank prediction CSVs)
python scripts/22_calibration_abstention_report.py \
    --predictions <best_predictions.csv> \
    --model_name finbert_zero_shot --dataset_name phrasebank

# 6. Error analysis (per best model)
python scripts/21_error_analysis.py \
    --predictions <best_predictions.csv> \
    --model_name finbert_zero_shot --dataset_name phrasebank

# 7. Aggregate
python scripts/20_aggregate_research_results.py --results_dir results

# 8. Research gate
python scripts/25_research_gate.py
```

---

## 4. Validation performed during this phase

```text
python -m compileall -q src scripts                              -> PASS
python scripts/22_calibration_abstention_report.py ... predictions=...    -> ECE=0.0344 Brier=0.1577 mean_conf=0.8862 (FinBERT zero-shot, PhraseBank)
python scripts/21_error_analysis.py ... predictions=...                   -> error_analysis_summary written, heatmap rendered
python scripts/24_build_external_retrieval_corpus.py                      -> 0-row corpus (no external inputs supplied yet) + audit row written
python scripts/19_run_seed_sweep.py --dataset_name phrasebank --models classical --seeds 42 43
                                                                          -> 14 rows aggregated across 7 classical models x 2 seeds
python scripts/20_aggregate_research_results.py --min_seeds 2             -> mean/std + CI95 + best leaderboards written
python scripts/25_research_gate.py --min_seeds 5                          -> research_gate=FAIL (expected — full FinBERT 5-seed sweep deferred)
```

The research gate produced 8/10 PASS and 2/10 FAIL:

- `min_seed_runs=FAIL` because only `classical=2` is present (5+ seeds for all four experiments require GPU time).
- `not_smoke_tests=FAIL` because no `finbert_finetuned` / `agreement_weighted` rows are in the sweep yet.

These are the explicit acceptance gates that the brief asks the agent to surface; they will flip to PASS once the full 5-seed sweep is run on a GPU host.

---

## 5. Deferred to GPU/long-runtime execution

The current environment is CPU-only. The following actions are still required
to flip the research gate to PASS but cannot be completed in this phase without
a GPU node:

1. `python scripts/19_run_seed_sweep.py --dataset_name phrasebank --models finbert_finetuned,agreement_weighted --seeds 42 43 44 45 46`
2. `python scripts/19_run_seed_sweep.py --dataset_name fiqa --models classical,finbert_zero_shot --seeds 42 43 44 45 46` (cheap; can also be done on CPU)
3. `python scripts/20_aggregate_research_results.py --results_dir results`
4. `python scripts/25_research_gate.py`

After step 4 returns `research_gate=PASS`, regenerate the calibration and error
analysis reports against the best PhraseBank prediction CSVs (`finbert_finetuned`
and `agreement_weighted`).

---

## 6. Scientific interpretation rules (recap)

The agent and downstream paper authors must continue to follow the rules from
the Phase 4 brief: never claim SOTA from one seed or a smoke test; never mix
PhraseBank and FiQA in the main benchmark; never claim retrieval benefit until
`data/retrieval_corpus/latest_retrieval_corpus.csv` is non-empty and external;
never report accuracy alone — always macro-F1, weighted-F1, MCC, ECE, Brier,
mean confidence, and abstention; always include dataset name, split source,
seed, test rows, and git commit SHA in result summaries.
