# Phase 4 Research-Grade Experiment Report

This report tracks the deliverables and gate status for the Phase 4 research
hardening pass. It is updated as runs complete.

## Acceptance criteria status

| Criterion | Status | Notes |
| --- | --- | --- |
| `compileall` PASS | PASS | `python -m compileall -q src scripts` returns clean. |
| `audit_gate` PASS | PASS | Verified via `scripts/01_audit_datasets.py` on both gold splits. |
| FinBERT label sanity PASS | PASS | `data/audit/finbert_label_mapping_*.json` artifacts present. |
| PhraseBank 5-seed benchmark complete | DEFERRED | CPU-only environment; needs GPU run of `scripts/19_run_seed_sweep.py --dataset_name phrasebank --seeds 42 43 44 45 46`. |
| FinBERT full fine-tuning complete on PhraseBank | DEFERRED | Same reason as above. |
| Agreement-aware fine-tuning complete on PhraseBank | DEFERRED (script fixed) | Bugs (Issues 2, 3, 4) corrected. Awaiting full GPU sweep. |
| Calibration report generated | PASS | Smoke run on FinBERT zero-shot PhraseBank predictions: ECE=0.0344, Brier=0.1577, mean_conf=0.8862. |
| Abstention curve generated | PASS | Same calibration run produced `abstention_curve_*.csv` and `figures/abstention_curve_*.png`. |
| Error analysis generated | PASS | `error_analysis_summary_*.csv` and `figures/error_confusion_heatmap_*.png` produced for the same predictions. |
| FiQA semantic audit generated | PENDING | `scripts/23_fiqa_semantics_audit.py` ready; awaiting next online run that can reach Hugging Face. |
| Cross-domain tests generated | PARTIAL | Phase 3 produced 1-seed cross-domain runs; needs to be re-run for seeds 42–46. |
| `research_gate=PASS` | FAIL (expected) | 8/10 checks pass; `min_seed_runs` and `not_smoke_tests` blocked by deferred GPU work. |

## Smoke validation log (CPU)

```
python -m compileall -q src scripts                                              -> PASS
python scripts/22_calibration_abstention_report.py --predictions results/2026-05-01/finbert_baseline_predictions_20260501_193637.csv \
    --model_name finbert_zero_shot --dataset_name phrasebank                      -> ECE=0.0344, Brier=0.1577
python scripts/21_error_analysis.py --predictions <same predictions>              -> error_analysis_summary written
python scripts/24_build_external_retrieval_corpus.py                              -> 0 rows (no external inputs supplied yet)
python scripts/19_run_seed_sweep.py --dataset_name phrasebank --models classical --seeds 42 43
                                                                                  -> 14 rows, no failures
python scripts/20_aggregate_research_results.py --min_seeds 2                     -> mean/std + CI95 + best leaderboards written
python scripts/25_research_gate.py --min_seeds 5                                  -> research_gate=FAIL (expected)
```

## Outstanding actions (GPU required)

1. `python scripts/19_run_seed_sweep.py --dataset_name phrasebank --models classical,finbert_zero_shot,finbert_finetuned,agreement_weighted --seeds 42 43 44 45 46`
2. `python scripts/19_run_seed_sweep.py --dataset_name fiqa --models classical,finbert_zero_shot --seeds 42 43 44 45 46`
3. `python scripts/23_fiqa_semantics_audit.py`
4. Cross-domain transfer for seeds 42–46 in both directions via `scripts/18_cross_domain_eval.py`.
5. Re-run calibration + error analysis on the best PhraseBank `finbert_finetuned` and `agreement_weighted` prediction CSVs.
6. `python scripts/20_aggregate_research_results.py --results_dir results`
7. `python scripts/25_research_gate.py` — target output `research_gate=PASS`.

## Notes for paper authors

- Treat FiQA results as an external stress test until `scripts/23_fiqa_semantics_audit.py` confirms otherwise.
- Do not claim retrieval benefit until `data/retrieval_corpus/latest_retrieval_corpus.csv` is non-empty after running `scripts/24_build_external_retrieval_corpus.py` with real external sources.
- All summary CSVs from Phase 4 carry `dataset_name`, `benchmark_mode`, `text_hash_leakage_count`, and `git_commit_sha`. The aggregator only flags `(model, dataset)` rows as `research_grade=True` once at least 5 seeds are present.
