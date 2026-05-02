# Phase 11 — Evidence Integrity Check

Date: 2026-05-02
Scope: Pre-flight integrity check for the paper-readiness phase. Confirms that the
Phase 8/9/10 evidence used for Phase 11 statistical validation, tables, and figures
is intact, leakage-controlled, and gate-passing.

## Repo state

| field | value |
|---|---|
| latest commit SHA | `d0e119faf59fc735d577b61299e64234d26320ee` |
| branch | `main` |
| working tree | clean (no uncommitted modifications relevant to evidence files) |
| `python -m compileall -q src scripts` | PASS |

## Evidence files used by Phase 11

| role | file | rows |
|---|---|---|
| final leaderboard | `results/2026-05-02/final_leaderboard_mean_std_20260502_110922.csv` | 10 model rows |
| seed sweep | `results/2026-05-02/seed_sweep_summary_20260502_105326.csv` | AW × 5 seeds |
| research gate | `results/2026-05-02/research_gate_report_20260502_110923.csv` | 11 checks |
| calibration summary | `results/2026-05-02/calibration_summary_20260502_081312.csv` | per-model ECE/Brier/coverage |
| abstention curve | `results/2026-05-02/abstention_curve_20260502_081312.csv` | threshold sweep |
| error analysis | `results/2026-05-02/error_analysis_summary_20260502_081316.csv` | per-model error breakdown |
| zero-shot predictions | `results/2026-05-02/finbert_baseline_predictions_20260502_081140.csv` | 959 |
| AW predictions | `results/2026-05-02/finbert_agreement_weighted_predictions_20260502_082228.csv` | 959 |

## Research gate status (all PASS)

| check | status | detail |
|---|---|---|
| compileall | PASS | ok |
| controlled_gold_splits | PASS | all present |
| audit_gate | PASS | overall=PASS, leakage=PASS |
| finbert_label_sanity | PASS | 4 mapping artifacts |
| min_seed_runs | PASS | classical=6; finbert_zero_shot=5; finbert_finetuned=5; agreement_weighted=5 |
| agreement_weighted_seed_runs | PASS | unique_seeds=5; seeds=[13, 21, 42, 87, 101] |
| no_text_hash_leakage | PASS | all zero |
| result_metadata | PASS | all present |
| calibration_report | PASS | 5 calibration_summary files |
| error_analysis_report | PASS | 4 error_analysis_summary files |
| not_smoke_tests | PASS | max num_epochs=3.0 |

## Leaderboard model coverage check

The leaderboard contains the four required experiments per Phase 11 Step 1:

| required experiment | present | model |
|---|---|---|
| `agreement_weighted` | YES | `finbert_agreement_weighted` (n_seeds=5) |
| `finbert_zero_shot` | YES | `finbert_base_zero_shot_ProsusAI/finbert` (n_seeds=5) |
| `finbert_finetuned` | YES | `finbert_finetuned_ProsusAI/finbert` (n_seeds=5) |
| `tfidf_linear_svm` | YES | `tfidf_linear_svm` (n_seeds=6) |

## Missing/at-risk items

None. All required evidence files are present, gate-passing, and free of text-hash
leakage. Phase 11 may proceed without any GPU re-run.
