# Phase 10 — Kaggle-First GPU Evidence Report

Date: 2026-05-02
Branch: `main`
Scope: Kaggle-first GPU evidence completion for the CARA-FinSent agreement-weighted FinBERT track.

## TL;DR

The agreement-weighted FinBERT 5-seed evidence required by the research gate is **already
complete and preserved on disk** from the Phase 9 run. **No new Kaggle GPU run was required.**
Kaggle CLI authentication was verified end-to-end and is ready for any future GPU re-run.
The pre-existing Phase 9 Kaggle kernel (`ranafarazahmed/cara-finsent-phase-9-agreement-weighted-v2`)
is retained as a redundant GPU evidence path.

## Inventory verification (Phase 8/9 evidence preserved)

Verified files (committed and present locally under `results/2026-05-02/`):

- `seed_sweep_summary_20260502_105326.csv` — 5 unique seeds: `[13, 21, 42, 87, 101]`
- `final_leaderboard_mean_std_20260502_110922.csv` — agreement_weighted ranked #1
- `research_gate_report_20260502_110923.csv` — overall PASS
- `research_gate_manifest_20260502_110923.json`
- `final_leaderboard_manifest_20260502_110922.json`

Key research-gate checks (all PASS):

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

## Agreement-weighted FinBERT — 5-seed summary (PhraseBank in-domain)

| seed | accuracy | macro_f1 | weighted_f1 | mcc | ECE@10 | brier |
|---|---|---|---|---|---|---|
| 13  | 0.8895 | 0.8860 | 0.8891 | 0.7985 | 0.0715 | 0.1799 |
| 21  | 0.8895 | 0.8901 | 0.8900 | 0.8019 | 0.0754 | 0.1852 |
| 42  | 0.8884 | 0.8859 | 0.8887 | 0.7960 | 0.0377 | 0.1640 |
| 87  | 0.8905 | 0.8853 | 0.8899 | 0.7986 | 0.0345 | 0.1645 |
| 101 | 0.8853 | 0.8841 | 0.8852 | 0.7926 | 0.0286 | 0.1628 |

Aggregate (from `final_leaderboard_mean_std_20260502_110922.csv`):
mean macro-F1 = **0.8863 ± 0.0023**, mean accuracy = **0.8886 ± 0.0020** (n_seeds=5).

## Kaggle CLI verification

- Kaggle CLI: **2.1.0** (installed in `.venv\Scripts\kaggle.exe`)
- Credentials: `~/.kaggle/kaggle.json` present
- Authentication: confirmed via `kaggle kernels list --user <self>` (succeeded)
- Pre-existing Kaggle kernels relevant to this track:
  - `ranafarazahmed/cara-finsent-phase-9-agreement-weighted-v2` (last run 2026-05-02 10:03 UTC)
  - `ranafarazahmed/cara-finsent-phase-9-agreement-weighted` (last run 2026-05-02 09:59 UTC)

Because the local 5-seed evidence was already complete and gate-passing, no new kernel
was created or pushed in this Phase 10 pass. Kaggle remains the **default first-choice
GPU surface** for any future re-runs (per Phase 10 rule #1). RunPod was **not** started.

## Compact evidence package

Saved under `artifacts/phase10_kaggle_agreement_evidence/`:

- `seed_sweep_summary_20260502_105326.csv`
- `research_gate_report_20260502_110923.csv`
- `research_gate_manifest_20260502_110923.json`
- `final_leaderboard_mean_std_20260502_110922.csv`
- `final_leaderboard_manifest_20260502_110922.json`

## Decision and follow-ups

- **No Kaggle GPU run executed**: agreement-weighted 5-seed evidence is already complete.
- **No RunPod usage**: Phase 10 rule #2 honored.
- **Future re-runs**: use the existing Kaggle kernel
  `ranafarazahmed/cara-finsent-phase-9-agreement-weighted-v2` as the GPU entry point.
- **If evidence is ever invalidated** (e.g., dataset rebuild or label-mapping fix), re-run
  the kernel and re-execute `scripts/27_summarize_seed_sweep.py --min_seeds 5` followed by
  `scripts/25_research_gate.py --min_seeds 5` to regenerate gate evidence.
