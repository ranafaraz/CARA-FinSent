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

## Step 9 deliverable — full required analysis

The Phase 10 plan requires the report to address 12 items. Each is answered below using
on-disk evidence (`results/2026-05-02/*`).

### 1. Kaggle authentication status
PASS. `kaggle kernels list --user ranafarazahmed` returned a valid kernel listing using
credentials from `~/.kaggle/kaggle.json`. CLI version 2.1.0 (in `.venv`).

### 2. Kaggle kernel slug (for Phase 10)
**No new Phase 10 kernel was created.** The pre-existing Phase 9 GPU kernel
`ranafarazahmed/cara-finsent-phase-9-agreement-weighted-v2` (last run
2026-05-02 10:03 UTC) is retained as the GPU entry point. New kernel push was not
required because the AW 5-seed evidence is already complete and gate-passing
(`docs/PHASE10_EVIDENCE_INVENTORY.md`).

### 3. Whether GPU was detected
N/A for this Phase 10 pass (no new GPU run executed). The Phase 8/9 GPU runs that
produced the on-disk evidence were executed on RunPod GPU and the existing Kaggle kernel
ran with `enable_gpu: true`.

### 4. Whether agreement-weighted ran for all 5 seeds
YES. `seed_sweep_summary_20260502_105326.csv` contains 5 unique seeds for the
`finbert_agreement_weighted` model on `phrasebank` (in-domain), namely
`[13, 21, 42, 87, 101]`. Gate check `agreement_weighted_seed_runs` is PASS.

### 5. Final leaderboard (in-domain PhraseBank)

Sorted by mean macro-F1 desc:

| rank | model | n_seeds | mean macro-F1 | std | mean ECE@10 | std ECE | mean Brier |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `finbert_agreement_weighted` | 5 | **0.8863** | 0.00227 | 0.0495 | 0.0221 | 0.1713 |
| 2 | `finbert_base_zero_shot_ProsusAI/finbert` | 5 | 0.8836 | 0.00000 | **0.0236** | 0.0000 | 0.1762 |
| 3 | `finbert_finetuned_ProsusAI/finbert` | 5 | 0.8821 | 0.00923 | 0.0608 | 0.0240 | 0.1810 |
| 4 | `tfidf_linear_svm` | 6 | 0.6941 | 0.00000 | n/a | n/a | n/a |
| 5 | `tfidf_logistic_regression` | 6 | 0.6838 | 0.00000 | 0.1732 | 0.0000 | 0.4250 |
| 6 | `tfidf_sgd_log_loss` | 6 | 0.6773 | 0.00238 | 0.0988 | 0.00164 | 0.3777 |
| 7 | `tfidf_xgboost` | 6 | 0.6216 | 0.00000 | 0.0321 | 0.0000 | 0.3931 |
| 8 | `tfidf_random_forest` | 6 | 0.5902 | 0.00666 | 0.0416 | 0.00444 | 0.3954 |
| 9 | `tfidf_multinomial_nb` | 6 | 0.3938 | 0.00000 | 0.1056 | 0.0000 | 0.4691 |
| 10 | `majority_baseline` | 6 | 0.2496 | ~0 | 0.4015 | 0.0000 | 0.8029 |

Source: `results/2026-05-02/final_leaderboard_mean_std_20260502_110922.csv`.

### 6. Whether `agreement_weighted` appears in the leaderboard
YES — and it ranks **#1 by mean macro-F1** across all 10 models.

### 7. Research gate status
PASS overall. All 11 checks PASS in
`results/2026-05-02/research_gate_report_20260502_110923.csv` (compileall,
controlled_gold_splits, audit_gate, finbert_label_sanity, min_seed_runs,
agreement_weighted_seed_runs, no_text_hash_leakage, result_metadata,
calibration_report, error_analysis_report, not_smoke_tests).

### 8. Best model by macro-F1
**`finbert_agreement_weighted`** with mean macro-F1 = 0.8863 ± 0.00227 (n=5).

The 95% confidence margin from a single training-data variance is small, but the gap to
zero-shot FinBERT (Δ = +0.0027 macro-F1) is **within** one standard deviation of the AW
runs (0.00227). Treat the macro-F1 advantage as a small, but consistent lift, not a
decisive win.

### 9. Best model by calibration / ECE
**`finbert_base_zero_shot_ProsusAI/finbert`** with mean ECE@10 = **0.0236** (effectively
deterministic, std ≈ 7e-9 because the zero-shot scoring is independent of the training
seed). This is more than 2× better calibrated than agreement-weighted FinBERT (0.0495).

Among trained classifiers (excluding zero-shot), the best ECE belongs to
`tfidf_xgboost` (0.0321), but its macro-F1 (0.6216) is far below FinBERT-class models.

### 10. Does agreement weighting improve macro-F1, calibration, or robustness?

| dimension | AW vs zero-shot FinBERT | AW vs fine-tuned FinBERT |
|---|---|---|
| macro-F1 | +0.0027 (mild lift, within 1σ) | +0.0042 (mild lift) |
| ECE@10 | **−0.0259 worse** (AW is ~2× less calibrated) | +0.0113 better |
| std macro-F1 across seeds | AW 0.00227 vs ZS 0.0 (ZS deterministic) | AW 0.00227 vs FT 0.00923 → AW is **~4× more stable** than vanilla FT |
| Brier score | +0.0049 better | +0.0098 better |

Verdict:
- AW gives a small macro-F1 lift over both FinBERT baselines.
- AW is **substantially more seed-robust** than vanilla fine-tuning (std 0.00227 vs
  0.00923).
- AW does **NOT** improve calibration over zero-shot FinBERT — zero-shot is the
  best-calibrated model in the leaderboard.
- AW improves Brier score relative to both FinBERT baselines.

### 11. What can be claimed in the paper
- AW FinBERT achieves the **highest mean macro-F1** in the in-domain PhraseBank
  benchmark across 5 unique seeds.
- AW FinBERT is **markedly more seed-stable** than vanilla FinBERT fine-tuning
  (std 0.00227 vs 0.00923 macro-F1, ~4× lower).
- AW FinBERT improves Brier score relative to both zero-shot and fine-tuned FinBERT.
- The full leaderboard (10 models, classical + FinBERT family) is research-grade
  (all gate checks PASS, no text-hash leakage, controlled gold splits, min seeds satisfied).

### 12. What cannot be claimed
- AW does **not** beat zero-shot FinBERT on calibration (ECE@10 0.0495 vs 0.0236).
- The macro-F1 advantage of AW over zero-shot FinBERT is small (Δ ≈ 0.003) and
  comparable to one standard deviation across AW seeds; do **not** claim a definitive
  accuracy/F1 win without an explicit paired statistical test.
- These results are in-domain (PhraseBank). Out-of-domain robustness, FiQA-headline
  generalization, or live-trading utility are NOT claims supported by this evidence.
- No latency claim is made (mean_latency_or_seconds is empty in the leaderboard).

## Decision and follow-ups

- **No Kaggle GPU run executed**: agreement-weighted 5-seed evidence is already complete.
- **No RunPod usage**: Phase 10 Strict Cost Rule honored. RunPod was not started.
- **Future re-runs**: use the existing Kaggle kernel
  `ranafarazahmed/cara-finsent-phase-9-agreement-weighted-v2` as the GPU entry point.
- **If evidence is ever invalidated** (e.g., dataset rebuild or label-mapping fix), re-run
  the kernel and re-execute `scripts/27_summarize_seed_sweep.py --min_seeds 5` followed by
  `scripts/25_research_gate.py --min_seeds 5` to regenerate gate evidence.
- **Paper framing**: per the plan's "Final Scientific Direction", position the paper as a
  reliability-first decision-grade evaluation (calibration + abstention + agreement
  weighting + cost), not as a single accuracy/F1 SOTA claim.
