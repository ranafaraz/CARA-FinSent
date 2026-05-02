# Phase 8 GPU Final Evidence Report

Date: 2026-05-02  
Repo: CARA-FinSent (branch: main)  
Pod: cara-research-pod (`zvqce99epg537l`, RTX 3090 24GB)

## 1) Cost Control Outcome

- Pod lifecycle automation is implemented in `scripts/runpod_pod.py`.
- RunPod API endpoint updated to REST (`https://rest.runpod.io/v1`) after GraphQL retirement (HTTP 403 / Cloudflare 1010).
- API key variable updated to `POD_API_KEY` (with fallback support for `API_KEY` and `RUNPOD_API_KEY`).
- Pod status verified as `EXITED` after stop command, so GPU billing has been halted.

## 2) Core Experiment Outcomes (PhraseBank, in-domain)

Source: `results/2026-05-02/final_leaderboard_mean_std_20260502_081311.csv`

- Best mean accuracy (5 seeds): `finbert_finetuned` = 0.8859
- Best mean macro-F1 (5 seeds): `finbert_zero_shot` = 0.8836 (tie-level with finetuned uncertainty)
- Classical best (macro-F1): `tfidf_linear_svm` = 0.6941

Top rows:

1. `finbert_zero_shot`: mean accuracy 0.8832, mean macro-F1 0.8836, mean ECE 0.0236
2. `finbert_finetuned`: mean accuracy 0.8859, mean macro-F1 0.8821, mean ECE 0.0608
3. `tfidf_linear_svm`: mean accuracy 0.7497, mean macro-F1 0.6941

## 3) Research Gate

Source: `results/2026-05-02/research_gate_report_20260502_081317.csv`

- Overall gate: PASS
- Checks passed: compileall, controlled splits, audit gate, leakage check, metadata, calibration and error reports.

Important caveat:
- `min_seed_runs` PASS was satisfied by `classical`, `finbert_finetuned`, and `finbert_zero_shot`.
- Agreement-weighted was not included in this gate requirement and therefore did not block PASS.

## 4) Agreement-Weighted Status (Not Fully Complete)

Sources:
- `results/2026-05-02/finbert_agreement_weighted_summary_20260502_082021.csv`
- `results/2026-05-02/finbert_agreement_weighted_summary_20260502_082228.csv`

Observed state:
- Both summary files are for seed `13` (duplicate rerun), with macro-F1 = 0.8860.
- Unique completed agreement-weighted seeds: 1/5.
- Remaining owed seeds: 21, 42, 87, 101.

## 5) Calibration, Abstention, Error Analysis

Calibration source: `results/2026-05-02/calibration_summary_20260502_081312.csv`
- Model audited: `finbert_zero_shot`
- Accuracy: 0.8874
- Macro-F1: 0.8828
- ECE (10 bins): 0.0768
- Brier score: 0.1859

Error analysis source: `results/2026-05-02/error_analysis_summary_20260502_081316.csv`
- Total rows: 959
- Errors: 108 (11.26%)
- Largest confusion mode: neutral -> positive (50 cases)
- Overconfident wrong predictions: 78

Abstention source: `results/2026-05-02/abstention_curve_20260502_081312.csv`
- At 70% confidence threshold: coverage 96.25%, accuracy on kept 0.9025, macro-F1 on kept 0.9034

## 6) Reproducibility and Artifact Handling

- Backup created from pod and pulled locally: `cara_phase8_results.tgz` (~9.5 MB).
- Extracted staging directory: `_phase8_extracted/`.
- Timestamped outputs preserved under `results/2026-05-02/`.
- No model binaries/checkpoints were added from backup archive.

## 7) Required Next GPU Session (Short Run)

To fully close agreement-weighted evidence:

1. Start pod via `scripts/runpod_pod.py start` then `wait`.
2. Run agreement-weighted only for seeds: 21, 42, 87, 101.
3. Rebuild aggregate leaderboard/calibration/error reports.
4. Pull artifacts and stop pod immediately (`scripts/runpod_pod.py stop`).

Estimated compute is short (single-digit minutes) if run is stable.

## 8) Final Honest Status

- Phase 8 has strong evidence for classical + FinBERT zero-shot + FinBERT finetuned.
- Cost-discipline automation is now in place and pod is currently stopped.
- Agreement-weighted evidence is partial and requires one follow-up GPU session for full 5-seed completeness.
