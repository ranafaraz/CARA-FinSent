# Phase 8 GPU Evidence Summary

Date: 2026-05-02
Repo: CARA-FinSent (`main`)
Pod: `zvqce99epg537l` (RTX 3090 24GB)

## Cost Control

- RunPod lifecycle control is automated in `scripts/runpod_pod.py`.
- RunPod API migrated to REST (`https://rest.runpod.io/v1`).
- API env var switched to `POD_API_KEY` (fallback support: `API_KEY`, `RUNPOD_API_KEY`).
- Pod was verified `EXITED` after stop command.

## Core Results (PhraseBank)

Source: `results/2026-05-02/final_leaderboard_mean_std_20260502_081311.csv`

- Best mean accuracy (5 seeds): `finbert_finetuned` = 0.8859
- Best mean macro-F1 (5 seeds): `finbert_zero_shot` = 0.8836
- Best classical macro-F1: `tfidf_linear_svm` = 0.6941

## Research Gate Snapshot

Source: `results/2026-05-02/research_gate_report_20260502_081317.csv`

- Overall status: PASS
- Caveat: this gate pass did not enforce 5 seeds for `agreement_weighted`.

## Agreement-Weighted Status At Phase 8 Close

Sources:
- `results/2026-05-02/finbert_agreement_weighted_summary_20260502_082021.csv`
- `results/2026-05-02/finbert_agreement_weighted_summary_20260502_082228.csv`

Status:
- Only seed `13` completed (duplicate rerun observed).
- Missing seeds at handoff: `21`, `42`, `87`, `101`.

## Calibration / Error / Abstention Artifacts

- `results/2026-05-02/calibration_summary_20260502_081312.csv`
- `results/2026-05-02/error_analysis_summary_20260502_081316.csv`
- `results/2026-05-02/error_examples_20260502_081316.csv`
- `results/2026-05-02/abstention_curve_20260502_081312.csv`

## Artifact Hygiene

- Archive backup exists locally as `cara_phase8_results.tgz`.
- Extracted staging (`_phase8_extracted/`) is excluded from source control and removed from the repo tree.
- Lightweight evidence was preserved under `artifacts/phase8_final_evidence/`.
