# Phase 15 — Agent Completion Report

**Generated (UTC):** 2026-05-02
**Repo head:** `2212d22` on `main` (in sync with `origin/main`, working tree clean)
**Phase 15 commit:** pending — all artefacts written; awaiting Rana's commit decision.

## Q1. Is the project on track for paper writing?

**Yes.** The Phase 15 paper-ready evidence package is complete at
[artifacts/phase15_paper_ready_evidence/](../artifacts/phase15_paper_ready_evidence/), pinned by
[artifacts/phase15_paper_ready_evidence_manifest.json](../artifacts/phase15_paper_ready_evidence_manifest.json) (git SHA, SHA-256 hashes, timestamp).
The reliability-first framing the spec requires is fully supported by the Phase 14
val-fit / test-eval pipeline, which was re-verified in Phase 15 Task 3 with bit-for-bit reproduction.

## Q2. Was the AW degradation issue fixed or just diagnosed?

**Diagnosed, not fixed.** See [PHASE15_AW_REPRODUCIBILITY_DIAGNOSIS.md](PHASE15_AW_REPRODUCIBILITY_DIAGNOSIS.md).

- Root cause space narrowed to: (1) CPU vs CUDA kernel non-determinism (most likely), (2) single-seed lower-tail draw, (3) AMP fp16 vs CPU fp32 logit precision, (4) library-version drift since Phase 11 RunPod image.
- All input-side / code-side / label-side / checkpoint-side variables are byte-identical between Phase 11 and Phase 14 (table in §2 of the diagnosis doc).
- A single Kaggle GPU rerun is staged in [artifacts/phase15_aw_kaggle/](../artifacts/phase15_aw_kaggle/) (`prepare_only` per Rana's instruction). PASS criterion: test macro-F1 ≥ 0.8817.
- Until the Kaggle rerun lands, the Phase 11 5-seed envelope `0.8863 ± 0.0023` remains the paper-headline AW number, and all Phase 14 AW-conditional results are reported as **lower bounds** with explicit caveats.

## Q3. What is the best-performing clean model?

Two answers, both relevant for the paper:

- **Best single model on test macro-F1:** `finbert_zero_shot_isotonic` at **macro-F1 = 0.9038, accuracy = 0.9030**. Single ProsusAI/finbert checkpoint with isotonic calibration fitted on the val split.
- **Best system on test macro-F1:** `ensemble_grid_val_AW0.40_ZS0.60_FT0.00` at **macro-F1 = 0.8953**, weights tuned only on the val split. The ensemble does **not** beat ZS-isotonic on macro-F1; its advantage is reliability geometry (Brier 0.1595 vs 0.1518) plus interpretability of the AW signal.

See [PHASE15_FINAL_CLEAN_LEADERBOARD.md](PHASE15_FINAL_CLEAN_LEADERBOARD.md) row table.

## Q4. What is the best-performing calibrated model?

**`finbert_fine_tuned_isotonic`** has the lowest test ECE at **ECE₁₀ = 0.0138**
(macro-F1 = 0.8958, accuracy = 0.8936). It is the most decision-grade model in
the leaderboard. `finbert_zero_shot_isotonic` is a close second (ECE₁₀ = 0.0165)
with higher macro-F1 — pick depending on whether the paper section is about
absolute accuracy or about calibration tightness.

## Q5. What claims are now safe to write in the paper?

✅ **Safe (in-domain, reliability-first):**
1. PhraseBank macro-F1 in the 0.86–0.91 band across {ZS, FT, AW, ensemble}; differences are not statistically significant within the Phase 11 paired-bootstrap CIs.
2. Post-hoc calibration (isotonic on val) reduces ECE₁₀ from ~0.034 to ~0.014 for FT and from ~0.024 to ~0.017 for ZS, **without degrading** macro-F1.
3. Platt scaling is the val-best calibrator for the AW variant on PhraseBank (ECE₁₀ 0.0418 → 0.0385; macro-F1 0.8648 → 0.8813).
4. Val-tuned ensemble of {AW, ZS, FT} reaches macro-F1 = 0.8953 with weights `(0.4, 0.6, 0.0)`; the ensemble winner is val-selected, not test-selected.
5. Cross-domain transfer to FiQA (polarity-corrected, n = 223) drops macro-F1 to 0.34–0.46 across all three FinBERT variants; PhraseBank fine-tuning, including agreement-weighted fine-tuning, **does not** improve out-of-domain accuracy or calibration.
6. CARA-FinSent is an **in-domain reliability framework**, not a cross-domain SOTA system.

## Q6. What claims are NOT safe and should be removed?

❌ **Unsafe (would not survive review):**
1. Any "universally outperforms FinBERT" claim. None of the in-domain comparisons produce a statistically significant winner.
2. Any "robust to domain shift" claim. The 0.45-macro-F1 PhraseBank → FiQA drop is the opposite of robustness.
3. Any "agreement-weighted training generalises out-of-domain" claim. AW is the *worst* model on FiQA.
4. Any "calibration repairs out-of-domain accuracy" claim — calibration was fitted on the PhraseBank val split.
5. Any cross-paper FiQA comparison. The polarity-corrected n=223 slice is not the same as published FiQA leaderboards.
6. Any AW point-estimate claim that omits the multi-seed envelope. The 0.8860 / 0.8648 numbers must always be presented inside the `[0.8817, 0.8909]` envelope context.

See [PHASE15_EXTERNAL_VALIDATION_CLAIM_BOUNDARIES.md](PHASE15_EXTERNAL_VALIDATION_CLAIM_BOUNDARIES.md) for the verbatim cross-domain paragraph.

## Q7. Was Kaggle used? Was RunPod used?

- **Kaggle:** *prepared, not submitted*. Kernel manifest, README, and notebook live at [artifacts/phase15_aw_kaggle/](../artifacts/phase15_aw_kaggle/). Submission deferred to Rana per the `prepare_only` instruction. A status stub is at [results/2026-05-02/phase15_aw_gpu_repro/STATUS.yml](../results/2026-05-02/phase15_aw_gpu_repro/STATUS.yml).
- **RunPod:** *not used.* Phase 15 hard rule (no auto-RunPod) was respected.

## Q8. What files should I send to ChatGPT for the paper?

Send the entire compact package as one tarball:

```
artifacts/phase15_paper_ready_evidence/
artifacts/phase15_paper_ready_evidence_manifest.json
```

It contains exactly 11 files (~tens of KB total):

- `phase15_final_clean_leaderboard.csv` — the headline 10-row leaderboard.
- `calibration_summary_{zs,ft,aw}.csv` — per-model val-fit/test-eval calibration table (uncalibrated / temperature / Platt / isotonic).
- `ensemble_summary.csv`, `ensemble_paired_comparison.csv` — the four ensemble strategies + paired bootstrap / McNemar.
- `external_fiqa_summary.csv` — polarity-corrected FiQA macro-F1 / ECE / Brier per family.
- `PHASE15_FINAL_CLEAN_LEADERBOARD.md` — narrative for the leaderboard.
- `PHASE15_AW_REPRODUCIBILITY_DIAGNOSIS.md` — full AW gap discussion.
- `PHASE15_EXTERNAL_VALIDATION_CLAIM_BOUNDARIES.md` — exactly which claims are safe.
- `PHASE15_REPO_HEAD_AUDIT.md` — provenance / phase-number confirmation.

Plus the manifest JSON, which includes git SHA + SHA-256 hashes so ChatGPT (or any reviewer) can verify the package is the canonical one.

## Q9. What is the headline framing for the paper?

> **CARA-FinSent is a reliability-first financial sentiment analysis framework that combines agreement-aware training, clean post-hoc calibration, abstention/error analysis, and cross-domain validation to expose when sentiment predictions are decision-grade and when they are unsafe.**

This is consistent with all evidence in the paper-ready package. Nothing in Phase 14 or Phase 15 supports a stronger claim.

---

## Summary table of Phase 15 deliverables

| Task | Output | Status |
|---|---|---|
| 1 | [docs/PHASE15_REPO_HEAD_AUDIT.md](PHASE15_REPO_HEAD_AUDIT.md) | ✅ |
| 2 | [results/2026-05-02/phase15_artifact_audit_20260502_185837.csv](../results/2026-05-02/phase15_artifact_audit_20260502_185837.csv) (15/15 present) | ✅ |
| 3 | [results/2026-05-02/phase15_calibration_check/](../results/2026-05-02/phase15_calibration_check/) (9 files; ZS/FT/AW reproduced bit-for-bit) | ✅ |
| 4 | [docs/PHASE15_AW_REPRODUCIBILITY_DIAGNOSIS.md](PHASE15_AW_REPRODUCIBILITY_DIAGNOSIS.md) | ✅ |
| 5 | [artifacts/phase15_aw_kaggle/](../artifacts/phase15_aw_kaggle/) (prepare-only per Rana) | ⚠ pending submission |
| 6 | [results/2026-05-02/phase15_final_clean_leaderboard_20260502_190410.csv](../results/2026-05-02/phase15_final_clean_leaderboard_20260502_190410.csv) + [docs/PHASE15_FINAL_CLEAN_LEADERBOARD.md](PHASE15_FINAL_CLEAN_LEADERBOARD.md) | ✅ |
| 7 | [docs/PHASE15_EXTERNAL_VALIDATION_CLAIM_BOUNDARIES.md](PHASE15_EXTERNAL_VALIDATION_CLAIM_BOUNDARIES.md) | ✅ |
| 8 | [results/2026-05-02/phase15_cleanup_manifest_20260502_190603.csv](../results/2026-05-02/phase15_cleanup_manifest_20260502_190603.csv) (363 keep-main / 297 keep-appendix / 685 review / 0 cache-or-temp) | ✅ |
| 9 | [artifacts/phase15_paper_ready_evidence/](../artifacts/phase15_paper_ready_evidence/) + [artifacts/phase15_paper_ready_evidence_manifest.json](../artifacts/phase15_paper_ready_evidence_manifest.json) | ✅ |
| 10 | this document | ✅ |

## Next actions for Rana

1. (Optional) Submit the Kaggle GPU AW seed=13 rerun to confirm the Phase 11 envelope.
2. Commit the Phase 15 deliverables (allow-list: `docs/PHASE15_*.md`, `scripts/5*_phase15_*.py`, `results/2026-05-02/phase15_*`, `artifacts/phase15_*`).
3. Hand the `artifacts/phase15_paper_ready_evidence/` package to ChatGPT for the paper draft.
