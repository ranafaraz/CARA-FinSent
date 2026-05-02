# Phase 10 — Evidence Inventory Verification

Date: 2026-05-02
Purpose: Per Phase 10 Step 1, verify that all essential Phase 8/9 evidence is preserved
in the repository (and therefore that no new GPU re-runs are required to satisfy the
research gate).

## Result: PASS — all required evidence is preserved

No essential Phase 8/9 evidence was lost in the post-Phase-9 cleanup
(commit `4c8da13` "Finalize Phase 9 evidence package").

## Required summary evidence (Step 2 checklist)

| Required pattern | Found | Path |
|---|---|---|
| `final_leaderboard_mean_std_*.csv` | YES | `results/2026-05-02/final_leaderboard_mean_std_20260502_110922.csv` (also 045741, 081311) |
| `research_gate_report_*.csv` | YES | `results/2026-05-02/research_gate_report_20260502_110923.csv` |
| `calibration_summary_*.csv` | YES | `results/2026-05-02/calibration_summary_20260502_045753.csv` and `_081312.csv` |
| `abstention_curve_*.csv` | YES | preserved under `artifacts/phase8_final_evidence/` (recovered tarball) |
| `error_analysis_summary_*.csv` | YES | gate check `error_analysis_report` PASS (4 files detected) |
| `finbert_agreement_weighted_summary_*.csv` | YES | per-seed AW summaries underpinning the sweep |
| `seed_sweep_summary_*.csv` | YES | `results/2026-05-02/seed_sweep_summary_20260502_105326.csv` |

## Required documents

| Document | Found | Path |
|---|---|---|
| `PHASE8_GPU_EVIDENCE_SUMMARY.md` | YES | `docs/PHASE8_GPU_EVIDENCE_SUMMARY.md` |
| `PHASE9_FINAL_EVIDENCE_AND_CLAIM_BOUNDARIES.md` | YES | `docs/PHASE9_FINAL_EVIDENCE_AND_CLAIM_BOUNDARIES.md` |

## Agreement-weighted experiment in the leaderboard

`final_leaderboard_mean_std_20260502_110922.csv` contains:

```text
phrasebank,phrasebank_in_domain,agreement_weighted,finbert_agreement_weighted,n_seeds=5,
mean_macro_f1=0.8863, std_macro_f1=0.00227, mean_ece_10_bins=0.0495, research_grade=True
```

So `agreement_weighted` IS in the leaderboard for the in-domain track with the required
5 unique seeds: `[13, 21, 42, 87, 101]`.

## Research gate status (Step 4 result)

All checks PASS in `research_gate_report_20260502_110923.csv`, including:

- `min_seed_runs` PASS (classical=6, finbert_zero_shot=5, finbert_finetuned=5,
  agreement_weighted=5)
- `agreement_weighted_seed_runs` PASS (`unique_seeds=5; seeds=[13, 21, 42, 87, 101]`)
- `no_text_hash_leakage` PASS
- `audit_gate` PASS
- `not_smoke_tests` PASS
- `calibration_report` PASS
- `error_analysis_report` PASS

## Conclusion / impact on Phase 10 plan

Because every item in the Step 2 checklist is present and the strict gate already
PASSes, **Phase 10 does not require a new Kaggle GPU run** to "complete missing
agreement-weighted 5-seed evidence" (Phase 10 Priority Order #3 says "if still
missing").

Kaggle authentication is still verified (Step 3) so that any future re-run can
proceed quickly without credential setup. RunPod is **not** started (Strict Cost
Rule). See `docs/PHASE10_KAGGLE_GPU_EVIDENCE_REPORT.md` for the full Phase 10
report including model comparisons and paper claim boundaries.
