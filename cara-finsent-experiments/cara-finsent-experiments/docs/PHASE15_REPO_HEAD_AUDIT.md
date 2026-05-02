# Phase 15 — Repository Head Audit

**Audit timestamp (UTC):** 2026-05-02
**Phase:** 15 (NOT Phase 12 — Phase 12 is historical and superseded)

## 1. Latest commit

| Field | Value |
|---|---|
| SHA | `2212d22` |
| Branch | `main` |
| Tracks | `origin/main` (in sync) |
| Working tree | clean |
| Message | `phase14: correct validation leakage and finalize paper-safe reliability results` |

## 2. Recent history (top 5)

| SHA | Message |
|---|---|
| `2212d22` | phase14: correct validation leakage and finalize paper-safe reliability results |
| `36dae7f` | add data audit files for May 2, 2026 (label distribution, duplicate report, split integrity, text length) |
| `a48bd34` | phase13: add external validation and reliability improvement experiments |
| `082c69d` | phase11: add statistical validation and paper-readiness assets |
| `d0e119f` | phase10: add evidence inventory and complete Step 9 analysis |

## 3. Phase 14 artefacts present

Confirmed on disk (see `results/2026-05-02/phase15_artifact_audit_*.csv` for the full machine-readable manifest):

- `docs/PHASE14_PHASE13_AUDIT.md`
- `docs/PHASE14_AW_CHECKPOINT_RECOVERY_REPORT.md`
- `docs/PHASE14_CORRECTED_VALIDATION_REPORT.md`
- `docs/PHASE14_PAPER_UPDATE_RECOMMENDATION.md`
- `scripts/41_generate_val_test_predictions.py`
- `scripts/42_clean_calibration_eval.py`
- `scripts/43_clean_ensemble_eval.py`
- `scripts/44_clean_neutral_mitigation_eval.py`
- `scripts/45_fix_fiqa_polarity_and_external_eval.py`
- `results/2026-05-02/phase14_predictions/`
- `results/2026-05-02/phase14_calibration/`
- `results/2026-05-02/phase14_ensemble/`
- `results/2026-05-02/phase14_external/`
- `results/2026-05-02/phase14_neutral/`
- `artifacts/phase14_corrected_validation/`

## 4. Phase identifier confirmation

The next phase of work is **Phase 15** ("Final Reliability Validation Before Paper Writing"). It is **not** Phase 12. Phase 12 is part of the older (Phase 8/9/10/12) lineage that has been superseded by the Phase 13 → Phase 14 corrected-validation pipeline.

All Phase 15 artefacts will be written under timestamped paths consistent with the existing `results/2026-05-02/phase14_*` and `artifacts/phase14_*` conventions, but namespaced with `phase15_*`.
