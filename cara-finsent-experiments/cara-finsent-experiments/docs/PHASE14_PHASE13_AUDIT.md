# Phase 14 — Phase 13 Audit

**Date:** 2026-05-02
**Latest reviewed commit:** `36dae7f` *(`a48bd34` is the Phase 13 commit; `36dae7f` adds data-audit files only)*
**Phase 13 commit under audit:** `a48bd3499b4e055c51cbb6f40a199df0697f7fab`
**Auditor:** automated CARA-FinSent agent (Phase 14 cycle)
**Frame:** *Reliability-first. Smaller clean result is better than larger test-tuned result.*

This document is the gating step of Phase 14: it classifies every Phase 13
finding as paper-clean, exploratory, or in need of correction, and decides
the go/no-go for the paper update.

---

## 1. Phase 13 findings — classification

| # | Finding | Phase 13 status | Phase 14 verdict | Required correction |
|---|---|---|---|---|
| F1 | FiQA external-validity gap (ZS macro-F1 0.884 → 0.457; absolute drop 0.43) | reported | **Main paper** | Re-run on a polarity-corrected derived FiQA file (raw unchanged). |
| F2 | FiQA gold split has positive↔negative inverted at source | reported in `PHASE13_FIQA_LABEL_POLARITY_BUG.md` | **Limitations + appendix** | Materialise a corrected derived CSV + manifest. Do not mutate raw. |
| F3 | One FT-FinBERT checkpoint (`finbert_20260430_215910`) is contaminated by F2 | disclosed | **Drop** from main numbers | Use the two later checkpoints; flag in appendix. |
| F4 | AW + per-class isotonic calibration: ECE@10 0.065 → 0.028 | computed on **stratified split of test predictions** (no val probs available) | **NEEDS CORRECTION** | Refit on a true validation split; evaluate once on test. |
| F5 | AW + temperature scaling: ECE@10 0.065 → 0.053 | same leakage as F4 | **NEEDS CORRECTION** | Same fix as F4. |
| F6 | Probability ensembles — simple-mean and manual-weighted (full-test) | computed on full test (a-priori weights) | **Acceptable as-is** for the manual/simple variants (no test-set tuning). | Re-confirm under Phase 14 pipeline; report stat tests. |
| F7 | Grid-search ensemble (ECE@10 0.016, eval half) | weights chosen on **half of test** | **NEEDS CORRECTION** | Re-tune weights on validation only. |
| F8 | Stacked-LR ensemble (macro-F1 0.895, eval half) | meta-classifier fit on **half of test** | **NEEDS CORRECTION** | Re-fit on validation predictions only. |
| F9 | Neutral-margin override at 0.40 reduces n→p errors 17 % | margin **selected on test set** | **NEEDS CORRECTION** | Select margin on validation only; report once on test. |
| F10 | AW external generalisation untested | AW checkpoint missing locally | **Limitations** + recovery action | Recover or retrain AW checkpoint. |
| F11 | PEFT baseline | deferred | **Future work** | No change. |
| F12 | RAG extension | plan-only document | **Future work** | No change. |
| F13 | AW vs ZS macro-F1 statistically inconclusive | quoted from Phase 11 | **Main paper** (unchanged) | None. |
| F14 | AW seed-stability (SD 0.0023 across 5 seeds) | from Phase 11 | **Main paper** (unchanged) | None. |

---

## 2. Specific evidence of leakage

### 2.1 Calibration (Phase 13 Step 3)

Source: `docs/PHASE13_CALIBRATION_REPORT.md` and
`scripts/36_calibrate_aw_finbert.py`.

> "AW source: `…/finbert_agreement_weighted_predictions_20260502_082228.csv`.
>  Stratified 50/50 split (calib_n=479, eval_n=480, seed=42)."

The AW checkpoint was unavailable locally, so no AW val-split probabilities
existed on disk. The Phase 13 author split the **test** predictions in half
and fit calibrators on one half, evaluated on the other half. Even though
the *evaluation* half is unseen by the calibrator, the *split* itself is
drawn from the same test set that is used to report the headline AW number.
This is **calibration leakage** for paper purposes and must be re-done with
a true validation split.

### 2.2 Ensembles (Phase 13 Step 5)

Source: `docs/PHASE13_ENSEMBLE_REPORT.md` and
`scripts/38_probability_ensemble.py`.

The grid-search ensemble (best ECE@10 = 0.016) and stacked-LR ensemble
(best macro-F1 = 0.895) used a 50/50 stratified split of the **test**
predictions for hyperparameter selection. Same leakage class as §2.1.

### 2.3 Neutral mitigation (Phase 13 Step 4)

Source: `docs/PHASE13_NEUTRAL_ERROR_REPORT.md` and
`scripts/37_neutral_error_mitigation.py`.

The recommended margin (0.40) was selected by sweeping margins on the
**full test set** and inspecting the resulting macro-F1 and n→p error count.
This is in-sample tuning. Phase 13 already flagged it ("appendix only,
illustration"); Phase 14 will additionally produce a clean
val-select / test-evaluate version.

### 2.4 AW checkpoint missing

`models/finbert_agreement_weighted_linear_20260502_082228/` is referenced in
the Phase 13 AW manifest but **does not exist on disk**. None of the existing
`models/finbert_*` directories have an AW marker (no `agreement` in any name,
no `agreement_weighted: true` field in any training_config).

The AW checkpoint was apparently produced on RunPod (Phase 8) and the
prediction CSV was synced back, but the model weights were not. Without
weights:
- AW val-split probabilities cannot be generated.
- AW external validation (FiQA) cannot be performed.
- Any new AW-related metric cannot be reproduced from inputs.

Phase 14 must either recover the checkpoint or formally document the gap.

---

## 3. What Phase 13 got right (no correction needed)

- **Honest framing**: every Phase 13 doc explicitly disclosed eval-half
  caveats. The leakage was admitted, not hidden.
- **External validation**: ZS-on-FiQA result is methodologically clean
  (model trained on ProsusAI weights, evaluated zero-shot on a different
  dataset). The polarity-correction step was done at runtime and disclosed.
- **Statistical posture**: Phase 13 paired bootstraps and McNemar tests are
  valid; their negative findings ("no statistically significant gain") still
  hold under Phase 14's cleaner pipeline (we expect the same negatives).
- **Data-quality bug discovery**: catching the FiQA polarity inversion is
  itself a positive contribution — it strengthens the limitations story and
  is reusable by other researchers.

---

## 4. Findings ready for the paper without further work

| Finding | Bucket |
|---|---|
| F1 — FiQA external-validity gap | **Main paper** |
| F2 — FiQA polarity bug | **Limitations + appendix** |
| F10 — AW external validation untested | **Limitations** |
| F13 — AW vs ZS statistically inconclusive | **Main paper** (Phase 11) |
| F14 — AW seed stability | **Main paper** (Phase 11) |

Everything else is gated on Phase 14 corrections.

---

## 5. Phase 14 corrective work plan

1. **Step 2** — Generate val + test prediction CSVs for ZS, FT, and AW
   FinBERT under the strict schema in PHASE14_AGENT_INSTRUCTIONS §Step 2.
2. **Step 3** — Recover AW checkpoint by retraining on the same split with
   the same seed as Phase 13 (`seed=13`, `weight_schedule=linear`,
   `num_epochs=3`, `batch_size=16`, `lr=2e-5`). On CPU this takes ~60–90
   minutes; preferred over starting RunPod automatically. Document the
   recovery in `docs/PHASE14_AW_CHECKPOINT_RECOVERY_REPORT.md`.
3. **Step 4** — Re-run calibration. Fit on val predictions only; evaluate
   on test. Methods: temperature, Platt, per-class isotonic.
4. **Step 5** — Re-run ensemble. Tune weights / fit stacked LR on val
   only; evaluate once on test. Compare via paired bootstrap + McNemar to
   AW baseline.
5. **Step 6** — Re-run neutral mitigation. Sweep margins on val; pick
   under the criterion *"max neutral recall subject to macro-F1 drop ≤
   0.002"*; evaluate once on test.
6. **Step 7** — Materialise polarity-corrected FiQA derived CSV. Re-run
   ZS external validation; re-run AW external validation if checkpoint
   recovered.
7. **Step 8** — Integrated Phase 14 corrected report.
8. **Step 9** — Updated paper-update recommendation.

---

## 6. Go / no-go decision

**Decision: GO** — proceed with Phase 14 corrections.

**Conditions on the paper update**:

- The paper may be updated **only after** Step 8 (the corrected integrated
  report) is written.
- Calibration / ensemble / neutral mitigation numbers in the paper must
  cite the **Phase 14 corrected** results, not the Phase 13 numbers.
- Phase 13 numbers move to the appendix as exploratory evidence with
  explicit "test-tuned" labels.
- If AW checkpoint cannot be recovered, AW-derived calibration / ensemble
  results are reported as *deferred*, and the paper's calibration story
  is told using ZS + FT only (still defensible).

**No-go triggers** (would block paper update):

- A Phase 14 result contradicts a frozen Phase 11 number (e.g. ZS macro-F1
  drops below the published 0.8836 on the same split). Action: investigate
  before paper update.
- A new statistically significant negative result emerges (e.g. cleaner
  ensemble does not beat AW). Action: report honestly, do not headline.

None of those triggers are tripped at audit time.
