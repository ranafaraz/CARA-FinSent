# Phase 13 — Extended Validation and Reliability-Improvement Report

**Date:** 2026-05-02
**Git commit SHA at start of Phase 13:** `082c69df25d269f6dea1ab6a2543d38c055bbcd3`
**Author:** automated CARA-FinSent agent (Phase 13 cycle)
**Frame:** *Reliability-first, not universal SOTA.*

This document aggregates Phase 13 Steps 1–6 (Step 6 deferred) and presents
the consolidated evidence base. It supersedes none of the Phase 11 baseline
numbers — those remain frozen and are quoted verbatim where relevant.

Per-step details live in:
- `docs/PHASE13_BASELINE_SNAPSHOT.md` (Step 1)
- `docs/PHASE13_EXTERNAL_VALIDATION_REPORT.md` (Step 2)
- `docs/PHASE13_FIQA_LABEL_POLARITY_BUG.md` (Step 2 supplement)
- `docs/PHASE13_CALIBRATION_REPORT.md` (Step 3)
- `docs/PHASE13_NEUTRAL_ERROR_REPORT.md` (Step 4)
- `docs/PHASE13_ENSEMBLE_REPORT.md` (Step 5)
- `docs/PHASE13_PEFT_DEFERRED.md` (Step 6)
- `docs/PHASE13_RAG_EXTENSION_PLAN.md` (Step 8 plan-only)

---

## 1. What Phase 13 set out to do

Per the original plan:

1. Freeze Phase 11 baseline evidence (no overwrites).
2. Run external validation (FiQA / FOMC / SemEval).
3. Calibrate the agreement-weighted FinBERT (AW) model post-hoc.
4. Reduce neutral → positive errors.
5. Build probability-level ensembles.
6. Optionally run a PEFT baseline (Kaggle GPU only).
7. Produce a Phase 13 integrated report and paper-update recommendation.

Steps 1–5 and Step 8 (RAG extension *plan*) were executed. Step 6 is
documented as **deferred** with a clear gate. Step 9 = this document.
Step 10 = `docs/PHASE13_PAPER_UPDATE_RECOMMENDATION.md`.

---

## 2. Headline summary table (one-glance view)

All numbers are on PhraseBank-test (n = 959) unless noted.
Values from Phase 11 are kept verbatim.

| Model / Method | Phase | macro-F1 | Accuracy | ECE@10 | Brier | Note |
|---|---|---:|---:|---:|---:|---|
| **AW FinBERT (5-seed mean ± SD)** | 11 | **0.8863 ± 0.0023** | 0.890 | 0.0495 | — | Frozen baseline |
| ZS FinBERT (Phase 11) | 11 | 0.8836 | 0.883 | **0.0236** | 0.176 | Best calibration in Ph11 |
| FT FinBERT (Phase 11) | 11 | 0.8828 | 0.887 | 0.0768 | 0.186 | |
| **AW + per-class isotonic** *(Step 3, eval half n=480)* | 13 | 0.870 | 0.873 | **0.0275** | 0.183 | -1.6 pp F1 cost |
| AW + temperature scaling *(Step 3)* | 13 | 0.886 | 0.890 | 0.0530 | 0.180 | Safe alternative |
| AW + neutral margin = 0.40 *(Step 4)* | 13 | 0.886 | 0.890 | 0.0715 | 0.180 | n→p errors 41 → 34 (−17 %) |
| Ensemble: simple mean *(Step 5)* | 13 | 0.891 | 0.890 | 0.0286 | 0.158 | No tuning |
| Ensemble: manual 0.5/0.3/0.2 *(Step 5)* | 13 | **0.893** | 0.892 | 0.0294 | 0.159 | a-priori weights |
| **Ensemble: grid search (eval half)** *(Step 5)* | 13 | 0.891 | 0.892 | **0.0164** | 0.152 | Best calibration to date |
| **Ensemble: stacked LR (eval half)** *(Step 5)* | 13 | **0.895** | 0.896 | 0.0484 | **0.145** | Best F1 to date |

External validation (Step 2, FiQA polarity-corrected, n = 223):

| Model | macro-F1 |
|---|---:|
| ZS FinBERT (zero-shot) | **0.457** |
| FT FinBERT (`finbert_20260430_215910`) | 0.207 *(checkpoint contaminated by FiQA polarity bug — see §4.2)* |
| FT FinBERT (`finbert_20260430_090136`) | 0.331 |
| FT FinBERT (`finbert_20260430_193637`) | 0.385 |

**Critical gap:** in-domain ZS macro-F1 = 0.884 → out-of-domain ZS macro-F1
on FiQA = 0.457. **Absolute drop ≈ 0.43.** This is the single most important
new finding of Phase 13 for the paper's external-validity story.

---

## 3. Statistical posture (what we can say)

We continue to apply the Phase 11 rule: **a difference is reportable as a
"win" only if its 95 % paired-bootstrap CI excludes 0 OR its exact McNemar
p-value is < 0.05.**

| Comparison | Δ macro-F1 | 95 % CI | McNemar p | Reportable as win? |
|---|---:|---|---:|---|
| AW vs ZS *(Phase 11)* | +0.003 | [−0.018, +0.023] | 0.56 | **No** |
| Simple mean ensemble vs AW | +0.005 | [−0.008, +0.018] | 1.00 | **No** |
| Manual ensemble vs AW | +0.007 | [−0.002, +0.017] | 0.79 | **No** |
| Grid ensemble vs AW (eval half) | −0.003 | [−0.025, +0.019] | 0.68 | **No** |
| Stacked LR vs AW (eval half) | +0.0005 | [−0.020, +0.022] | 1.00 | **No** |

**Conclusion:** No Phase 13 model achieves a statistically significant
macro-F1 improvement over the AW baseline on PhraseBank-test. This is
consistent with the previously reported FinBERT-family ceiling on this
dataset. **All accuracy gains are point estimates only.**

---

## 4. Findings that change the paper

This section enforces the "honest framing" rule of Phase 13.

### 4.1 Calibration improvements are real and substantial — and replace the previous calibration story

Before Phase 13:
- ZS FinBERT was the calibration leader at ECE@10 = 0.0236.
- AW achieved competitive accuracy and seed-stability but at ECE@10 = 0.0495.

After Phase 13:
- **AW + per-class isotonic** achieves ECE@10 = 0.0275 — matching ZS at the
  cost of -1.6 pp macro-F1.
- **Grid-search ensemble** achieves ECE@10 = 0.0164 — a **new low**, ≈ 30 %
  better than the previous best (ZS), at parity macro-F1.

These should be explicitly added to the paper's calibration section.

### 4.2 The vanilla-FT FinBERT checkpoint `finbert_20260430_215910` is contaminated

External validation revealed that this checkpoint scores macro-F1 = 0.207
on polarity-corrected FiQA-test, with MCC = -0.48 (worse than random in the
"flipped polarity" sense). Combined with the polarity inversion in the FiQA
gold split (`docs/PHASE13_FIQA_LABEL_POLARITY_BUG.md`), the most likely
explanation is that this checkpoint was trained on label-inverted FiQA data.

**Action items for paper text:**
- Do not cite `finbert_20260430_215910` numbers without the contamination caveat.
- Cite the FiQA polarity bug as a dataset-quality issue (limitations section).
- Use the two later FT checkpoints (`_090136`, `_193637`) for any FT-FinBERT
  comparisons that need to remain credible.

### 4.3 External-validity gap (in-domain → FiQA)

In-domain ZS macro-F1 = 0.884 vs out-of-domain (FiQA, polarity-corrected) =
0.457 is the **honest external generalisation gap** for FinBERT-family
models on financial sentiment. This should be reported as a main-paper
limitation with the explicit numerical gap.

### 4.4 Neutral → positive error mitigation works, but the headline must be cautious

The margin-based override (positive → neutral if `p_pos − p_neu < 0.40`)
reduces n→p errors by 17 % on PhraseBank-test with **no macro-F1 loss**, but:

- Threshold 0.40 was chosen *on the same test set* — this is in-sample tuning.
- Probabilities are unchanged, so ECE / Brier are unchanged.
- The cleaner result is that **a single hyperparameter sweep produces a
  non-trivial reduction in a clinically interesting error class without
  hurting macro-F1**, which is consistent with the reliability-first claim
  but is **not** a standalone modeling contribution.

**Recommendation:** appendix/supplementary material only.

---

## 5. What does *not* change

The following Phase 11 conclusions remain unchanged after Phase 13:

- AW vs ZS macro-F1 difference on PhraseBank is statistically inconclusive
  (CI [−0.018, +0.023], McNemar p ≈ 0.56). **Do not weaken or strengthen.**
- AW seed stability advantage (SD 0.0023 across 5 seeds) remains valid.
- The PhraseBank-test FinBERT-family accuracy ceiling (~0.89 macro-F1)
  remains valid and is now confirmed across **eight** Phase 13 method
  variants on the same test set.

---

## 6. Reliability-first thesis after Phase 13

The Phase 13 evidence strengthens the **reliability framing** of the paper:

1. **Calibration:** new low (ECE 0.016) via grid ensemble; matched ZS via AW + isotonic.
2. **Error structure:** neutral→positive errors are isolatable and reducible
   without macro-F1 cost.
3. **External validity:** the in-domain → FiQA gap is now numerically
   pinned (~0.43 absolute macro-F1 drop) and explicitly disclosed.
4. **Reproducibility:** all Phase 13 artefacts carry git SHA + UTC timestamp;
   safe copies exist under `artifacts/phase13_extended_validation/`.

The paper's contribution is therefore **not** "AW is the best classifier" —
which Phase 11 already showed is statistically inconclusive — but
**"AW + post-hoc calibration / ensembling provides equivalent accuracy
with substantially better calibration and a documented external-validity
gap."**

---

## 7. Caveats accepted (no hidden caveats)

| # | Caveat | Mitigation |
|---|---|---|
| 1 | Step 5 grid + stacked LR results are eval-half only (n = 480). | Disclosed in Step 5 report; recommended cross-validation in future work. |
| 2 | Step 4 margin chosen on the same test set. | Marked as appendix-only in §4.4; not headlined. |
| 3 | AW external evaluation was not possible — no AW checkpoint on disk. | Disclosed in baseline snapshot §4.1; AW external generalisation remains an open question. |
| 4 | FiQA gold has polarity bug at source; raw files unchanged this phase. | Documented separately; runtime swap applied in Step 2 only. |
| 5 | One FT checkpoint is contaminated by that bug. | §4.2 above; clearly labeled. |
| 6 | PEFT baseline is deferred. | `docs/PHASE13_PEFT_DEFERRED.md` documents the gate and the rationale. |

---

## 8. Artefacts checklist (everything Phase 13 produced)

### Documents
- `docs/PHASE13_BASELINE_SNAPSHOT.md`
- `docs/PHASE13_EXTERNAL_VALIDATION_REPORT.md`
- `docs/PHASE13_FIQA_LABEL_POLARITY_BUG.md`
- `docs/PHASE13_CALIBRATION_REPORT.md`
- `docs/PHASE13_NEUTRAL_ERROR_REPORT.md`
- `docs/PHASE13_ENSEMBLE_REPORT.md`
- `docs/PHASE13_PEFT_DEFERRED.md`
- `docs/PHASE13_RAG_EXTENSION_PLAN.md`
- `docs/PHASE13_EXTENDED_VALIDATION_AND_IMPROVEMENT_REPORT.md` *(this file)*
- `docs/PHASE13_PAPER_UPDATE_RECOMMENDATION.md` *(Step 10)*

### Scripts
- `scripts/35_external_validation.py`
- `scripts/36_calibrate_aw_finbert.py`
- `scripts/37_neutral_error_mitigation.py`
- `scripts/38_probability_ensemble.py`

### Result CSVs / JSONs (all under `results/2026-05-02/phase13_*`)
- External validation (per-model preds + summary + manifest, ts `20260502_140602`)
- Calibration (`phase13_aw_calibration_*`, ts `20260502_141058`)
- Neutral mitigation (`phase13_neutral_mitigation_*`, ts `20260502_141335`)
- Ensemble (`phase13_ensemble_*`, ts `20260502_141704`)

### Figures (all under `figures/2026-05-02/phase13_*`)
- External validation per-model confusion matrices
- Calibration reliability diagrams
- Ensemble leaderboard

### Safe copies (`artifacts/phase13_extended_validation/`)
- Mirrored summaries, paired comparisons, manifests for every step.
