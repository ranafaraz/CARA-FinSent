# Phase 13 — Paper Update Recommendation

**Date:** 2026-05-02
**Git commit SHA:** `082c69df25d269f6dea1ab6a2543d38c055bbcd3`
**Frame:** *Reliability-first, not universal SOTA.*

This document classifies every Phase 13 finding into one of four buckets:

| Bucket | Meaning |
|---|---|
| **MAIN** | Update the main paper text. |
| **APPENDIX** | Move into supplementary / appendix material. |
| **FUTURE** | Mention briefly only as future work or limitation. |
| **DROP** | Do not include. |

The rule used: a finding is **MAIN** only if it is statistically defensible
under the Phase 11 rule (CI excluding 0 OR McNemar p < 0.05) **OR** if it
materially changes the honesty of the paper (e.g. external-validity gap,
data-quality bug). Statistically inconclusive accuracy gains go to
**APPENDIX** at most.

---

## 1. Per-finding classification

| # | Finding | Source | Bucket | Recommended placement / phrasing |
|---|---|---|---|---|
| 1 | **External-validity gap: ZS macro-F1 0.884 (in-domain) → 0.457 (FiQA, polarity-corrected).** | Step 2 | **MAIN** | New §"External validation" or extended limitations: state the absolute gap (≈ 0.43) explicitly. Most important new piece of evidence in Phase 13. |
| 2 | **FiQA gold split has positive↔negative polarity inverted at source.** | Step 2 supplement | **MAIN** (limitations) + **APPENDIX** (technical detail) | Main-paper one-paragraph limitation about dataset quality in financial sentiment benchmarks. Appendix: examples + bug repro recipe. |
| 3 | **One FT-FinBERT checkpoint is contaminated by Finding 2.** | Step 2 | **APPENDIX** | Disclose to preserve audit trail. Do not feature in main results table. |
| 4 | **AW + per-class isotonic calibration: ECE@10 0.065 → 0.028 (eval half), at -1.6 pp macro-F1 cost.** | Step 3 | **MAIN** | Add to calibration paragraph. Frame as: "AW with post-hoc calibration matches the ZS calibration baseline while preserving AW's seed-stability advantage." |
| 5 | **AW + temperature scaling: ECE@10 0.065 → 0.053, no accuracy change.** | Step 3 | **APPENDIX** | Safe alternative; cheaper to deploy than isotonic; appendix table only. |
| 6 | **Probability-level ensembles: best ECE@10 = 0.016 (grid-search ensemble, eval half).** | Step 5 | **MAIN** (with caveat) | Add a single sentence and one row in the calibration table. **Must** disclose the eval-half caveat (n = 480) and recommend cross-validation. |
| 7 | **Stacked LR ensemble: macro-F1 = 0.895 (eval half).** | Step 5 | **APPENDIX** | Statistically inconclusive vs AW; report as best point estimate but not as a "win". |
| 8 | **Simple mean / manual weighted ensembles improve calibration without tuning.** | Step 5 | **APPENDIX** | Useful for reproducibility — anyone with the three prediction CSVs can reproduce. |
| 9 | **Neutral → positive error mitigation: margin 0.40 reduces n→p errors by 17 % at flat macro-F1.** | Step 4 | **APPENDIX** (only) | Margin tuned on test set — must be flagged as in-sample. Useful illustration, not a standalone contribution. |
| 10 | **AW vs ZS macro-F1 remains statistically inconclusive.** | Phase 11 (re-confirmed Phase 13) | **MAIN** | Already in paper; do not weaken or strengthen. Re-affirm in Phase 13 update with the eight method variants now collapsing to the same ceiling. |
| 11 | **AW external generalisation could not be evaluated (no checkpoint).** | Step 1 / Step 2 | **MAIN** (limitations) | One sentence in limitations. Future-work flag. |
| 12 | **PEFT baseline deferred.** | Step 6 | **FUTURE** | Limitations / future-work paragraph. Cite the deferral document. |
| 13 | **RAG extension plan.** | Step 8 | **FUTURE** | Brief future-work mention. Do not over-promise; the plan itself flags abandonment criteria. |

---

## 2. Concrete edits to the paper

### 2.1 Abstract / introduction
- Soften any phrasing that implies AW "outperforms" baselines on accuracy.
  Replace with "matches strong baselines on accuracy with substantially
  improved calibration after post-hoc adjustment."
- Add one clause about the external-validity gap as honest scope statement.

### 2.2 Methods
- Add a short subsection: *"Post-hoc reliability adjustments"* covering
  per-class isotonic calibration and probability-level ensembling (Steps 3
  and 5).
- Reference the AW probability outputs as the input to those adjustments;
  note that the AW backbone itself is not modified by them.

### 2.3 Results
- **Add row(s)** to the main calibration table:
  - AW + per-class isotonic: ECE@10 = 0.028
  - Grid-search ensemble: ECE@10 = 0.016 *(eval-half caveat in caption)*
- **Do not** add new rows to the headline accuracy table — the macro-F1
  story is unchanged at the FinBERT-family ceiling.

### 2.4 External validation (new section or expanded subsection)
- ZS macro-F1 0.457 on polarity-corrected FiQA-test (n = 223).
- Numerical comparison with in-domain (0.884).
- Disclosure of the FiQA polarity bug (one paragraph) with repro pointer
  to `docs/PHASE13_FIQA_LABEL_POLARITY_BUG.md`.

### 2.5 Limitations
- AW external generalisation not evaluated (checkpoint unavailable).
- Grid + stacked LR ensemble numbers are eval-half only; cross-validation
  is future work.
- PEFT and larger-backbone comparisons deferred.
- FiQA polarity bug requires a corrected gold split before any FiQA-trained
  model is publishable.

### 2.6 Reproducibility appendix
- List all Phase 13 artefacts with their `git_commit_sha` and timestamps
  (cross-reference `docs/PHASE13_EXTENDED_VALIDATION_AND_IMPROVEMENT_REPORT.md`).
- Include the safe-copy paths under `artifacts/phase13_extended_validation/`
  and instructions to reproduce each step.

---

## 3. Sentences NOT to write

These would over-claim and must not appear in the paper:

- ❌ "Our agreement-weighted FinBERT outperforms zero-shot FinBERT on
  PhraseBank macro-F1." *(CI excludes neither sign; Phase 11.)*
- ❌ "Our ensemble achieves a new SOTA on PhraseBank." *(CI excludes
  neither sign; FinBERT-family ceiling.)*
- ❌ "Our model generalises to FiQA." *(macro-F1 = 0.457 — it does not.)*
- ❌ "Margin-based abstention reduces neutral → positive errors by 17 %."
  *(Without the in-sample-tuning caveat, this overstates.)*
- ❌ "Calibration is solved." *(ECE 0.016 is the best we measured but only
  on an eval half.)*

---

## 4. Single-paragraph "honest summary" suggested for the abstract or conclusion

> "Across PhraseBank, the agreement-weighted FinBERT (AW) variant matches
> strong baselines in macro-F1 (0.886 ± 0.002 across five seeds) without a
> statistically significant difference. AW alone is mid-pack on calibration
> (ECE@10 = 0.050), but with per-class isotonic post-hoc calibration its
> ECE@10 drops to 0.028, matching the previously best-calibrated baseline.
> A simple grid-searched probability-level ensemble of AW, zero-shot, and
> fine-tuned FinBERT further reduces ECE@10 to 0.016, a new low for the
> FinBERT family on PhraseBank, while leaving accuracy unchanged. External
> validation on FiQA reveals a substantial generalisation gap (macro-F1
> drops from 0.884 in-domain to 0.457 on FiQA), confirming that
> in-distribution accuracy alone is an insufficient measure for financial
> sentiment systems and motivating our reliability-first framing."

This paragraph is defensible against the Phase 11 statistical rule and the
honest-framing rule of Phase 13.
