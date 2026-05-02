# Phase 14 — Corrected Validation Report

> **Status:** living document. Numbers will be filled in as Phase 14 runs
> complete. The structure and selection rules are fixed at the time of
> writing; only the result cells should change.

**Git commit at start:** `36dae7f`
**Phase 13 reference commit:** `a48bd34`
**Audit document:** [docs/PHASE14_PHASE13_AUDIT.md](PHASE14_PHASE13_AUDIT.md)
**AW recovery report:** [docs/PHASE14_AW_CHECKPOINT_RECOVERY_REPORT.md](PHASE14_AW_CHECKPOINT_RECOVERY_REPORT.md)
**FiQA polarity bug evidence:** [docs/PHASE13_FIQA_LABEL_POLARITY_BUG.md](PHASE13_FIQA_LABEL_POLARITY_BUG.md)

## 1. What Phase 13 got right

These Phase 13 findings stand on their own and require no Phase 14
correction. They are eligible for the main paper as-is.

| Phase 13 finding | Why it stands |
|---|---|
| F1 — FiQA generalization gap | Gap was measured on raw FiQA which we now know had inverted polarity. Phase 14 reruns confirm the gap on the polarity-corrected file (see §7), which strengthens rather than weakens the claim. |
| F2 — FiQA polarity bug | Empirically diagnosed and now addressed by a corrected derived file (§7). |
| F10 — AW external behaviour untested | Phase 13 explicitly framed this as "untested"; no overclaim. |
| F13 — AW vs ZS inconclusive on PhraseBank | Phase 13 reported the paired bootstrap CI `[-0.018, +0.023]` and McNemar p ≈ 0.56 honestly. |
| F14 — AW seed stability | 5-seed mean macro-F1 = 0.8863 ± 0.0023 was reported with uncertainty. |
| F3 — AW seed=13 single-seed point | 0.8860 macro-F1 / 0.8895 accuracy on test reproduces within Phase 11 envelope (subject to Phase 14 retrain confirmation, §3). |

## 2. What Phase 13 was exploratory

These Phase 13 findings were honest exploratory work but used test data
either to fit a calibrator, to choose ensemble weights, or to pick a
neutral threshold. They are **not** safe to put in the paper without a
correction pass.

| Phase 13 finding | Leakage type |
|---|---|
| F4 — AW post-hoc temperature scaling | Calibrator fit on a stratified split of *test* predictions. |
| F5 — AW isotonic calibration | Same fit-on-test issue. |
| F7 — Grid-search ensemble weights | Grid scored on half of test. |
| F8 — Stacked logistic-regression ensemble | Meta-classifier trained on half of test. |
| F9 — Neutral margin = 0.40 | Threshold selected on test. |

Phase 14 replaces each of these with a clean validation-fit / test-eval
result (§4 – §6).

## 3. What Phase 14 corrected — the AW checkpoint

The AW checkpoint underlying the Phase 13 prediction CSV
(`models/finbert_agreement_weighted_linear_20260502_082228/`) was not on
disk locally. We retrained AW with seed=13, linear schedule, 3 epochs,
batch size 16, on CPU only (per the Phase 14 hard rule that forbids
auto-starting RunPod). Detailed log:
[`logs/phase14_aw_train_seed13.log`](../logs/phase14_aw_train_seed13.log).

| Metric | Phase 13 baseline (RunPod GPU) | Phase 14 retrain (CPU) | Within Phase 11 envelope? |
|---|---|---|---|
| seed | 13 | 13 | – |
| macro-F1 (test) | 0.8860 | 0.8648 | **No** (envelope lower bound 0.8817; below by 0.0169 ≈ 7.4 σ) |
| accuracy (test) | 0.8895 | 0.8749 | – |

Envelope = `0.8863 ± 0.0023` (Phase 11 5-seed mean ± std). The Phase 14 CPU retrain is ~0.02 macro-F1 below the GPU envelope. We retain the retrained checkpoint as the canonical AW model for all downstream Phase 14 results, but treat AW deltas as a **lower bound** — the Phase 13 GPU number is plausibly recoverable on GPU but was not reproduced here. See [docs/PHASE14_AW_CHECKPOINT_RECOVERY_REPORT.md](PHASE14_AW_CHECKPOINT_RECOVERY_REPORT.md) for full diagnosis.

## 4. Clean calibration results (val-fit / test-eval)

Source script: [scripts/42_clean_calibration_eval.py](../scripts/42_clean_calibration_eval.py).
Selection rule: calibrators are **fit on validation only** and evaluated
**once on test only**. ECE at 10 bins, macro-F1, and Brier score are
reported on test for each method.

### 4.1 Zero-shot (`ProsusAI/finbert`, val_n=479, test_n=959)

| Method | accuracy | macro-F1 | ECE@10 | Brier | mean conf | fit |
|---|---|---|---|---|---|---|
| uncalibrated | 0.8832 | 0.8836 | 0.0236 | 0.1762 | 0.8746 | none |
| temperature  | 0.8832 | 0.8836 | 0.0233 | 0.1770 | 0.8950 | val |
| Platt        | 0.9030 | 0.9041 | 0.0298 | 0.1479 | 0.9080 | val |
| isotonic     | 0.9030 | 0.9038 | 0.0165 | 0.1518 | 0.9008 | val |

Best-ECE method: **isotonic** (0.0165), with macro-F1 simultaneously
improving from 0.8836 → 0.9038. This is a clean improvement over the
Phase 11 ZS baseline (ECE 0.0236, macro-F1 0.8836) — and the
calibrator never saw test data.

### 4.2 Fine-tuned FinBERT (val_n=479, test_n=959)

Checkpoint: `models/finbert_20260501_193637`.

| Method | accuracy | macro-F1 | ECE@10 | Brier | mean conf | fit |
|---|---|---|---|---|---|---|
| uncalibrated | 0.8895 | 0.8884 | 0.0344 | 0.1577 | 0.8862 | none |
| temperature  | 0.8895 | 0.8884 | 0.0301 | 0.1582 | 0.9017 | val |
| Platt        | 0.8916 | 0.8881 | 0.0390 | 0.1584 | 0.9023 | val |
| isotonic     | 0.8936 | 0.8958 | 0.0138 | 0.1628 | 0.9058 | val |

Best-ECE method: **isotonic** (0.0138), with macro-F1 simultaneously
improving from 0.8884 → 0.8958. Clean win.

### 4.3 Agreement-weighted FinBERT (val_n=479, test_n=959)

Checkpoint: `models/finbert_agreement_weighted_linear_20260502_145813/checkpoint-210` (best by val macro-F1, epoch 1).

| Method | accuracy | macro-F1 | ECE@10 | Brier | mean conf | fit |
|---|---|---|---|---|---|---|
| uncalibrated | 0.8749 | 0.8648 | 0.0418 | 0.1812 | 0.9167 | none |
| temperature  | 0.8749 | 0.8648 | 0.0455 | 0.1796 | 0.9103 | val |
| Platt        | 0.8822 | 0.8813 | 0.0385 | 0.1707 | 0.9106 | val |
| isotonic     | 0.8780 | 0.8724 | 0.0421 | 0.1767 | 0.9120 | val |

Best-ECE method: **Platt** (0.0385); macro-F1 simultaneously improves
from 0.8648 → 0.8813. Note: AW uncalibrated is below ZS uncalibrated
(0.8836) and FT uncalibrated (0.8884) on this Phase 14 CPU retrain;
AW does **not** dominate on PhraseBank under this checkpoint.

## 5. Clean ensemble results (val-tune / test-eval)

Source script: [scripts/43_clean_ensemble_eval.py](../scripts/43_clean_ensemble_eval.py).
Selection rule: ensemble weights or stacking meta-classifier are
**chosen on validation only** and evaluated **once on test only**.
Paired bootstrap and exact McNemar tests are computed against the AW
single-model baseline.

| Method | macro-F1 | ECE@10 | tune split | bootstrap CI vs AW (Δ macro-F1) | McNemar p |
|---|---|---|---|---|---|
| AW baseline | 0.8648 | 0.0418 | none | – | – |
| ZS baseline | 0.8836 | 0.0236 | none | – | – |
| FT baseline | 0.8884 | 0.0344 | none | – | – |
| simple mean | 0.8940 | 0.0339 | none | +0.0292 [0.0130, 0.0471] | 0.0029 |
| manual (AW 0.5 / ZS 0.3 / FT 0.2) | 0.8845 | 0.0389 | a-priori | +0.0197 [0.0064, 0.0339] | 0.0192 |
| val-tuned grid (step 0.1) → AW 0.4 / ZS 0.6 / FT 0.0 | 0.8953 | 0.0274 | val | +0.0305 [0.0114, 0.0502] | 0.0226 |
| stacked LR | 0.8924 | 0.0382 | val | +0.0276 [0.0130, 0.0443] | 0.0009 |

All non-baseline rows show statistically significant improvement over
the AW single-model baseline (paired bootstrap CI excludes zero;
McNemar exact two-sided p < 0.05). The val-tuned grid winner places
zero weight on FT and 60 % on ZS, indicating that on PhraseBank the
strongest two contributors to a clean ensemble are AW and ZS, not
FT. Caveat: AW baseline here is the Phase 14 CPU-retrained model
(macro-F1 0.0212 below the GPU original); on GPU the AW baseline
would likely be ~0.886, which would shrink the ensemble Δ to ~+0.009
and may move some comparisons below the significance threshold. Treat
ensemble Δ values as **upper bounds**.

## 6. Clean neutral mitigation results (val-select / test-eval)

Source script: [scripts/44_clean_neutral_mitigation_eval.py](../scripts/44_clean_neutral_mitigation_eval.py).
Selection rule: scan margins {0.02, 0.04, 0.06, 0.08, 0.10, 0.20, 0.30,
0.40, 0.50}; select the largest margin on **validation** that maximises
neutral recall subject to validation macro-F1 drop ≤ 0.002. Apply that
margin to **test** once.

| Quantity | Value |
|---|---|
| selected margin | 0.00 (no override) |
| val macro-F1 (selected) | 0.8898 |
| test macro-F1 (selected) | 0.8648 |
| test neutral recall (selected) | 0.9042 |
| test n→p errors (selected) | 51 |
| overridden positive→neutral count (test) | 0 |

No positive margin in the scan beat the baseline on validation under
the ≤ 0.002 macro-F1-drop constraint. Phase 13's reported margin = 0.40
was chosen on test; on validation the same margin loses 0.016 macro-F1
(0.890 → 0.873). The clean rule **rejects** the Phase 13 neutral
mitigation result. Recommended paper action: relegate to limitations
or remove.

## 7. Clean FiQA polarity-corrected validation

Source script: [scripts/45_fix_fiqa_polarity_and_external_eval.py](../scripts/45_fix_fiqa_polarity_and_external_eval.py).
Polarity-corrected derived file:
[data/processed/gold/latest_gold_fiqa_split_polarity_corrected.csv](../data/processed/gold/latest_gold_fiqa_split_polarity_corrected.csv).
Manifest:
[data/processed/gold/latest_gold_fiqa_split_polarity_corrected_manifest.json](../data/processed/gold/latest_gold_fiqa_split_polarity_corrected_manifest.json).

The raw file is preserved; we only swapped `positive ↔ negative` in a
**separate derived file** and kept neutral untouched. Pre-correction
counts: `{negative: 682, positive: 345, neutral: 84}`. Post-correction
counts: `{positive: 682, negative: 345, neutral: 84}`. 682 rows had
their label flipped; 345 had the symmetric flip applied; 84 neutral
rows are unchanged.

Evaluated on the test split of the corrected derived file (n=223).

| Family | n | accuracy | macro-F1 | MCC | ECE@10 |
|---|---|---|---|---|---|
| zero-shot finbert | 223 | 0.4664 | 0.4567 | 0.3391 | 0.3321 |
| fine-tuned finbert (`models/finbert_20260501_193637`) | 223 | 0.3677 | 0.3850 | 0.2992 | 0.4533 |
| agreement-weighted finbert (Phase 14 CPU retrain) | 223 | 0.3229 | 0.3392 | 0.2525 | 0.5320 |

PhraseBank test macro-F1 references for comparison: ZS = 0.8836, FT =
0.8884, AW = 0.8648 (Phase 14 CPU retrain). The drop on polarity-corrected FiQA is substantial:

- ZS: 0.8836 → 0.4567 (Δ = -0.4269)
- FT: 0.8884 → 0.3850 (Δ = -0.5034) — fine-tuning on PhraseBank
  appears to *hurt* FiQA performance, consistent with overfitting to
  PhraseBank's headline-style register.
- AW: 0.8648 → 0.3392 (Δ = -0.5256) — agreement weighting does **not**
  improve cross-dataset generalisation on this checkpoint; it is the
  worst of the three on FiQA.

The FiQA generalization gap (Phase 13 finding F1) is **confirmed on a
clean dataset**. ECE@10 also degrades severely on FiQA, indicating that
both models are overconfidently wrong on out-of-distribution finance
text. This justifies the paper's headline reliability framing.

## 8. Updated paper inclusion decision

Recommendations are formalised in
[docs/PHASE14_PAPER_UPDATE_RECOMMENDATION.md](PHASE14_PAPER_UPDATE_RECOMMENDATION.md).
Headline rules from this report:

1. ZS isotonic calibration on val (ECE 0.0236 → 0.0165, macro-F1 0.8836
   → 0.9038) is **paper-safe** and may go in the main paper.
2. AW calibration result depends on the retrain envelope check (§3) and
   the AW val-fit calibration outcome (§4.3).
3. Ensemble results may go in the main paper only if the val-tuned
   variant beats AW baseline with bootstrap CI excluding zero **and**
   McNemar p < 0.05; otherwise appendix.
4. Neutral mitigation goes in the appendix unless §6 selects a positive
   margin that improves neutral recall without macro-F1 loss on test.
5. FiQA polarity bug is reported in the limitations / appendix and the
   corrected derived file is shipped as part of the data release.

## 9. Honest framing rule

Where a Phase 14 result weakens or invalidates a Phase 13 claim, that
finding is recorded here under §3–§7 with the original Phase 13 number
shown next to the corrected number. Nothing is hidden.

## 10. Provenance

Every artefact referenced in §4–§7 carries `git_commit_sha` and
`generated_at_utc` columns or fields. Mirrored copies are written under
`artifacts/phase14_corrected_validation/` for inclusion in the paper
reproducibility appendix.
