# Phase 14 — Paper Update Recommendation

> **Status:** living document. Final cell values depend on the runs
> still completing in
> [docs/PHASE14_CORRECTED_VALIDATION_REPORT.md](PHASE14_CORRECTED_VALIDATION_REPORT.md).
> The structural decisions and reasoning below are fixed.

**Default rule:** *A smaller clean result is better than a larger
test-tuned result.* When Phase 14 cannot defend a Phase 13 claim, the
Phase 13 number does not go in the main paper.

## 1. Summary table — what goes where in Research Paper 3

| Topic | Phase 13 status | Phase 14 status | Recommended placement |
|---|---|---|---|
| AW vs ZS on PhraseBank | Inconclusive (CI crosses 0, McNemar p ≈ 0.56) | Same; reproduced on retrain | **Main**, framed as "AW is competitive but not statistically distinguishable from zero-shot at n=959". |
| AW seed stability | 5-seed mean 0.8863 ± 0.0023 | Confirmed | **Main**, single sentence with σ. |
| ZS post-hoc calibration | Tested on test (leak) | Clean val-fit / test-eval: ECE 0.0236 → 0.0165 (isotonic), macro-F1 0.8836 → 0.9038 | **Main**, §"Reliability". Cite isotonic. |
| AW post-hoc calibration | Tested on test (leak) | Clean val-fit / test-eval: ECE 0.0418 → 0.0385 (Platt), macro-F1 0.8648 → 0.8813 | **Appendix only**. AW baseline is itself depressed by the Phase 14 CPU retrain (−0.02 vs Phase 11 envelope), so the calibrated number is not strictly comparable to the GPU baseline. ZS isotonic is the headline calibration result. |
| Ensembles (simple mean / fixed weights) | Reported | Re-evaluated cleanly. Val-tuned grid (AW 0.4 / ZS 0.6 / FT 0.0) reaches macro-F1 0.8953 on test, +0.0305 vs AW (CI [0.0114, 0.0502], McNemar p = 0.023). | **Main**, with caveat that AW baseline is on the Phase 14 CPU retrain (−0.02 below GPU envelope) so reported Δ is an upper bound. |
| Stacked LR ensemble | Trained on test (leak) | Re-trained on val: macro-F1 0.8924, ECE 0.0382 (worse than simple mean and val-grid). | **Appendix**. Does not dominate simpler ensembles. |
| Neutral margin | Picked on test at 0.40 | Val rule selects margin = 0.00 (no override beats baseline under ≤ 0.002 macro-F1-drop constraint). | **Limitations / appendix**. The Phase 13 result does not survive validation gating. |
| FiQA generalization gap | Reported on raw FiQA | Reproduced on polarity-corrected FiQA | **Main**. The clean number replaces the raw number. |
| FiQA polarity bug | Diagnosed | Corrected derived file shipped | **Limitations / appendix**, with reference to the corrected derived file. |
| PEFT (LoRA / adapters) | Not run | Not run | **Future work**. |
| RAG augmentation | Not run | Not run | **Future work**. |

## 2. Specific text changes recommended

### 2.1 Main paper, §"Calibration"

Replace any phrase that compares calibrated AW to AW-on-test with the
clean ZS isotonic numbers from the Phase 14 corrected validation
report. Suggested wording:

> "Post-hoc isotonic calibration of the zero-shot FinBERT, fit on the
> 479-row PhraseBank validation split and evaluated once on the
> 959-row held-out test split, lowers ECE@10 from 0.0236 to 0.0165 and
> raises test macro-F1 from 0.8836 to 0.9038. The calibrator was never
> exposed to test data."

If the AW calibration cell in §4.3 also produces a clean ECE drop with
its bootstrap CI excluding zero, append a parallel sentence for AW.
Otherwise omit the AW calibration claim from the main paper.

### 2.2 Main paper, §"Ensembles"

If the validation-tuned ensemble (§5 of the corrected validation
report) beats AW baseline with a paired-bootstrap CI excluding zero
**and** McNemar p < 0.05, include one sentence with the val-tuned
weights and the test macro-F1. Otherwise, replace any ensemble claim
with: "Probability-level ensembles of ZS, FT, and AW were not
statistically distinguishable from the AW single model on PhraseBank
test." This is the honest fallback.

### 2.3 Main paper, §"Generalization to FiQA"

Use the polarity-corrected numbers from §7. State that the original raw
FiQA file had `positive` and `negative` swapped at source; cite the
corrected derived file path. Suggested wording:

> "We re-evaluated all three model families on a polarity-corrected
> derivative of the FiQA gold split (raw labels swapped per the
> diagnosis recorded in `docs/PHASE13_FIQA_LABEL_POLARITY_BUG.md`).
> The corrected derived file is shipped alongside the codebase as
> `data/processed/gold/latest_gold_fiqa_split_polarity_corrected.csv`
> with manifest. The PhraseBank → FiQA generalization gap persists
> after correction: macro-F1 drops from 0.8836 → 0.4567 for zero-shot
> FinBERT, 0.8884 → 0.3850 for fine-tuned FinBERT, and 0.8648 →
> 0.3392 for the agreement-weighted variant (n = 223 corrected FiQA
> test rows). All three families are also severely miscalibrated on
> the out-of-distribution split (ECE@10 = 0.33 / 0.45 / 0.53
> respectively)."

### 2.4 Appendix — Reliability and ablations

Move into the appendix, with full tables but smaller emphasis:

- The full clean-calibration sweep (uncalibrated, temperature, Platt,
  isotonic) for all three model families.
- The full neutral-margin sweep with val/test pairs at every margin.
- The simple-mean and a-priori-weighted ensembles, even if the
  val-tuned variant goes in the main paper.
- The Phase 11 5-seed AW spread.

### 2.5 Appendix — Limitations

State explicitly:

- "External validation on FiQA used a polarity-corrected derived file
  because the raw FiQA gold labels were inverted at source."
- "AW retrain for Phase 14 was performed on CPU only; we did not run
  AW on a GPU during this phase."
- "Confidence intervals are bootstrap intervals at B = 2000."

### 2.6 Future work section

Add a short paragraph naming PEFT (LoRA / adapter) and RAG variants of
CARA-FinSent as planned next steps; do not present numbers we did not
collect.

## 3. Conservative defaults applied

The recommendations above implement these defaults verbatim from the
Phase 14 hard rules:

- "Do not overclaim. A smaller clean result is better than a larger
  test-tuned result."
- Anything tuned on test in Phase 13 is downgraded to appendix or
  removed unless re-validated cleanly in Phase 14.
- Any Phase 14 result that depends on the AW retrain checkpoint is
  reported with the retrain provenance from
  [docs/PHASE14_AW_CHECKPOINT_RECOVERY_REPORT.md](PHASE14_AW_CHECKPOINT_RECOVERY_REPORT.md).

## 4. Provenance for paper reviewers

Reviewers can reproduce every Phase 14 number by running, in order:

```
python scripts/41_generate_val_test_predictions.py --model_family zero_shot
python scripts/41_generate_val_test_predictions.py --model_family fine_tuned --checkpoint <ft_dir>
python scripts/41_generate_val_test_predictions.py --model_family agreement_weighted --checkpoint <aw_dir>
python scripts/42_clean_calibration_eval.py --val_predictions <zs_val> --test_predictions <zs_test> --model_label zs
python scripts/42_clean_calibration_eval.py --val_predictions <ft_val> --test_predictions <ft_test> --model_label ft
python scripts/42_clean_calibration_eval.py --val_predictions <aw_val> --test_predictions <aw_test> --model_label aw
python scripts/43_clean_ensemble_eval.py --zs_val <...> --zs_test <...> --ft_val <...> --ft_test <...> --aw_val <...> --aw_test <...>
python scripts/44_clean_neutral_mitigation_eval.py --val_predictions <aw_val> --test_predictions <aw_test>
python scripts/45_fix_fiqa_polarity_and_external_eval.py --ft_checkpoint <ft_dir> --aw_checkpoint <aw_dir>
```

Every output CSV/JSON/PNG carries `git_commit_sha` and
`generated_at_utc`. The mirrored copies under
`artifacts/phase14_corrected_validation/` are the canonical artefact
set for the paper reproducibility appendix.
