# Phase 13 — Step 2 External Validation Report (FiQA gold split)

**Date:** 2026-05-02
**Git commit SHA:** `082c69df25d269f6dea1ab6a2543d38c055bbcd3`
**Script:** `scripts/35_external_validation.py`
**Run timestamp (canonical):** `20260502_140602`
**Eval split:** FiQA gold *test* (n = 223)
**Label distribution (post polarity-correction):** positive 137 / negative 69 / neutral 17

> **Important precondition:** The FiQA gold split shipped with positive↔negative
> labels inverted at the source. `scripts/35_external_validation.py` applies a
> runtime swap (`--swap_pos_neg`, default True). Every metric below is computed
> against the **polarity-corrected** labels. See
> [PHASE13_FIQA_LABEL_POLARITY_BUG.md](PHASE13_FIQA_LABEL_POLARITY_BUG.md) for
> full evidence and follow-up plan.

---

## 1. Headline results — zero-shot vs vanilla fine-tuned (per checkpoint)

Source CSV: `results/2026-05-02/phase13_external_validation_summary_20260502_140602.csv`
Safe copy: `artifacts/phase13_extended_validation/external_validation_summary_20260502_140602.csv`

| Model | Family | n | Accuracy | macro-F1 | weighted-F1 | MCC | ECE@10 | Brier | mean conf |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FinBERT zero-shot (`ProsusAI/finbert`) | zero-shot | 223 | 0.466 | **0.457** | 0.537 | **+0.339** | 0.332 | 0.799 | 0.798 |
| FinBERT FT `finbert_20260430_215910` | finetuned | 223 | 0.157 | 0.207 | 0.145 | −0.477 | 0.807 | 1.627 | 0.960 |
| FinBERT FT `finbert_20260501_090136` | finetuned | 223 | 0.529 | 0.331 | 0.477 | +0.115 | 0.079 | 0.631 | 0.583 |
| FinBERT FT `finbert_20260501_193637` | finetuned | 223 | 0.368 | 0.385 | 0.453 | +0.299 | 0.453 | 0.999 | 0.821 |

Agreement-weighted FinBERT was **not evaluated**: no AW checkpoint exists on
disk (Kaggle-trained, weights not synced back). This gap is recorded in the
manifest under `agreement_weighted_skipped_reason` and in
[PHASE13_BASELINE_SNAPSHOT.md](PHASE13_BASELINE_SNAPSHOT.md) §4.1.

---

## 2. In-domain → out-of-domain generalisation gap

Comparing zero-shot FinBERT across domains, on the same model weights:

| Domain | macro-F1 | ECE@10 | n_test |
|---|---:|---:|---:|
| PhraseBank in-domain (Phase 8/11 baseline) | **0.884** | **0.0236** | 959 |
| FiQA out-of-domain (this report, polarity-corrected) | **0.457** | 0.332 | 223 |

**Macro-F1 drop ≈ 0.43 absolute (≈ 48% relative).** ECE@10 degrades by ≈ 14×.
This is an **honest external generalisation gap** and should be reported as
such in the paper — it does **not** invalidate the in-domain reliability
claims, but it does bound the practical scope of CARA-FinSent to PhraseBank-
style financial prose.

---

## 3. Why fine-tuned checkpoints underperform zero-shot on FiQA

All three local FT checkpoints score worse than zero-shot on FiQA macro-F1.
The most informative case is `finbert_20260430_215910` (MCC = **−0.477**, mean
confidence = 0.96 on wrong predictions). A negative MCC at high confidence is
the signature of a model that **learned the inverted labels** from training.

This is consistent with the polarity bug: any FiQA-derived training run would
have absorbed the inversion. Three implications:

1. The single FT checkpoint trained on PhraseBank only (`finbert_20260501_193637`,
   id2label `{0:positive, 1:negative, 2:neutral}` matching the original
   ProsusAI head) gives the best of the FT checkpoints on FiQA (macro-F1 0.385),
   still well below zero-shot.
2. The 215910 checkpoint is contaminated and should not be used for any
   downstream analysis until re-trained on corrected data.
3. The 090136 checkpoint shows partial training (mean confidence 0.58, lowest
   ECE 0.08) — it appears to be an early-stopped run; it is honest but not
   representative of the leaderboard's vanilla FT row.

**No Phase 13 claim relies on any local FT checkpoint as a generalisation
anchor.** The single defensible external-validation number is the zero-shot
result (macro-F1 = 0.457).

---

## 4. What the FiQA result lets us claim, and not claim

### Claims it supports
- FinBERT (and by extension all CARA-FinSent FinBERT-based models) is **not
  domain-invariant**. There is a substantial in-domain → out-of-domain drop
  on FiQA tweets.
- Reliability metrics (ECE, mean confidence) **also degrade** out-of-domain,
  not just accuracy. This is the most important reliability finding.

### Claims it does NOT support
- It does **not** show that AW FinBERT generalises better or worse than
  zero-shot out-of-domain — AW was not evaluated.
- It does **not** show a SOTA on FiQA. Zero-shot 0.457 macro-F1 is a baseline
  number, not a competitive result vs FiQA-tuned models.
- It does **not** invalidate the PhraseBank-based AW vs ZS comparison.

---

## 5. Artefacts written

All under `results/2026-05-02/` (gitignored — must `git add -f` if publishing
to repo) and a parallel safe copy under `artifacts/phase13_extended_validation/`:

| Artefact | Path |
|---|---|
| Summary CSV (per-model metrics) | `phase13_external_validation_summary_20260502_140602.csv` |
| Manifest JSON | `phase13_external_validation_manifest_20260502_140602.json` |
| ZS predictions | `phase13_external_finbert_zero_shot_ProsusAI_finbert_predictions_20260502_140602.csv` |
| ZS classwise / confusion matrix | `*_classwise_*.csv`, `*_confusion_matrix_*.csv` |
| FT predictions × 3 | `phase13_external_finbert_finetuned_finbert_*_predictions_20260502_140602.csv` |
| FT classwise / CMs × 3 | as above |
| Confusion-matrix figures × 4 | `figures/2026-05-02/phase13_external_*_confusion_matrix_20260502_140602.png` |
| Safe summary copy | `artifacts/phase13_extended_validation/external_validation_summary_20260502_140602.csv` |
| Safe manifest copy | `artifacts/phase13_extended_validation/external_validation_manifest_20260502_140602.json` |

---

## 6. Recommendation for the paper

- **Include** the zero-shot FinBERT FiQA macro-F1 = 0.457 in the appendix or a
  "Limitations and external generalisation" section, exactly framed as an
  honest out-of-domain drop. It strengthens the reliability-first narrative
  by acknowledging scope limits.
- **Do not** report any of the FT checkpoint numbers from this run in the
  paper without first patching the FiQA polarity bug at source and re-training.
- **Do not** report AW external numbers until an AW checkpoint can be re-trained
  and run against the corrected FiQA test split.

The combined recommendation will be consolidated in
`docs/PHASE13_PAPER_UPDATE_RECOMMENDATION.md` (Step 10).
