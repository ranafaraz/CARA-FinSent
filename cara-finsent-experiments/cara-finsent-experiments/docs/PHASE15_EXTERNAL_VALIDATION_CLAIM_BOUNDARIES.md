# Phase 15 — External Validation Claim Boundaries

**Generated (UTC):** 2026-05-02
**Source:** [results/2026-05-02/phase14_external/phase14_external_summary_20260502_165943.csv](../results/2026-05-02/phase14_external/phase14_external_summary_20260502_165943.csv)
**Source corpus:** `data/processed/gold/latest_gold_fiqa_split_polarity_corrected.csv` (n=682 total; n=223 evaluated test rows after the polarity-corrected gold split, with `positive ↔ negative` swap applied to recover FiQA's reversed labels; `neutral` preserved).

## 1. The single external benchmark we have

| Family | n | Accuracy | macro-F1 | ECE₁₀ | Brier |
|---|---:|---:|---:|---:|---:|
| finbert_zero_shot (ProsusAI/finbert) | 223 | 0.4664 | **0.4567** | 0.3321 | 0.7987 |
| finbert_fine_tuned (PhraseBank gold) | 223 | 0.3677 | 0.3850 | 0.4533 | 0.9986 |
| finbert_agreement_weighted (CPU retrain) | 223 | 0.3229 | 0.3392 | 0.5320 | 1.1145 |

The polarity correction is mandatory: FiQA's published labels use the opposite sign convention to PhraseBank for `positive` and `negative`. Without it, all three FinBERT variants score below random.

## 2. What we are allowed to claim

✅ **Claim 1 (safe).** "Zero-shot ProsusAI/FinBERT achieves macro-F1 = 0.46 on FiQA after polarity correction (n = 223), with high calibration error (ECE = 0.33)."

✅ **Claim 2 (safe).** "Both PhraseBank-fine-tuned variants — vanilla FT (macro-F1 = 0.39) and agreement-weighted FT (macro-F1 = 0.34) — *under-perform* the zero-shot FinBERT baseline on FiQA. PhraseBank fine-tuning, including agreement-weighted fine-tuning, transfers poorly to the FiQA distribution."

✅ **Claim 3 (safe).** "Calibration error grows monotonically across {ZS → FT → AW} on FiQA (0.33 → 0.45 → 0.53), indicating that PhraseBank-tuned variants are not just less accurate but also less reliable out-of-domain."

✅ **Claim 4 (safe).** "Cross-domain generalisation is the dominant failure mode for CARA-FinSent: in-domain PhraseBank macro-F1 sits in the 0.86–0.91 range, while FiQA macro-F1 sits in the 0.34–0.46 range — a drop of ≈ 0.45 macro-F1."

## 3. What we are NOT allowed to claim

❌ **No "universally outperforms" claim.** With a single external corpus of n = 223, no single-model dominance claim can be made for general financial text.

❌ **No "robust to domain shift" claim.** The 0.45-macro-F1 drop is precisely the opposite of robustness.

❌ **No "agreement-weighted training improves out-of-domain" claim.** AW is in fact the *worst* performer on FiQA. Inside the paper, the AW story is reliability-first (calibration / agreement modelling) and **not** cross-domain transfer.

❌ **No "calibration repairs out-of-domain accuracy" claim.** Calibration was fitted on the PhraseBank val split; it does not move ECE on FiQA in any direction we measured here. Out-of-domain calibration would require a FiQA-internal val split, which we did not carve out.

❌ **No comparison vs prior published numbers on FiQA.** The polarity-corrected subset (n = 223) is not the same evaluation slice as any published FiQA leaderboard; cross-paper comparisons would be apples-to-oranges.

## 4. The single external claim that goes in the paper

> "On the polarity-corrected FiQA test slice (n = 223), zero-shot FinBERT achieves macro-F1 = 0.46 and PhraseBank fine-tuning — including agreement-weighted fine-tuning — does not improve out-of-domain performance (FT 0.39, AW 0.34). Calibration error grows from ECE = 0.33 (ZS) to 0.53 (AW), indicating that fine-tuned variants are simultaneously less accurate and less reliable on FiQA. We therefore frame CARA-FinSent as an *in-domain* reliability framework, with cross-domain generalisation an open problem."

This is the only external-validation paragraph that is defensible from the evidence we have.

## 5. Open work (out of scope for Phase 15)

- A second external corpus (e.g., StockTwits gold-relabelled, SEC 10-K paragraphs with manual labels) would be required to make any *general* claim about CARA-FinSent's external behaviour.
- A FiQA-internal val/test split would be required to ask whether out-of-domain calibration is fixable with a small held-out FiQA tuning set.

Both are deferred to future work and explicitly listed in the limitations section of the paper.
