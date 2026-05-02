# Phase 11 — Paper-Ready Results Narrative

Date: 2026-05-02
Scope: Academic, IEEE-tone narrative for the CARA-FinSent results section.
All numbers below are sourced from the verified evidence in
[results/2026-05-02/](../results/2026-05-02/) and aggregated by the Phase 11
scripts under [paper_assets/tables/](../paper_assets/tables/).

## 1. Dataset and controlled split

We evaluate on the Financial PhraseBank corpus, restricted to instances with
≥75% annotator agreement and projected to a fixed three-class label set
{negative, neutral, positive}. Sentences are partitioned with a leakage-controlled
gold split into 3,353 train / 479 validation / 959 test instances. Strict
text-hash leakage checks run on every artifact and report zero overlap across
splits.

## 2. Baselines

We compare seven classical baselines (TF-IDF features paired with linear SVM,
logistic regression, SGD log-loss, XGBoost, random forest, multinomial naïve Bayes,
and a majority-class baseline). The strongest classical model (TF-IDF + linear SVM)
reaches mean macro-F1 0.6941, which is roughly 19 macro-F1 points below any
FinBERT-class model. Classical models remain materially cheaper to train and
deploy and are useful as resource-constrained reference points.

## 3. FinBERT zero-shot

Zero-shot inference with `ProsusAI/finbert` yields mean macro-F1 0.8836 across
five seeds. Because the underlying model parameters are seed-invariant, the
across-seed standard deviation is effectively zero. Notably, zero-shot FinBERT is
also the best-calibrated model in the entire leaderboard, with mean ECE@10 of
0.0236.

## 4. Fine-tuned FinBERT

Vanilla supervised fine-tuning of FinBERT for three epochs across five seeds
gives mean macro-F1 0.8821 with std 0.00923. Mean ECE@10 degrades to 0.0608, and
mean Brier score is 0.1810. Despite extra training signal, vanilla fine-tuning
does not consistently beat the zero-shot variant on macro-F1, but does shift
class-wise behaviour as observed in the per-class confusion matrices.

## 5. Agreement-weighted FinBERT

Agreement-weighted (AW) FinBERT augments the supervised loss with an annotator-
agreement-derived per-instance weight, downweighting low-agreement examples and
upweighting high-agreement ones during training. Across the same five seeds,
AW achieves the **highest mean macro-F1 in our leaderboard at 0.8863 (std 0.00227)**.
The macro-F1 lift over zero-shot FinBERT (Δ = +0.0027) is small and within one
standard deviation of the AW seed distribution. A paired bootstrap on prediction-
level outputs (n=2000 resamples on n=959 test items) yields a 95% CI of
[-0.018, +0.023] for AW − zero-shot and a McNemar p-value of 0.56, so the
in-domain accuracy advantage of AW over zero-shot is **not statistically
decisive**. The lift over vanilla fine-tuned FinBERT (Δ = +0.0042, paired
bootstrap p≈0.06, McNemar p≈0.06) is borderline.

## 6. Calibration findings

Zero-shot FinBERT is the best-calibrated model with ECE@10 = 0.0236.
Agreement-weighted FinBERT is roughly 2× less calibrated (ECE@10 = 0.0495), and
vanilla fine-tuning further degrades calibration (ECE@10 = 0.0608). In Brier
terms, however, AW improves over both zero-shot (0.1762) and fine-tuned (0.1810)
FinBERT, reaching 0.1713. This suggests AW shifts probability mass toward sharper
correct predictions but does not improve bin-wise calibration relative to the
zero-shot prior.

## 7. Abstention findings

Abstention curves on the FinBERT family show that all three FinBERT variants
gain accuracy when low-confidence predictions are deferred, but the benefit
saturates quickly. On zero-shot FinBERT, raising the confidence floor from 0.50
to 0.70 improves accuracy from 0.887 to 0.902 while reducing coverage from 1.00
to 0.96, indicating a useful 3–4 percentage-point gain at modest abstention.

## 8. Seed stability findings

Agreement weighting is the most seed-stable trained variant, with
across-seed std of macro-F1 = 0.00227 versus 0.00923 for vanilla fine-tuning
(~4× lower variance). This is one of the stronger and more defensible empirical
contributions of the system.

## 9. Error analysis findings

Across FinBERT variants, the dominant error pattern is "neutral predicted as
positive". Mean confidence on incorrect predictions remains high (≈0.87), and a
non-trivial number of overconfident-wrong predictions persist. This motivates
the abstention layer and the reliability-first framing of the system.

## 10. Practical implications

For decision-grade financial sentiment, a reasonable deployment strategy is:
- Use zero-shot FinBERT when calibration is the primary requirement (e.g. raw
  probabilities feeding a downstream Bayesian aggregator).
- Use agreement-weighted FinBERT when seed-stable point predictions are required
  and calibration is repaired downstream (e.g. temperature scaling).
- Pair either model with an abstention threshold and a TF-IDF fallback for
  cost-constrained inference paths.

## 11. Limitations

(a) Evaluation is in-domain on Financial PhraseBank; out-of-domain robustness on
FiQA-headlines, news streams, or live filings is not yet quantified. (b) The
sample size is n=959 test items per seed, limiting the resolution of pairwise
significance tests. (c) Calibration is reported via ECE@10 and Brier on a single
test split and may understate sensitivity to bin choice. (d) No latency
benchmark is reported; cost claims are qualitative.

## 12. What should be claimed carefully

- Agreement-weighted FinBERT achieved the highest mean macro-F1 among evaluated
  models, but the gain over zero-shot FinBERT was modest. The more defensible
  contribution is therefore not a large accuracy improvement claim, but a
  **reliability-focused evaluation framework** that combines agreement-aware
  training, calibration, abstention, seed-stability analysis, and strict
  leakage-controlled experimentation.
- We do **not** claim universal SOTA for financial sentiment analysis.
- We do **not** claim live-trading or investment-decision validation.
