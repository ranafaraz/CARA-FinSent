# CARA-FinSent: A Reliability-First Decision-Grade Framework for Financial Sentiment Analysis

**Authors:** Rana Faraz Ahmed et al.
**Target venue:** IEEE single-column draft.

> Draft skeleton. All citations are placeholders. Tables and figures are
> referenced by their files in [paper_assets/tables/](../paper_assets/tables/)
> and [paper_assets/figures/](../paper_assets/figures/).

## Abstract

Financial sentiment classifiers are typically reported on accuracy or macro-F1
alone, which does not reflect the way they are deployed. We present
**CARA-FinSent**, a reliability-first evaluation framework that combines
agreement-aware fine-tuning of FinBERT, calibration analysis, abstention,
seed-stability auditing, and strict leakage-controlled experimentation on the
Financial PhraseBank corpus. Across five seeds, agreement-weighted FinBERT
achieves the highest mean macro-F1 in our leaderboard (0.8863) and is
approximately 4× more seed-stable than vanilla fine-tuning, while zero-shot
FinBERT remains the best-calibrated model. The gain of agreement-weighted
FinBERT over zero-shot FinBERT is modest and not statistically decisive under
paired bootstrap and McNemar tests. Our contribution is therefore not a single
SOTA accuracy claim, but a decision-grade evaluation pipeline and a discussion
of what can and cannot be claimed for production deployment.

[CITATION: FinBERT]
[CITATION: Financial PhraseBank]
[CITATION: calibration in neural networks]

**Keywords:** financial sentiment analysis, FinBERT, agreement-aware learning,
calibration, abstention, decision-grade evaluation.

## 1. Introduction

[Insert opening on the importance of financial sentiment for decision-grade
applications, the limitations of single-metric reporting, and a preview of
the reliability-first framing.]

## 2. Related Work

- FinBERT and domain-adapted transformer baselines [CITATION: FinBERT].
- Annotator-agreement-aware learning and noisy labels [CITATION: agreement-aware
  learning survey].
- Calibration of neural classifiers [CITATION: Guo et al. on calibration].
- Selective prediction and abstention [CITATION: selective classification].
- Financial sentiment surveys [CITATION: financial sentiment survey].
- Recent LLM- and RAG-based financial assistants [CITATION: financial sentiment
  RAG/LLM study].

## 3. Research Gap and Motivation

Existing financial sentiment work tends to (a) report a single accuracy or F1
number, (b) train one seed, and (c) ignore calibration. None of these are
sufficient for trading or compliance use, where probabilities, confidence, and
robustness matter as much as point predictions.

## 4. Research Questions and Contributions

We address four research questions (see
[docs/PHASE11_METHOD_BLUEPRINT.md](PHASE11_METHOD_BLUEPRINT.md)):

- RQ1: Does agreement-aware fine-tuning improve financial sentiment
  classification?
- RQ2: Does it improve seed stability?
- RQ3: How do calibration and abstention change practical usefulness?
- RQ4: Are classical baselines still competitive in resource-constrained
  settings?

**Contributions.**

1. A leakage-controlled, multi-seed evaluation pipeline for financial sentiment.
2. An agreement-weighted FinBERT variant with per-instance loss weights derived
   from PhraseBank annotator agreement.
3. A unified calibration / abstention / seed-stability reporting layer.
4. A claim-boundary matrix that explicitly states what the evidence supports
   and what it does not.

## 5. Dataset and Experimental Design

[Insert Table 1 here] — `paper_assets/tables/table1_dataset_summary.csv`.

We use the Financial PhraseBank with ≥75% annotator agreement, projected to
{negative, neutral, positive}, with a controlled gold split of 3,353 / 479 /
959 instances [CITATION: Financial PhraseBank]. Strict text-hash leakage checks
run on every artifact.

## 6. Methodology

Methodology follows
[docs/PHASE11_METHOD_BLUEPRINT.md](PHASE11_METHOD_BLUEPRINT.md), summarised as:

- Classical baselines (TF-IDF + {linear SVM, logistic regression, SGD log-loss,
  XGBoost, random forest, multinomial NB, majority}).
- FinBERT zero-shot inference with `ProsusAI/finbert`.
- FinBERT supervised fine-tuning over five seeds.
- Agreement-weighted FinBERT with a `linear` weight schedule applied at the
  dataset level (`batched=True` map).
- Calibration via ECE@10 and Brier; abstention via a confidence threshold
  sweep; statistical validation via paired bootstrap, McNemar, and Cohen's d.

[Insert Figure 1 here] — `paper_assets/figures/fig1_pipeline_architecture.png`.

## 7. Experimental Setup

- Hardware: training executed on NVIDIA GPU (RunPod and Kaggle GPU); inference
  validated on local CPU.
- Software: Python 3.11, `transformers==4.49.0` (security pin), PyTorch CUDA
  build, scikit-learn, pandas, matplotlib.
- Seeds: {13, 21, 42, 87, 101}.
- Epochs: 3.
- Batch size: 16.
- Logging: every summary CSV records `git_commit_sha` and is timestamped.

## 8. Results

[Insert Table 2 here] — `paper_assets/tables/table2_model_leaderboard.csv`.
[Insert Figure 2 here] — `paper_assets/figures/fig2_macro_f1_leaderboard.png`.

[Insert Table 3 here] — `paper_assets/tables/table3_calibration_abstention.csv`.
[Insert Figure 3 here] — `paper_assets/figures/fig3_calibration_comparison.png`.

[Insert Table 4 here] — `paper_assets/tables/table4_statistical_validation.csv`.
[Insert Figure 4 here] — `paper_assets/figures/fig4_seed_stability.png`.
[Insert Figure 5 here] — `paper_assets/figures/fig5_abstention_tradeoff.png`.

[Insert Table 6 here] — `paper_assets/tables/table6_ablation_summary.csv`.

Verbatim narrative and per-section wording are taken from
[docs/PHASE11_RESULTS_NARRATIVE.md](PHASE11_RESULTS_NARRATIVE.md).

## 9. Discussion

We discuss why the macro-F1 advantage of agreement-weighted FinBERT over
zero-shot FinBERT is modest and not statistically decisive, why the seed
stability gain is the more defensible empirical contribution, and why
calibration favours the zero-shot prior. We argue this is consistent with the
broader observation that supervised fine-tuning shifts probability mass in a
way that hurts bin-wise calibration even when it improves Brier and macro-F1.

[Insert Figure 6 here] — `paper_assets/figures/fig6_claim_boundary_visual.png`.
[Insert Table 5 here] — `paper_assets/tables/table5_claim_boundary_matrix.csv`.

## 10. Practical Implementation

We outline a deployment blueprint (Hugging Face model card, Gradio demo,
FastAPI service, dashboard, and an RAG finance assistant) in
[docs/PHASE11_PRACTICAL_IMPLEMENTATION_BLUEPRINT.md](PHASE11_PRACTICAL_IMPLEMENTATION_BLUEPRINT.md).
This is the deployment plan implied by our reliability-first framing and is
not a claim of current production readiness.

## 11. Threats to Validity

- **External validity.** Single corpus (PhraseBank); no FiQA / news / SEC live
  evaluation yet.
- **Internal validity.** n=5 seeds limits the resolution of the paired
  comparisons.
- **Construct validity.** ECE@10 is a single calibration metric; alternative
  binning may shift conclusions.
- **Operational validity.** No latency benchmark; deployment-cost claims are
  qualitative.

## 12. Conclusion

CARA-FinSent demonstrates that decision-grade financial sentiment is not a
single-metric problem. Agreement-weighted FinBERT achieves the highest mean
macro-F1 and the highest seed stability among supervised variants, while
zero-shot FinBERT remains the best-calibrated. The headline contribution is the
reliability-first evaluation framework and the disciplined claim boundary, not
a universal SOTA claim.

## References Placeholder

- [CITATION: FinBERT]
- [CITATION: Financial PhraseBank]
- [CITATION: calibration in neural networks]
- [CITATION: selective classification]
- [CITATION: financial sentiment survey]
- [CITATION: agreement-aware learning survey]
- [CITATION: financial sentiment RAG/LLM study]
