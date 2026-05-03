# Phase 16 — Final Paper Claim Boundaries

This document defines what may be claimed where in the paper, given the Phase 16 evidence that the Phase 11 AW envelope **is not reproducible** from the controlled gold split (`data/processed/gold/latest_gold_phrasebank_split.csv`, SHA-256 `f0f4404f…`) on either CPU (Phase 14) or Kaggle Tesla P100 GPU (Phase 16, seed=13: test macro_F1 = 0.8695 < gate 0.8817).

## A. Main-paper safe (claim freely)

These claims are reproducible from the controlled split with the in-repo scripts at HEAD `6a999be`.

1. **Fine-tuned FinBERT** on PhraseBank gold attains test **acc 0.8895 / macro_F1 0.8884** (uncalibrated) and **acc 0.8936 / macro_F1 0.8958** with **isotonic calibration on val** ([`phase14_calibration/clean_calibration_summary_ft_*.csv`](../results/phase14_calibration/)).
2. **Zero-shot FinBERT** attains test **acc 0.8832 / macro_F1 0.8836**, calibrating to **acc 0.9030 / macro_F1 0.9038** with isotonic on val.
3. **Best convex ensemble** (weights tuned on val only): `aw=0.40, zs=0.60, ft=0.00` → test **acc 0.8916 / macro_F1 0.8953**.
4. **Calibration improves ECE** for both ZS and FT FinBERT under val-fit Platt and isotonic; isotonic is the val-best calibrator for ZS and FT.
5. **Classical baselines** (TF-IDF + Linear SVM 0.7497 / 0.6941; majority 0.5985 / 0.2496) establish a clear weak-baseline floor.
6. **Reproducibility methodology**: controlled `train/val/test = 3353/479/959` split with frozen SHA-256, deterministic label remap `[1,2,0] → [negative, neutral, positive]` from FinBERT native id2label `{0:positive, 1:negative, 2:neutral}`, fixed seeds, written manifests per run.

## B. Appendix-only (cite with explicit caveats)

These results are **historical** and **not reproducible from the controlled split**; they may be discussed in an appendix or limitations section but **must not** be the paper headline.

1. **Phase 11 AW 5-seed envelope** macro_F1 **0.8863 ± 0.0023** (95% CI [0.8817, 0.8909]). Caveat: when the same recipe (seed=13, weight_schedule=linear, 3 epochs, batch=16, LR=2e-5, warmup=100, max_len=128, base `ProsusAI/finbert`, `load_best_model_at_end` on val macro_F1) is rerun against the controlled gold split, two independent retrains land 0.014–0.022 macro_F1 below the envelope (CPU 0.8648; Kaggle GPU 0.8695). This indicates Phase 11's training data, hyperparameters, or evaluation slice differed in some unrecorded way from the locked Phase 13 controlled split; the envelope cannot be defended as state-of-the-art on the released split.
2. **Any AW gain over zero-shot or fine-tuned FinBERT** as observed in Phase 11. On the controlled split, AW is ~0.02–0.03 macro_F1 *worse* than fine-tuned FinBERT and ~0.01 macro_F1 *worse* than zero-shot, ruling out an AW-as-headline claim.
3. **External validation on FiQA** results from Phase 15 — keep their existing claim-boundary scope (already in [PHASE15_EXTERNAL_VALIDATION_CLAIM_BOUNDARIES.md](../artifacts/phase15_paper_ready_evidence/PHASE15_EXTERNAL_VALIDATION_CLAIM_BOUNDARIES.md)).

## C. Unsafe — do not claim

1. "Agreement-weighted FinBERT achieves macro_F1 ≈ 0.886 on PhraseBank." Replace with an explicit reproducibility statement: "Agreement-weighted training did not improve over fine-tuned FinBERT on the controlled split; an earlier 5-seed envelope of 0.8863 ± 0.0023 reported in Phase 11 could not be reproduced under the locked recipe on either CPU or a Kaggle Tesla P100 GPU."
2. Any state-of-the-art comparison framed around the Phase 11 AW number.
3. Calibrated AW (`finbert_agreement_weighted_platt_cpu_retrain`, macro_F1 0.8813) as a headline result — it is fitted on val, evaluated on test with a CPU retrain that already underperforms the Phase 11 envelope; report as appendix.

## Why downgrading AW is the right call

- Two independent retrains (Phase 14 CPU, Phase 16 Kaggle GPU at seed=13) on the released split land below the Phase 11 envelope by ~0.017 macro_F1.
- Both retrains used the same recipe and the same data file; the only known difference between them and Phase 11 is the original (uncontrolled) split material.
- The Phase 11 envelope therefore likely reflected a different effective sample (e.g., pre-controlled split, pre-dedup, or a different label distribution). Releasing the AW number as a paper headline would be unverifiable by readers who use the released split.
- Fine-tuned FinBERT is reproducible, calibratable, and beats AW on the controlled split — making it a stronger and more defensible headline.
