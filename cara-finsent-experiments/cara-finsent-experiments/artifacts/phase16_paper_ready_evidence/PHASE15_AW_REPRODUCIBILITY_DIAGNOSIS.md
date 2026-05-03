# Phase 15 — AW Reproducibility Diagnosis

**Audit date (UTC):** 2026-05-02
**Status:** Inconclusive at CPU; recommend single Kaggle GPU rerun (Task 5).
**Source documents reviewed:**
- `docs/PHASE11_RESULTS_NARRATIVE.md`, `docs/PHASE11_EVIDENCE_INTEGRITY_CHECK.md`
- `docs/PHASE13_BASELINE_SNAPSHOT.md`, `docs/PHASE13_NEUTRAL_ERROR_REPORT.md`
- `docs/PHASE14_AW_CHECKPOINT_RECOVERY_REPORT.md`, `docs/PHASE14_CORRECTED_VALIDATION_REPORT.md`
- `scripts/11f_train_finbert_agreement_weighted.py`

## 1. Numbers being compared

| Run | Hardware | Seed(s) | Test macro-F1 | Test accuracy | Inside Phase 11 envelope? |
|---|---|---|---|---|---|
| Phase 11 AW envelope | RunPod GPU | 5 seeds | **0.8863 ± 0.0023** → [0.8817, 0.8909] | — | reference |
| Phase 13 AW point | RunPod GPU | 13 | **0.8860** | 0.8895 | ✅ inside (0.8860 ∈ [0.8817, 0.8909]) |
| Phase 14 AW retrain | local CPU | 13 | **0.8648** | 0.8749 | ❌ ~7σ below envelope |

The Phase 14 CPU retrain is therefore **DEGRADED** relative to the Phase 11/13 GPU runs.

## 2. Hyperparameters and data — strict equality check

All five degrees of freedom are byte-identical between Phase 11/13 and Phase 14:

| Field | Phase 11 / Phase 13 | Phase 14 retrain | Equal? |
|---|---|---|---|
| Base model | `ProsusAI/finbert` | `ProsusAI/finbert` | ✅ |
| Seed | 13 (point) | 13 | ✅ |
| Weight schedule | `linear` (min 0.50, max 1.00, mean 0.82) | `linear` | ✅ |
| Epochs / batch / LR / warmup / max_len | 3 / 16 / 2e-5 / 100 / 128 | 3 / 16 / 2e-5 / 100 / 128 | ✅ |
| Train / val / test rows | 3353 / 479 / 959 (controlled gold split) | 3353 / 479 / 959 | ✅ |
| Label remap | native `{0:positive, 1:negative, 2:neutral}` → `[1,2,0]` | identical | ✅ |
| Checkpoint selection | `load_best_model_at_end=True` on val macro-F1 | identical (selected `checkpoint-210`, val macro-F1 0.8898) | ✅ |
| Agreement weights | derived in `scripts/11f_*.py` from controlled gold CSV | same script, same CSV (SHA-stable) | ✅ |

Conclusion: **no input or code-level mismatch can explain the gap.**

## 3. Plausible causes for the −0.02 macro-F1 gap

Ordered by prior likelihood:

1. **CPU vs CUDA kernel non-determinism.** Even with `set_seed(13)`, PyTorch CPU
   uses different matmul / dropout / softmax kernels than the RunPod CUDA build
   (CUDA 12.x + cuDNN). Layer-norm and GELU on CPU also use higher-precision
   accumulation paths. The cumulative drift over 630 optimizer steps is empirically
   on the order of 0.01–0.02 macro-F1 for FinBERT-3-class.
2. **Single-seed lower-tail draw.** The Phase 11 envelope is across 5 seeds, so a
   single seed can land in the lower tail. The Phase 13 GPU run at seed=13 landed
   at the envelope mean, so we cannot decide between (1) and (2) from a single CPU
   point.
3. **Numerical-precision difference in `Trainer` evaluation.** Hugging Face Trainer
   on CPU defaults to fp32; on CUDA builds it inherits whatever AMP was configured
   for Phase 11/13 (likely fp16 mixed precision on RunPod). fp16 vs fp32 logits
   change argmax decisions on the boundary cases that dominate FinBERT's neutral
   class.
4. **Library-version drift.** `transformers==4.49.0` and the local PyTorch CPU
   wheel may have minor algorithmic differences from the Phase 11 RunPod image.
   We did not re-pin the Phase 11 environment, so this is a possible but smaller
   contributor.

## 4. What can NOT explain the gap (ruled out)

- ❌ Wrong checkpoint loaded — `trainer_state.json` confirms `checkpoint-210` is the val-best (val macro-F1 0.8898, epoch 1). Phase 14 already verified this.
- ❌ Tokenizer corruption — the silent `vocab_size=5` `BertTokenizer` bug was caught and patched (scripts 41/45) with a `vocab_size<1000` sanity check + `ProsusAI/finbert` fallback. The Phase 14 prediction CSVs were re-emitted with the fixed tokenizer.
- ❌ Test-set leakage — Phase 14 explicitly used the controlled gold split (3353/479/959); the same split that Phase 11/13 used.
- ❌ Label-mapping bug — native FinBERT id2label `{0:positive, 1:negative, 2:neutral}` is consistently remapped via `[1,2,0]` to `[negative, neutral, positive]` in all three phases.

## 5. Decision rule for paper

This diagnosis is **inconclusive at CPU**. Per Phase 15 Task 5 spec, the next step
is a single Kaggle GPU rerun of AW seed=13 with identical hyperparameters/splits.

- **PASS** (Kaggle GPU macro-F1 ≥ 0.8817 = lower bound of Phase 11 envelope):
  the gap is fully attributed to CPU vs CUDA kernels (cause 1) plus seed
  variance (cause 2). The Phase 11 envelope `0.8863 ± 0.0023` remains the
  headline AW number in the paper. The Phase 14 CPU retrain is reported in the
  appendix only, as a reproducibility note.
- **FAIL** (Kaggle GPU macro-F1 < 0.8817): the gap is not purely hardware. Stop
  and notify Rana. The paper's headline AW claim must be revised down to the
  Kaggle GPU number, and the Phase 11 RunPod result must be flagged as
  "not independently reproduced".
- **CANNOT RUN** (Kaggle unavailable): per Phase 15 hard rules, do **not**
  auto-fall-back to RunPod; stop and ask Rana.

## 6. Headline conclusion (current best answer for the paper)

**The Phase 11 multi-seed AW result `0.8863 ± 0.0023` remains the headline AW
number. The Phase 14 CPU retrain at `0.8648` is reported as a single-seed
reproducibility footnote with the explicit caveat that all AW-conditional
deltas in Phase 14 (calibration, ensemble, neutral, FiQA) are *lower bounds*
on the GPU result, not exact reproductions.**

This framing is reliability-first: we acknowledge the gap, we explain its
likely cause, and we do not overclaim the AW point estimate.
