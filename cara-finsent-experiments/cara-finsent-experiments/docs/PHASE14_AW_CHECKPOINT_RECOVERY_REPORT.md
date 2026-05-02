# Phase 14 — AW Checkpoint Recovery Report

**Status:** complete (CPU retrain finished 2026-05-02; metrics flagged)
**Selected recovery path:** Option 1 from `vscode_askQuestions` — *Retrain AW on CPU (1 seed=13, ~60–90 min, no cloud)*.
**Date initiated (UTC):** see `logs/phase14_aw_train_seed13.log`.
**Git commit at retrain start:** `36dae7f`.
**New checkpoint:** `models/finbert_agreement_weighted_linear_20260502_145813`
**Train wall-clock (CPU):** 4463 s (~74 min).

## 1. Problem statement

The Phase 13 manifest references the AW checkpoint
`models/finbert_agreement_weighted_linear_20260502_082228/` as the source
of the canonical AW predictions in
`results/2026-05-02/finbert_agreement_weighted_predictions_20260502_082228.csv`
(seed=13, macro-F1 = 0.8860, accuracy = 0.8895).

Phase 14 environment audit on `36dae7f` showed that this checkpoint
directory does **not** exist locally. It was trained on a Phase 8 RunPod
GPU instance and the model weights were never synced back into the
workspace before that pod was destroyed.

This blocks Phase 14 Steps 4 (AW calibration), 5 (ensembles that
include AW), and 7 (AW external validation on the polarity-corrected
FiQA file), all of which require AW probability outputs.

## 2. Options considered

The user was asked, via `vscode_askQuestions`, to choose among:

| Option | Description | Cost | Cloud? |
|---|---|---|---|
| 1 (chosen) | CPU retrain seed=13, linear schedule, 3 epochs | ~60–90 min estimate; actual ~2 h on this CPU | No |
| 2 | Defer AW work to a future GPU session, finish Phase 14 with ZS+FT only | 0 min | No |
| 3 | Re-use the Phase 13 AW prediction CSV directly without re-training | 0 min | No (but cannot derive new probabilities for Phase 14 calibration / ensemble) |
| 4 | Run on Kaggle GPU | ~10 min training time + setup | Yes |

The user selected **Option 1**. RunPod was explicitly excluded by the
Phase 14 hard rule: *"Do not start RunPod automatically. Use Kaggle
first if GPU is required. If Kaggle fails, ask Rana before using
RunPod."*

## 3. Retrain configuration

Exact command (from `logs/phase14_aw_train_seed13.log`):

```
.venv\Scripts\python.exe scripts/11f_train_finbert_agreement_weighted.py \
    --seed 13 --weight_schedule linear --num_epochs 3 --batch_size 16
```

Hyperparameters (identical to Phase 11 / Phase 13 baseline):

- base model: `ProsusAI/finbert`
- learning rate: 2e-5
- warmup steps: 100
- max length: 128
- agreement weight schedule: `linear` (min 0.50, max 1.00, mean 0.82)
- training rows: 3353; val rows: 479; test rows: 959 (controlled gold split)
- native `id2label = {0: positive, 1: negative, 2: neutral}`
- canonical remap to `[negative, neutral, positive]` = `[1, 2, 0]`

CPU-only confirmed; per-step time was approximately 12–15 s/iter.
Total expected wall-clock at 630 iterations is therefore ~2 h 10 min,
slower than the 60–90 min initial estimate. The user accepted this as
the cost of staying off cloud GPUs.

## 4. New checkpoint location

Written by `scripts/11f_train_finbert_agreement_weighted.py` to
`models/finbert_agreement_weighted_linear_20260502_145813/`.
Not committed to git (forbidden by Phase 14 allow-list).

## 5. Reproducibility check

| Metric | Phase 13 baseline (RunPod GPU) | Phase 14 retrain (local CPU) | Δ |
|---|---|---|---|
| seed | 13 | 13 | – |
| macro-F1 (test) | 0.8860 | **0.8648** | **-0.0212** |
| accuracy (test) | 0.8895 | **0.8749** | **-0.0146** |
| eval val accuracy | – | 0.8977 | – |
| eval val macro-F1 | – | 0.8803 | – |

**Reproducibility verdict: INCONCLUSIVE / DEGRADED.** The Phase 14 CPU
retrain at seed=13 lands at macro-F1 = 0.8648, which is **below** the
Phase 11 5-seed envelope `0.8863 ± 0.0023` ≈ `[0.8817, 0.8909]` by
about 0.017 (≈ 7σ of the Phase 11 std).

Likely contributors:

1. **Non-deterministic CPU vs CUDA kernels.** Even at seed=13, PyTorch
   matmul/dropout fusion paths differ between CPU and the original
   RunPod CUDA build, and this is enough to put the run outside the
   tight ±0.0023 std band reported in Phase 11.
2. **Single-seed instability.** The Phase 11 envelope is across 5
   seeds; a single seed can land in the lower tail. Without re-running
   on a comparable GPU we cannot disentangle (1) and (2).

**Implication for Phase 14 downstream.** All AW-conditional results in
§4.3, §5, §6 and §7 of `docs/PHASE14_CORRECTED_VALIDATION_REPORT.md`
are reported with the explicit caveat *"based on the Phase 14 CPU
retrain checkpoint, which is below the Phase 11 GPU envelope by ~0.02
macro-F1; treat AW-vs-baseline deltas as a lower bound on the GPU
result, not as a tight reproduction."*

For paper integrity: **the Phase 11 multi-seed AW number remains the
headline AW result**; the Phase 14 retrain is used only for clean
val/test calibration and ensemble experiments where the *relative*
behaviour (calibration improvement, ensemble winner) is what matters,
not the absolute AW point estimate.

## 6. Why this matters for the paper

The Phase 13 AW result is being used as the headline reliability claim
in the Research Paper 3 draft. Reproducing it locally — even on CPU —
gives the paper a concretely re-runnable artefact, which is required by
the "reproducibility appendix" reviewer expectations. It also unblocks
all Phase 14 paper-safe corrections (clean calibration, clean
ensembles, clean neutral mitigation, FiQA polarity-corrected external
validation).

## 7. Forbidden actions

In line with Phase 14 hard rules, the new checkpoint will **not** be
committed to git. Only its path and metric tables are committed. The
allowed-paths whitelist is enforced at commit time via the rules in
`docs/PHASE14_AGENT_INSTRUCTIONS.md` Step 11.
