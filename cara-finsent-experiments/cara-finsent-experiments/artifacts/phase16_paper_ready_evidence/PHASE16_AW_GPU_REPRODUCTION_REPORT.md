# Phase 16 — AW seed=13 GPU Reproduction Report

**Decision: `FAIL_NOT_REPRODUCED_ON_KAGGLE_GPU`**

## Summary

Phase 11's headline AW envelope `macro_F1 = 0.8863 ± 0.0023` (test, 5 seeds → 95% CI ≈ [0.8817, 0.8909]) was **not reproduced** on Kaggle GPU under the locked AW recipe at the canonical Phase 16 seed (seed = 13).

| split | accuracy | macro_F1 | gate (≥ 0.8817) |
| ---   | ---      | ---      | ---             |
| val   | 0.9081   | 0.8978   | n/a             |
| test  | 0.8780   | **0.8695**   | **FAIL** (−0.0122 below lower bound, −0.0168 below mean) |

The Kaggle GPU result (0.8695) is essentially identical to the Phase 14 CPU retrain on the same controlled split (0.8648), and ~0.017 below the Phase 11 mean (0.8863). This rules out a CPU-vs-GPU non-determinism explanation.

## Reproduction setup

- **Kaggle kernel**: `ranafarazahmed/cara-finsent-phase-16-aw-seed-13-script` (v5)
- **Kaggle dataset**: `ranafarazahmed/cara-finsent-phase16-aw-bundle`
- **Account**: `ranafarazahmed`
- **Repo HEAD**: `6a999be6cbba48d5dbd3e08021fddcb78ed4dfe7`
- **GPU**: Tesla P100-PCIE-16GB (sm_60)
- **Torch**: 2.4.1+cu121 (Kaggle's preinstalled torch 2.10.0+cu128 dropped sm_60; kernel pinned a compatible version in a clean subprocess)
- **CUDA available**: True
- **Python**: 3.12.12
- **Train wall clock**: 139.7s (3 epochs)

### Locked AW hyperparameters
| param | value |
| ---   | ---   |
| seed | 13 |
| weight_schedule | linear |
| num_epochs | 3 |
| batch_size | 16 |
| learning_rate | 2e-5 (script default) |
| warmup_steps | 100 (script default) |
| max_seq_length | 128 (script default) |
| base_model | `ProsusAI/finbert` |
| load_best_model_at_end | True |
| metric_for_best_model | macro_f1 (val) |

### Controlled split
- **Path**: `data/processed/gold/latest_gold_phrasebank_split.csv`
- **SHA-256**: `f0f4404fe330e693f2e16af639e2e864cb814514396adbcc3804f36bc11bc144`
- **Counts**: train=3353, val=479, test=959
- **Native FinBERT id2label**: `{0: positive, 1: negative, 2: neutral}`
- **Canonical remap (idx→label)**: `[1, 2, 0]` → `[negative, neutral, positive]`

### Selected checkpoint
- **Rule**: `trainer_state.json::best_model_checkpoint` (val-best by macro_f1)
- **Best**: `models/finbert_agreement_weighted_linear_20260502_215830/checkpoint-210` (epoch 1)
- **Best val macro_F1 reported by trainer**: 0.8978 (epoch 1); 0.8899 at epoch 3

## Cross-check vs. earlier evidence

| run | env | seed | test macro_F1 |
| --- | --- | ---  | ---           |
| Phase 11 AW (5-seed mean) | mixed | various | **0.8863 ± 0.0023** |
| Phase 14 CPU retrain | local CPU | 13 | 0.8648 |
| **Phase 16 Kaggle GPU (this run)** | **Kaggle P100** | **13** | **0.8695** |

Both fresh retrains land ~1.5–2.2 macro-F1 points below the Phase 11 envelope. The Phase 11 number is therefore **not safely reproducible from the controlled gold split alone with the locked recipe**.

## Stop condition triggered

`AW seed 13 fails gate (≥ 0.8817)` is a hard stop in the Phase 16 plan. Per spec, the agent halts here and surfaces the result; no RunPod fallback is invoked autonomously. The user has indicated they will run RunPod manually if/when needed.

## Artefacts (all under `results/2026-05-02/phase16_aw_gpu_repro/`)

| file | size | sha256 |
| ---  | ---  | ---    |
| `phase16_aw_seed13_environment.txt` | 18,660 | (Kaggle pip-freeze) |
| `phase16_aw_seed13_gpu_train_log.txt` | 49,836 | full stdout/stderr from training |
| `phase16_aw_seed13_val_predictions.csv` | 134,527 | `d73ed358bfd9acc7254aceca0fe7d0f3db98374dfb5089d2c220007ba875dc4d` |
| `phase16_aw_seed13_test_predictions.csv` | 270,204 | `c1e3d058b482ff58375c3e3ac5e625a59724b5b8ecaaa9bdb0cf4b93a9b20b35` |
| `phase16_aw_seed13_metrics.csv` | 156 | (test+val rows with `pass_gate_0p8817`) |
| `phase16_aw_seed13_manifest.json` | — | full provenance manifest |

> The Kaggle web UI reports run status `ERROR`, but this is post-hoc: training and val/test prediction generation completed normally and the metrics + predictions were already written before a non-zero exit in the kernel's tail post-processing block. The artefacts above are complete and self-consistent.
