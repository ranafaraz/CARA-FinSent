# Phase 2 Model Evaluation - Session 3 Summary

**Date**: 2026-05-01  
**Status**: In Progress (Fine-tuning running, 15/630 steps completed)  
**Git Commits**: 
- `afe33b1` (HEAD) - Phase 2 Step A-F: Model Evaluation Pipeline
- Prior: `5a28568` (Phase 1 Final)

---

## Phase 2 Objectives

Transition from data correction (Phase 1) to controlled model evaluation:
- Establish clean baseline comparisons on Phase 1 canonical splits
- Verify FinBERT label mapping before large-scale use
- Generate multi-model benchmark on PhraseBank-only data
- Test external generalization on FiQA independent test set
- Explore agreement-aware training effectiveness

---

## Completed (Phase 2 Steps A-C)

### ✓ Step A: FinBERT Label Mapping Sanity Check

**File**: `scripts/11a_verify_finbert_label_mapping.py` (NEW)

**Execution**:
```bash
python scripts/11a_verify_finbert_label_mapping.py --seed 42
```

**Results**:
- **Status**: PASS ✓
- 15 hand-written financial examples (5 positive, 5 negative, 5 neutral)
- Native order accuracy: 100%
- Canonical order accuracy: 100%
- Remap confirmed: `[1, 2, 0]` (prosus [pos, neg, neu] → canonical [neg, neu, pos])

**Outputs**:
- `results/2026-05-01/finbert_label_mapping_sanity_20260501_163236.csv`
- `results/2026-05-01/finbert_label_mapping_manifest_20260501_163236.json`

---

### ✓ Step B: Classical Baselines on PhraseBank

**File**: `scripts/10_run_classical_baselines.py` (MODIFIED)

**Modification**: Added pre-split detection to use Phase 1 controlled splits instead of re-splitting

**Execution**:
```bash
python scripts/10_run_classical_baselines.py \
  --data data/processed/gold/latest_gold_phrasebank_split.csv \
  --seed 42
```

**Results**:
| Model | Accuracy | Macro-F1 | Weighted-F1 |
|-------|----------|----------|-------------|
| **tfidf_linear_svm** | **0.7497** | **0.6941** | **0.7429** |
| tfidf_sgd_log_loss | 0.7466 | 0.6755 | 0.7348 |
| tfidf_logistic_regression | 0.7404 | 0.6838 | 0.7364 |
| tfidf_xgboost | 0.7205 | 0.6216 | 0.6942 |
| tfidf_random_forest | 0.7185 | 0.5972 | 0.6868 |
| tfidf_multinomial_nb | 0.6517 | 0.3938 | 0.5718 |
| majority_baseline | 0.5985 | 0.2496 | 0.4482 |

**Dataset**: PhraseBank split (train 3,353 / test 959)

**Outputs**:
- `results/2026-05-01/classical_baseline_summary_20260501_163355.csv`
- `results/2026-05-01/classical_baselines_manifest_20260501_163355.json`

---

### ✓ Step C: FinBERT Zero-Shot Evaluation

**File**: `scripts/11c_eval_finbert_zero_shot.py` (MODIFIED)

**Modification**: Added pre-split detection; uses Phase 1 controlled PhraseBank test set

**Execution**:
```bash
python scripts/11c_eval_finbert_zero_shot.py \
  --data data/processed/gold/latest_gold_phrasebank_split.csv \
  --seed 42
```

**Results**:
| Metric | Value |
|--------|-------|
| **Accuracy** | **0.8832** |
| **Macro-F1** | **0.8836** |
| Weighted-F1 | 0.8844 |
| MCC | 0.8003 |
| Latency (total) | 97.6 seconds |
| Latency (per example) | 0.102 seconds |
| Coverage (non-abstained) | 95.4% |
| Accuracy (non-abstained) | 0.8984 |

**Dataset**: PhraseBank test set (959 examples)

**Outputs**:
- `results/2026-05-01/finbert_baseline_summary_20260501_163500.csv`
- `results/2026-05-01/finbert_baseline_predictions_20260501_163500.csv`
- `results/2026-05-01/finbert_baseline_manifest_20260501_163500.json`

---

## FinBERT Zero-Shot vs Classical Comparison

**Baseline Gap Analysis**:
```
Best Classical (tfidf_linear_svm):
  - Accuracy: 0.7497
  - Macro-F1: 0.6941

FinBERT Zero-Shot:
  - Accuracy: 0.8832  (+13.35 points, +17.8%)
  - Macro-F1: 0.8836  (+0.1895 points, +27.3%)

→ FinBERT zero-shot substantially outperforms best classical approach
→ Validates investment in transformer-based methods
```

---

## In Progress (Phase 2 Step D)

### ⏳ Step D: FinBERT Fine-Tuning on PhraseBank

**File**: `scripts/11d_train_finbert_phrasebank.py` (NEW)

**Status**: Running (15/630 steps after 2:36, estimated ~3.8 hours total)

**Execution**:
```bash
python scripts/11d_train_finbert_phrasebank.py \
  --data data/processed/gold/latest_gold_phrasebank_split.csv \
  --num_epochs 3 \
  --batch_size 16 \
  --seed 42
```

**Configuration**:
- Model: `ProsusAI/finbert` with classifier head reinitialization
- Dataset: PhraseBank train (3,353) / test (959)
- Epochs: 3
- Batch size: 16
- Learning rate: 2e-5
- Warmup steps: 100
- Max length: 128

**Expected Output**:
- `results/2026-05-01/finbert_finetuned_summary_*.csv`
- `results/2026-05-01/finbert_finetuned_predictions_*.csv`
- `models/finbert_finetuned_*/` checkpoint directory
- Manifest JSON with training metadata

**Hypothesis**:
Fine-tuning should further improve over zero-shot, though risk of overfit on PhraseBank exists. FiQA external validation will reveal generalization.

---

## Pending (Phase 2 Steps E-F and Beyond)

### ⬜ Step E: FiQA External Validation

**File**: `scripts/11e_eval_finbert_external.py` (NEW, not yet executed)

**Purpose**: Evaluate fine-tuned model from Step D on independent FiQA test set (223 examples)

**Expected to show**: Whether fine-tuning gains transfer or overfit to PhraseBank

---

### ⬜ Step F: Agreement-Aware Fine-Tuning

**File**: `scripts/11f_train_finbert_agreement_weighted.py` (NEW, not yet executed)

**Weight Schedules**:
- `all_equal`: Baseline (1.0 for all)
- `linear`: 0.50/0.66/0.75/1.00 for agreement quartiles
- `strong`: 0.25/0.50/0.75/1.00 for agreement quartiles
- `high_only`: Train only on agreement ≥ 75%, test on full set

**Purpose**: Test whether agreement scores improve or hurt fine-tuning effectiveness

---

### ⬜ Phase 2 Multi-Seed Runs (Not Yet Executed)

**Planned Seeds**: [13, 21, 42, 87, 100]

**Models to run per seed**:
1. Classical baselines (7 models × 5 seeds = 35 runs)
2. FinBERT zero-shot (5 runs)
3. FinBERT fine-tuned (5 runs)
4. FinBERT agreement-weighted (4 schedules × 5 seeds = 20 runs)

**Rationale**: Statistical credibility; capture variance across random seeds

---

## Data Status

**PhraseBank** (4,791 total):
- Train: 3,353 ✓
- Val: 479 ✓
- Test: 959 ✓
- Leakage check: PASS (zero duplicates across splits)

**FiQA** (1,111 total):
- Train: 777 ✓
- Val: 111 ✓
- Test: 223 ✓
- Leakage check: PASS (deduped from 87 pre-split duplicates)

---

## Git History

```
afe33b1 (HEAD -> main) Phase 2 Step A-F: Model Evaluation Pipeline
  - 7 files changed, 962 insertions
  - Scripts 11a, 11d, 11e, 11f added
  - Scripts 10, 11c modified for pre-split support
  - docs/PHASE2_MODEL_EVALUATION_PLAN.md created

5a28568 Phase 1 Final: Data Correction Complete
  - Audit, splits, label mapping implemented
  - All gates PASS (leakage_count=0)
  
9fe248c Phase 1 Initial: Foundational Data Scripts
```

---

## Key Findings

1. **FinBERT zero-shot beats all classical baselines by 17.8% accuracy**
   - Justifies transformer-first strategy
   - Even without fine-tuning, FinBERT is strong

2. **Label mapping verified**
   - 100% accuracy on sanity check
   - Remap logic [1, 2, 0] confirmed correct

3. **Pre-split strategy working**
   - Modified scripts correctly detect and use Phase 1 splits
   - Guarantees no leakage across evaluation sets

4. **Fine-tuning in progress**
   - Estimated completion: ~1-2 hours
   - Will determine if gains are from base model capability or tuning

---

## Next Steps (After Fine-Tuning Completes)

1. Execute FiQA external validation (Step E)
2. Execute agreement-aware fine-tuning (Step F, all 4 schedules)
3. Run multi-seed validation (5 seeds for statistical credibility)
4. Generate leaderboard comparison
5. Create error analysis and confusion matrices
6. Finalize Phase 2 report with conclusions

---

## Session Timeline

| Time | Event |
|------|-------|
| 16:30 | Session started; verified Phase 1 complete |
| 16:32 | Created FinBERT sanity check script (11a) |
| 16:33 | Executed 11a: **PASS** |
| 16:33 | Modified script 10 for pre-split support |
| 16:33 | Executed classical baselines: **tfidf_linear_svm 74.97% acc** |
| 16:34 | Modified script 11c for pre-split support |
| 16:38 | Executed zero-shot FinBERT: **88.32% acc** (+17.8% vs classical) |
| 16:39 | Created scripts 11d, 11e, 11f |
| 16:40 | Committed Phase 2 pipeline (afe33b1) |
| 16:41 | Fine-tuning started (running at 15/630 steps) |
| 16:42 | Created this summary |

---

## Lessons & Notes

- CPU training is slow (~21s/step) but acceptable for Phase 2 (no hardware constraint)
- Pre-split architecture prevents re-splitting mistakes
- Agreement column is present in PhraseBank data (required for weighted training)
- Fine-tuning script includes custom WeightedTrainer for agreement-aware weighting
- All scripts use HF token for model access (verified working)

---

**Status**: Ready to await fine-tuning completion, then proceed to Steps E-F
