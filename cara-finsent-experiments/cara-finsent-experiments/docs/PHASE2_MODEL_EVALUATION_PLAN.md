# CARA-FinSent Phase 2: Model Evaluation Plan

**Date**: 2026-05-01  
**Status**: In Progress  
**Data Readiness**: ✓ PASS

## Phase 2 Objectives

1. Verify FinBERT label mapping (critical fix)
2. Build clean baseline results (classical + FinBERT zero-shot)
3. Fine-tune FinBERT on PhraseBank gold data
4. Cross-validate on FiQA external benchmark
5. Test agreement-aware training on high-confidence examples
6. Establish CARA-lite rebuild criteria
7. Multi-seed runs (5 seeds: 13, 21, 42, 87, 100)
8. Comprehensive reporting (macro-F1, ECE, MCC, class-wise metrics)

## Experiment Pipeline

### Stage A: FinBERT Label Mapping Verification
```bash
# Check that model label order is correct before any eval
python scripts/11a_verify_finbert_label_mapping.py --seed 42
```
Expected: sanity_check passes, label order documented

### Stage B: PhraseBank Main Benchmark
```bash
python scripts/10_run_classical_baselines.py --data data/processed/gold/latest_gold_phrasebank_split.csv --seed 42
python scripts/11c_eval_finbert_zero_shot.py --data data/processed/gold/latest_gold_phrasebank_split.csv --seed 42
python scripts/11d_train_finbert_phrasebank.py --data data/processed/gold/latest_gold_phrasebank_split.csv --seed 42 --epochs 3
```
Expected: clean results on in-domain data; FinBERT fine-tuned should improve over zero-shot

### Stage C: FiQA External Validation
```bash
python scripts/10_run_classical_baselines.py --data data/processed/gold/latest_gold_fiqa_split.csv --seed 42
python scripts/11e_eval_finbert_external.py --data data/processed/gold/latest_gold_fiqa_split.csv --checkpoint models/<best_phrasebank_checkpoint>
```
Expected: cross-dataset robustness check; may be lower than PhraseBank

### Stage D: Agreement-Aware Training
```bash
python scripts/11f_train_finbert_agreement_weighted.py --data data/processed/gold/latest_gold_phrasebank_split.csv --seed 42 --epochs 3 --weight_scheme linear
```
Expected: separate results for high/low agreement test subsets

### Stage E: Multi-Seed Final Runs
```bash
# Run Stages A-D for seeds: 13, 21, 42, 87, 100
for seed in 13 21 42 87 100; do
  python scripts/10_run_classical_baselines.py --data data/processed/gold/latest_gold_phrasebank_split.csv --seed $seed
  python scripts/11d_train_finbert_phrasebank.py --data data/processed/gold/latest_gold_phrasebank_split.csv --seed $seed --epochs 3
  python scripts/11f_train_finbert_agreement_weighted.py --data data/processed/gold/latest_gold_phrasebank_split.csv --seed $seed --epochs 3
done
```
Expected: mean/std CI across seeds

## Required Metrics per Model

- Accuracy
- Macro-F1 (primary)
- Weighted-F1
- Matthews Correlation Coefficient (MCC)
- Per-class recall (esp. neutral vs. negative)
- Expected Calibration Error (ECE, 10 bins)
- Brier score
- Latency (seconds)
- Train/test row counts
- Agreement distribution (if applicable)

## Key Deliverables

| File | Purpose |
|------|---------|
| `docs/PHASE2_MODEL_EVALUATION_PLAN.md` | This file + actual changes |
| `results/phase2_data_readiness_check_*.csv` | Input verification |
| `results/finbert_label_mapping_sanity_*.csv` | FinBERT label check |
| `results/phase2_baseline_leaderboard_*.csv` | Clean model comparison |
| `results/phase2_multiseed_summary_*.csv` | Mean/std/CI across seeds |
| `results/phase2_error_analysis_*.csv` | Confusion, neutral/neg errors |
| `figures/phase2_*.png` | Leaderboard, calibration, confusion |
| `PHASE2_AGENT_REPORT.md` | Final summary |

## Acceptance Criteria

- [ ] FinBERT label sanity check PASS
- [ ] PhraseBank baseline beats majority baseline clearly
- [ ] Fine-tuned FinBERT separate from zero-shot with metrics
- [ ] Agreement-aware results reported (even if no improvement)
- [ ] No weak-labeled data in supervised training
- [ ] All outputs timestamped, seed-specified, dataset-named
- [ ] Multi-seed runs completed for main models
- [ ] Comprehensive error analysis + confusion matrices

## Known Issues / Blockers

- None identified; Phase 1 gate passed

## Notes

- FinBERT native label order: id2label={0:positive, 1:negative, 2:neutral}
- Canonical order: [negative, neutral, positive]
- Remap: [1, 2, 0] to convert model output to canonical
- PhraseBank agreement preserved (0.5, 0.66, 0.75, 1.0)
- No retrieval from test set; no self-retrieval from training corpus
- CARA-lite rebuilt only after baselines stabilize
