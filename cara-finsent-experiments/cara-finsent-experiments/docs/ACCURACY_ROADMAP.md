# Accuracy Improvement Roadmap

## Current Performance

**FinBERT Baseline (Canonical Dataset)**
- Accuracy: 0.583 (58.3%)
- Macro F1: 0.558
- Weighted F1: 0.585
- Matthews Correlation Coefficient: 0.367
- Dataset: 13,916 rows (92.9% gold, 7.0% silver)
- Class Balance: Negative (20%) < Neutral/Positive (40% each), imbalance 2.0×
- Test Set: 1,205 gold rows, stratified by source

**Baseline (Majority Neutral)**: 0.314 accuracy → **FinBERT improves by 1.86×**

**vs. Best Classical (Linear SVM)**: 0.532 accuracy → **FinBERT improves by +5.2 pp macro F1**

## Theoretical Ceiling

Based on label noise analysis and dataset heterogeneity:
- **Estimated Ceiling**: ~0.70–0.75 accuracy (before reaching diminishing returns)
- **Why Lower Than 0.90+?**
  1. Dataset is multi-source with inherent label noise (~15–20% of rows have noisy/ambiguous labels)
  2. SemEval data binarized with ±0.15 threshold creates ambiguous zone (±0.25)
  3. Test set contains diverse sources (gold multi-source: PhraseBank, FiQA, SemEval, FOMC, auditor), each with domain drift
  4. Fine-tuning on limited negatives (20% of dataset) increases false positive rate
  5. Financial sentiment is inherently subjective; human inter-annotator agreement ~0.70

## Phase 1: Quick Fixes (Expected: +3–5 pp macro F1)

### ✅ Issue 1: Agreement-Weighted Training (NEW)
**Status**: IMPLEMENTED in `scripts/11_run_finbert_baseline_improved.py`

**Problem**: Current training treats all samples equally, ignoring annotator agreement levels.

**Solution**: 
- Add `--use_agreement_weight` flag to FinBERT script
- Weight training samples by `agreement` column (0.5–1.0 scale)
- Samples with high agreement (0.9–1.0) get higher gradient signal
- Expected Gain: +1–2 pp macro F1

**Validation**:
```bash
# Run with agreement weighting
python scripts/11_run_finbert_baseline_improved.py \
    --seed 42 --epochs 3 --batch_size 8 \
    --use_agreement_weight
```

### ✅ Issue 2: Down-weight Uncertain Rows (NEW)
**Status**: IMPLEMENTED in `scripts/11_run_finbert_baseline_improved.py`

**Problem**: SemEval data with continuous scores ±0.10–0.15 are inherently ambiguous.

**Solution**:
- Flag rows where `|score| < 0.25` as `is_uncertain=True`
- During training, weight uncertain rows by 0.5x
- Updated `scripts/08_collect_semeval2017.py` to compute `is_uncertain` flag
- Expected Gain: +0.5–1.0 pp accuracy (mainly reduces false positives)

**Validation**:
```bash
# Run with uncertainty down-weighting
python scripts/11_run_finbert_baseline_improved.py \
    --seed 42 --epochs 3 --batch_size 8 \
    --down_weight_uncertain --uncertain_threshold 0.3
```

### ✅ Issue 3: Increase Synthetic Negatives (NEW)
**Status**: IMPLEMENTED in `scripts/07_balance_classes_improved.py`

**Problem**: Negative class only 20% of training set; few negative samples → model underestimates negatives.

**Solution**:
- Increase T5-based paraphrasing from 30% cap to 50%+ synthetic generation
- Tier-aware undersampling: drop silver before gold to preserve high-quality data
- Tag synthetic negatives with `is_synthetic=True` for downstream analysis
- Expected Gain: +1–2 pp recall for negatives, +0.5–1.0 pp macro F1

**Validation**:
```bash
# Rebuild with 50% synthetic intensity
python scripts/07_balance_classes_improved.py \
    --input data/processed/latest/dataset.csv \
    --output data/processed/latest/balanced_increased_synthetic.csv \
    --synthetic_intensity 0.5
```

### ✅ Issue 4: Confidence & Abstention Scoring (NEW)
**Status**: PARTIALLY IMPLEMENTED in `scripts/11_run_finbert_baseline_improved.py`

**Problem**: No way to abstain on low-confidence predictions; precision/recall trade-off is hidden.

**Solution**:
- Extract `confidence = proba.max(axis=1)` from FinBERT outputs
- Use Expected Calibration Error (ECE) to set optimal abstention threshold
- Output `is_abstain=True` when confidence < threshold (default 0.65)
- Expected Impact:
  - Non-abstained accuracy: +2–3 pp higher
  - Coverage: 80–90% (10–20% of test set)
  - Example: Instead of 0.583 overall, could achieve 0.62 on 90% of data

**Validation**: Already enabled; check predictions CSV for `confidence` column.

### ⏳ Issue 5: SemEval Borderline Audit (FUTURE)
**Status**: TOOLING READY, requires manual review

**Problem**: ~500 rows with |score| ∈ [0.10, 0.25] are genuinely ambiguous. Some may warrant relabeling or removal.

**Solution**:
- Create `scripts/18_audit_borderline_cases.py` to export borderline rows
- User manually reviews CSV, decides: keep / discard / relabel
- Rebuild dataset with refined labels
- Expected Gain: +1–2 pp depending on manual decisions

---

## Phase 2: Moderate-Effort Improvements (+2–3 pp macro F1)

### ✅ Ensemble Stacking (FUTURE)
**Status**: Framework available in `scripts/16_run_full_cara_lite_experiment.py`

**Problem**: FinBERT captures semantic meaning but misses domain-specific signals (price volatility, financial terminology).

**Solution**:
- Combine FinBERT logits (trained on text) with structured features (TF-IDF + financial signals)
- Soft voting or meta-learner stacking
- Expected Gain: +1–2 pp if FinBERT & structured features are uncorrelated
- Note: Structured features alone score 0.537; combined could reach ~0.60

### Domain Adaptation (FUTURE)
**Status**: Requires new data collection

**Scenario**: Fine-tune FinBERT on domain-specific corpora before final training.

**Options**:
1. **Masked Language Modeling (MLM)** on financial news corpus (Reddit, SEC filings, Bloomberg summaries)
2. **Contrastive pre-training** on sector-specific pairs (bull vs. bear sentiment)

**Expected Gain**: +2–3 pp (better financial vocabulary representation)

---

## Phase 3: Long-Term Research Directions (+3–5 pp, requires significant work)

### Multi-Task Learning
- Joint training: sentiment + financial aspect extraction + price movement prediction
- Expected gain: +2–3 pp (auxiliary tasks regularize learning)

### Retrieval-Augmented Fine-Tuning
- Retrieve similar examples with high agreement during training
- Use retrieved context to disambiguate borderline cases
- Expected gain: +1–2 pp

### Active Learning
- Identify most uncertain predictions
- Ask human annotators to label them
- Re-train with new labels
- Expected gain: +3–5 pp (with ~100 manual labels)

### Hierarchical Classification
- First classify: sentiment is strong (confident) vs. weak (uncertain)
- Then: predict label only for strong-sentiment examples
- Skip weak ones (abstain)
- Expected gain: +2–3 pp accuracy on decided examples

---

## Implementation Priority

### Before Next Experiments

1. ✅ **Agreement-weighted training** (scripts/11_run_finbert_baseline_improved.py)
   - Effort: LOW (1 hour)
   - Expected gain: +1–2 pp
   - Blocker: None
   
2. ✅ **Uncertainty down-weighting** (scripts/08_collect_semeval2017.py updated)
   - Effort: LOW (30 min)
   - Expected gain: +0.5–1.0 pp
   - Blocker: None

3. ✅ **Increased synthetic generation** (scripts/07_balance_classes_improved.py)
   - Effort: MEDIUM (2 hours, T5 inference can be slow)
   - Expected gain: +1–2 pp
   - Blocker: GPU or patience required

4. ✅ **Confidence scoring** (already in improved script)
   - Effort: DONE (logs confidence in predictions CSV)
   - Expected gain: Enables abstention trade-offs

### Next Session (if Phase 1 gives disappointing results)

5. ⏳ **SemEval borderline audit** (manual review)
   - Effort: HIGH (3–4 hours with domain expert review)
   - Expected gain: +1–2 pp
   - Blocker: Requires user decision-making

6. ⏳ **Ensemble stacking** (scripts/19_ensemble_stacking.py)
   - Effort: MEDIUM (2 hours)
   - Expected gain: +1–2 pp
   - Blocker: Requires both FinBERT and structured models trained

---

## How to Run Full Ablation

```bash
# 1. Quick smoke test: agreement weighting only
python scripts/11_run_finbert_baseline_improved.py \
    --seed 42 --epochs 1 --batch_size 8 --max_rows 1000 \
    --use_agreement_weight

# 2. Full Phase 1 implementation
python scripts/11_run_finbert_baseline_improved.py \
    --seed 42 --epochs 3 --batch_size 8 \
    --use_agreement_weight --down_weight_uncertain

# 3. Systematic ablation: try all configurations
python scripts/20_run_ablation_analysis.py \
    --seed 42 --epochs 2 --batch_size 8

# 4. Compare all results
python scripts/17_compare_all_experiments.py
```

---

## Success Metrics

**Target for Phase 1 + 2**: Reach **0.62–0.65 accuracy** (5–7 pp improvement)

- Accuracy: ≥0.62
- Macro F1: ≥0.59
- Weighted F1: ≥0.62
- MCC: ≥0.42

**Evidence of success**:
- FinBERT outperforms all baselines (currently does: 0.583 vs best classical 0.532)
- Improvement from each component is measurable (ablation shows +0.5–1.0 pp per step)
- Negative class recall improves (currently often confused with neutral)

---

## Debugging If Improvements Don't Materialize

If Phase 1 gives <1 pp gain:

1. **Check data quality**: Run `pytest tests/test_dataset_quality.py -v`
   - Verify `is_uncertain` flag is actually being used
   - Check agreement column has variance (not all 1.0 or all NaN)

2. **Verify weighting is applied**: Add logging to `WeightedTrainer.compute_loss()`
   - Print sample weights distribution before training
   - Confirm weights are actually affecting gradients

3. **Check for data leakage**: Run contamination check again
   - Some train-test overlap could be hiding improvements
   - Move any borderline pairs to test-only

4. **Try different thresholds**: Experiment with `--uncertain_threshold 0.4` or 0.2

5. **Increase epochs**: With agreement weighting, needs more epochs to converge

---

## References

- Label noise estimation: Confident Learning (Northcutt et al., 2021)
- Agreement weighting: Raykar & Yu, 2012 ("Learning from crowds")
- SemEval binarization: Cortis et al., 2017 (SemEval-2017 Task 5)
- FinBERT: Huang et al., 2022 (Domain-specific BERT for finance)
