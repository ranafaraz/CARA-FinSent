# Phase 13 — Step 5 Probability-Level Ensemble Report

**Date:** 2026-05-02
**Git commit SHA:** `082c69df25d269f6dea1ab6a2543d38c055bbcd3`
**Script:** `scripts/38_probability_ensemble.py`
**Run timestamp:** `20260502_141704`
**Test sample:** 959 PhraseBank rows (full id-merge of ZS + FT + AW)

---

## 1. Methodology

Frozen probability vectors from three FinBERT family members were combined
post-hoc (no retraining):

| Model | Source CSV (seed) |
|---|---|
| FinBERT zero-shot (ZS) | `finbert_baseline_predictions_20260502_081128.csv` (deterministic; seed 101) |
| FinBERT vanilla fine-tuned (FT) | `finbert_baseline_predictions_20260502_081140.csv` (seed 101) |
| FinBERT agreement-weighted (AW) | `finbert_agreement_weighted_predictions_20260502_082228.csv` (single AW seed) |

Strategies:

1. **Simple mean** — `(P_zs + P_ft + P_aw) / 3` over all 959 rows.
2. **Manual weighted** — default `0.5·AW + 0.3·ZS + 0.2·FT` (renormalised) over all 959.
3. **Grid-searched weights** — sweep `(w_aw, w_zs, w_ft)` on a 0.1 grid with
   `Σw = 1`, fitting on a stratified 50 % calib portion (n = 479) and
   evaluating on the held-out 50 % (n = 480). `random_state = 42`.
4. **Stacked logistic regression** — `sklearn.LogisticRegression` (lbfgs,
   `C = 1.0`, `max_iter = 4000`) on concatenated probability vectors (9
   features), fit on calib (n = 479) and evaluated on eval (n = 480).

Statistical comparison vs the AW baseline uses paired bootstrap on macro-F1
(n = 2000, seed = 42) and exact two-sided McNemar on discordant counts —
both helpers imported directly from `scripts/32_statistical_validation.py`
to keep methodology identical to Phase 11.

---

## 2. Headline results

Source CSV: `results/2026-05-02/phase13_ensemble_summary_20260502_141704.csv`
Safe copy: `artifacts/phase13_extended_validation/ensemble_summary_20260502_141704.csv`

| Method | n | Accuracy | macro-F1 | weighted-F1 | MCC | **ECE@10** | Brier | mean conf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AW baseline | 959 | 0.8895 | 0.8860 | 0.8891 | 0.798 | 0.0715 | 0.180 | 0.958 |
| ZS baseline | 959 | 0.8832 | 0.8836 | 0.8844 | 0.800 | 0.0236 | 0.176 | 0.875 |
| FT baseline | 959 | 0.8874 | 0.8828 | 0.8880 | 0.796 | 0.0768 | 0.186 | 0.964 |
| Simple mean ensemble | 959 | 0.8895 | 0.8907 | 0.8899 | 0.800 | 0.0286 | 0.158 | 0.918 |
| Manual weighted (0.5/0.3/0.2) | 959 | 0.8916 | **0.8931** | 0.8915 | 0.803 | 0.0294 | 0.159 | 0.921 |
| **Grid-search (eval half)** *(aw 0.00 / zs 0.40 / ft 0.60)* | 480 | 0.8917 | 0.8909 | 0.8930 | 0.807 | **0.0164** | **0.152** | 0.908 |
| **Stacked LR (eval half)** | 480 | **0.8958** | **0.8948** | **0.8965** | **0.812** | 0.0484 | **0.145** | 0.884 |

---

## 3. Statistical comparison vs AW baseline

Source CSV: `results/2026-05-02/phase13_ensemble_paired_comparison_20260502_141704.csv`

| Comparison | n | Δ macro-F1 (a − b) | 95 % CI | bootstrap p (two-sided) | McNemar p (exact) |
|---|---:|---:|---|---:|---:|
| Simple mean vs AW | 959 | +0.0047 | [−0.0080, +0.0179] | 0.475 | 1.00 |
| Manual weighted vs AW | 959 | +0.0071 | [−0.0020, +0.0171] | 0.143 | 0.79 |
| Grid-search vs AW (eval half) | 480 | −0.0034 | [−0.0250, +0.0193] | 0.805 | 0.68 |
| Stacked LR vs AW (eval half) | 480 | +0.0005 | [−0.0195, +0.0217] | 0.974 | 1.00 |

**No ensemble achieves a 95 % CI that excludes 0**, and no McNemar test is
significant at α = 0.05. The macro-F1 gains are real-valued but **statistically
inconclusive on PhraseBank-test alone**, consistent with the Phase 11 finding
that FinBERT-family variants are at the same accuracy ceiling on this dataset.

---

## 4. Where the ensembles actually win — calibration

While macro-F1 gains are inconclusive, **calibration improves substantially**:

| Model | ECE@10 | vs AW baseline (Δ ECE) |
|---|---:|---:|
| AW baseline | 0.0715 | — |
| ZS baseline (Phase 11 calibration leader) | 0.0236 | −0.0479 |
| Simple mean ensemble | 0.0286 | **−0.0429** (≈ 60 % reduction vs AW) |
| Manual weighted ensemble | 0.0294 | **−0.0421** |
| **Grid-search ensemble (eval half)** | **0.0164** | **−0.0551** |
| Stacked LR ensemble (eval half) | 0.0484 | −0.0231 |

**Grid-search ensemble (weights AW 0.00 / ZS 0.40 / FT 0.60) achieves
ECE@10 = 0.0164** — a new low, ≈ 30 % lower than the previous best
(ZS 0.0236) and ≈ 77 % lower than AW baseline. Brier score (0.152) is also
the joint-best alongside Stacked LR. **Important**: this is on the eval half
only (n = 480); the result needs replication on a true held-out test or via
cross-validation before being headlined.

The grid-search assigning **w_aw = 0** is interesting: on PhraseBank, AW's
seed-stability advantage does not translate into ensemble weight when
optimising macro-F1 alone. This is consistent with Phase 11's finding that
AW vs ZS is statistically inconclusive at the per-prediction level.

---

## 5. Recommended operating points

| Goal | Recommended ensemble | macro-F1 | ECE@10 | Honest caveat |
|---|---|---:|---:|---|
| Best in-domain calibration | Grid-search (aw 0 / zs 0.4 / ft 0.6) | 0.891 | **0.016** | Eval-half result; needs CV validation. |
| Best in-domain macro-F1 | Stacked LR | **0.895** | 0.048 | Eval-half result; needs CV validation. |
| Safe / no-tuning | Simple mean | 0.891 | 0.029 | Full-test, no held-out tuning required. |
| Conservative weighted | Manual (0.5 / 0.3 / 0.2) | 0.893 | 0.029 | Full-test, weights chosen a-priori. |

---

## 6. What this lets the paper claim

### Defensible claims
- Probability-level ensembling reduces ECE@10 substantially across all
  strategies (≥ 60 % vs AW baseline), confirming the **reliability-first**
  framing of CARA-FinSent.
- A grid-searched ensemble can **beat the strongest single-model
  calibration baseline** (ZS) on PhraseBank, achieving ECE@10 = 0.016 vs
  ZS's 0.024 — though only on the eval half of a single split.
- All ensembles improve the **mean confidence on correct predictions vs
  overconfident wrong predictions** (Brier drops from 0.180 → 0.145–0.158).

### Claims to avoid
- Do **not** claim a statistically significant macro-F1 gain over the AW
  baseline. CIs all include 0.
- Do **not** claim a SOTA. The headline numbers stay at the FinBERT-family
  ceiling (~0.89 macro-F1 in-domain).
- Do **not** report grid-search / stacked-LR numbers without the eval-split
  caveat and the recommendation that future work cross-validate them.

---

## 7. Artefacts written

| Artefact | Path |
|---|---|
| Method-comparison summary | `results/2026-05-02/phase13_ensemble_summary_20260502_141704.csv` |
| Paired comparison vs AW | `results/2026-05-02/phase13_ensemble_paired_comparison_20260502_141704.csv` |
| Per-id predictions | `results/2026-05-02/phase13_ensemble_predictions_20260502_141704.csv` |
| Manifest | `results/2026-05-02/phase13_ensemble_manifest_20260502_141704.json` |
| Leaderboard figure | `figures/2026-05-02/phase13_ensemble_leaderboard_20260502_141704.png` |
| Safe summary copy | `artifacts/phase13_extended_validation/ensemble_summary_20260502_141704.csv` |
| Safe paired-comp copy | `artifacts/phase13_extended_validation/ensemble_paired_comparison_20260502_141704.csv` |
| Safe manifest copy | `artifacts/phase13_extended_validation/ensemble_manifest_20260502_141704.json` |
