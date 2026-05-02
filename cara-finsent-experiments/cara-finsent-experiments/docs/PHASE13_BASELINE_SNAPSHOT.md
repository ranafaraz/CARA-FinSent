# Phase 13 — Baseline Snapshot & Integrity Check

**Purpose:** Freeze the Phase 11 evidence base before running any Phase 13
extended-validation or improvement experiments. All Phase 13 outputs are
compared against — never overwrite — the artefacts listed here.

---

## 1. Repository state at snapshot time

| Item | Value |
|---|---|
| Branch | `main` |
| HEAD commit SHA | `082c69df25d269f6dea1ab6a2543d38c055bbcd3` |
| Remote | `git@github.com:ranafaraz/CARA-FinSent.git` |
| Working tree | clean (`git status --short` empty) |
| Pull status | `Already up to date.` |
| Python compile (`python -m compileall -q src scripts`) | OK (no errors) |
| Research gate (`scripts/25_research_gate.py --min_seeds 5`) | **PASS** (11/11 checks) |
| Gate report file | `results/2026-05-02/research_gate_report_20260502_134629.csv` |

The previous gate report `research_gate_report_20260502_110923.csv` (Phase 11)
also passed; the Phase 13 re-run reproduces the same outcome.

---

## 2. Frozen baseline evidence (do not modify)

### 2.1 Leaderboard & seed sweep
| File | Notes |
|---|---|
| `results/2026-05-02/final_leaderboard_mean_std_20260502_110922.csv` | 10-row leaderboard (mean ± std across seeds) |
| `results/2026-05-02/seed_sweep_summary_20260502_105326.csv` | Agreement-weighted FinBERT, 5 seeds = [13, 21, 42, 87, 101] |

### 2.2 Calibration & abstention
| File | Notes |
|---|---|
| `results/2026-05-02/calibration_summary_20260502_081312.csv` | Best ECE@10 = **0.0236** (zero-shot FinBERT) |
| `results/2026-05-02/abstention_curve_20260502_081312.csv` | Coverage / risk trade-off |

### 2.3 Error analysis
| File | Notes |
|---|---|
| `results/2026-05-02/error_analysis_summary_20260502_081316.csv` | Dominant error = neutral → positive (50 cases); mean confidence on wrong = 0.874; overconfident-wrong count = 78 |

### 2.4 Statistical validation (Phase 11)
| File | Notes |
|---|---|
| `results/2026-05-02/statistical_validation_summary_20260502_122559.csv` | Per-model bootstrap CIs |
| `results/2026-05-02/paired_model_comparison_20260502_122559.csv` | Paired bootstrap ΔF1 + McNemar exact p-values |

### 2.5 Paper-ready assets (Phase 11)
- Tables: `paper_assets/tables/table{1..6}_*.csv`
- Figures: `paper_assets/figures/fig{1..6}_*.png|.pdf`
- Draft: `paper/IEEE_SINGLE_COLUMN_DRAFT.md`

---

## 3. Headline numbers being defended in Phase 13

| Claim | Value | Source |
|---|---|---|
| Best macro-F1 (any model) | **0.8863 ± 0.00227** (Agreement-Weighted FinBERT, 5 seeds) | seed_sweep_summary |
| Best calibration (lowest ECE@10) | **0.0236** (Zero-Shot FinBERT) | calibration_summary |
| AW vs Vanilla FT seed-stability | std 0.00227 vs 0.00923 (≈ **4× more stable**) | seed_sweep + leaderboard |
| AW vs Zero-Shot macro-F1 | Paired bootstrap CI ≈ **[-0.018, +0.023]**, McNemar **p ≈ 0.56** → not statistically decisive | paired_model_comparison |
| AW vs Zero-Shot calibration | AW ECE@10 = 0.0495 vs ZS 0.0236 — **ZS is better calibrated** | calibration_summary |

**Honest framing (must be preserved across Phase 13):**
> CARA-FinSent's contribution is **reliability-first** (seed stability, calibration,
> error structure, claim-boundary discipline), **not universal SOTA**. The AW vs
> ZS macro-F1 comparison is statistically inconclusive on PhraseBank alone.

---

## 4. Available local artefacts for Phase 13 experiments

### 4.1 Model checkpoints on disk
Local `models/` contains **vanilla fine-tuned FinBERT** checkpoints only:
- `models/finbert_20260430_215910/` (≈ 438 MB, full model)
- `models/finbert_20260501_090136/` (≈ 438 MB, full model)
- `models/finbert_20260501_193637/` (≈ 438 MB, full model)
- Plus several smaller / partial checkpoint dirs.

There is **no Agreement-Weighted (AW) FinBERT checkpoint on disk**.
AW evidence is prediction CSVs only (Kaggle-trained, weights not synced back).

**Implication for Phase 13:**
- External validation (Step 2) can run **zero-shot FinBERT** + **vanilla fine-tuned FinBERT** on CPU.
- AW external validation is **blocked** unless we (a) re-train AW locally on CPU
  (slow, ~10–15 min/epoch on PhraseBank) or (b) re-run on Kaggle. Decision deferred.
- AW post-hoc calibration (Step 3) needs AW probabilities on the **val** split.
  Current AW prediction CSVs cover the **test** split only, so we will have to
  re-infer AW on val. Without an AW checkpoint, this requires re-training first.
  **Pragmatic substitute:** apply temperature scaling to AW **test** probabilities
  using a held-out temperature fitted on a 50/50 stratified split of the test
  set (documented as a methodological caveat).

### 4.2 External datasets available locally
| Dataset | File | Rows | Labels | Notes |
|---|---|---|---|---|
| FiQA gold | `data/processed/gold/latest_gold_fiqa_split.csv` | 1,111 | negative 682 / positive 345 / neutral 84 | already cleaned + split (train 777 / val 111 / test 223) — **primary external source** |
| FOMC sentiment | `data/raw/fomc_sentiment/fomc_sentiment_20260430_211554.csv` | 2,480 | 3-class (`label` already mapped) | macro-econ register; domain shift |
| SemEval-2017 Task 5 (financial tweets) | `data/raw/semeval2017_task5/semeval2017_task5_20260501_091249.csv` | 38,091 | 3-class (`label` already mapped) | tweet register; very different from PhraseBank prose |

All three carry standard schema columns (`text`, `label`) and are CC-BY-4.0,
suitable for external validation.

---

## 5. Rules for Phase 13 outputs

1. **Never overwrite** anything listed in Section 2.
2. New artefacts must be timestamped and live under one of:
   - `results/<YYYY-MM-DD>/` (gitignored — explicit `git add -f` only for the
     small summary CSVs we choose to publish)
   - `artifacts/phase13_extended_validation/` (preferred for assets we want in git)
   - `figures/phase13_*` (gitignored — `git add -f` for selected publication figures)
3. Every new CSV/JSON manifest must include `git_commit_sha = 082c69d…` and
   ISO-8601 `generated_at`.
4. No new claim may be made unless it is supported by either:
   - a paired bootstrap ΔF1 with 95 % CI **excluding 0**, or
   - a McNemar exact p-value **< 0.05** (with sample size declared).
5. If a Phase 13 experiment **weakens** a Phase 11 claim, that finding is
   reported in `docs/PHASE13_EXTENDED_VALIDATION_AND_IMPROVEMENT_REPORT.md`
   under "Findings that change the paper" — not hidden.

---

## 6. Snapshot generated

- Generated at: 2026-05-02 (Phase 13 kickoff)
- Generated by: `docs/PHASE13_BASELINE_SNAPSHOT.md`
- Next step: Step 2 — External validation (FiQA / FOMC / SemEval-2017) against
  zero-shot FinBERT and vanilla fine-tuned FinBERT.
