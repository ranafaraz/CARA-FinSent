# 00 · Overview

## Research goal

Build a small, **honest, well-balanced, license-tracked** financial sentiment
benchmark — call it CARA-FinSent — and evaluate how much classical baselines,
domain-pretrained transformers (FinBERT) and a CARA-style enrichment stack
(retrieval + structured features + calibration + agreement-aware loss) move
the needle on it.

The questions we want to answer:

1. **Where does the ceiling actually sit?** What is the macro-F1 of a strong
   TF-IDF baseline on a *clean* tri-class dataset that has no train/test
   contamination, no near-duplicates, no encoding bugs and a non-trivial
   negative class?
2. **Is FinBERT worth the cost?** On the same clean split, does
   `ProsusAI/finbert` fine-tuned for 3 epochs beat the classical baseline by
   a margin that justifies the runtime?
3. **What does CARA-Lite add?** Do retrieval-augmented features, structured
   features (token counts, ticker counts, length, financial-lexicon hits)
   and post-hoc calibration meaningfully improve macro-F1 and Expected
   Calibration Error?
4. **Is the dataset itself defensible?** Can we hand it to a reviewer and
   pass a 14-test data-quality audit (no leakage, no empty rows, etc.)?

## High-level system

```
                       ┌─────────────────────────┐
                       │   Hugging Face hubs     │
                       │  (PhraseBank, FiQA,     │
                       │   SemEval-2017 task 5,  │
                       │   FOMC sentiment,…)     │
                       └────────────┬────────────┘
                                    │ scripts/00,04,08,09
                                    ▼
                       data/raw/<source>/<ts>.csv
                                    │ scripts/05_build_dataset.py
                                    ▼
        ┌─── preprocessing ──── deduplication ──── test carving ──── balance ───┐
        │      ftfy / NFKC      MinHash LSH        stratified gold     ≤ 2.0×   │
        │      URL/@user/$TKR    threshold 0.85     (1,205 rows)        ratio   │
        │      mask + emoji      tier-aware                                    │
        │      langdetect (≥20)  ↓                                             │
        └──────────────────────────┬────────────────────────────────────────────┘
                                   ▼
                  data/processed/latest/{dataset.csv,manifest.json,data_card.md}
                                   │
            ┌──────────────────────┼─────────────────────────────────────┐
            ▼                      ▼                                     ▼
   tests/test_dataset_quality   scripts/10..16            scripts/17 (compare)
            (14 PASS)           (classical + FinBERT)        leaderboard + plots
```

## Deliverables

| Artefact | Path |
|---|---|
| Cleaned, balanced 3-class CSV | [data/processed/latest/dataset.csv](../data/processed/latest/dataset.csv) |
| Build manifest (provenance + counts) | [data/processed/latest/manifest.json](../data/processed/latest/manifest.json) |
| HF-style data card | [data/processed/latest/data_card.md](../data/processed/latest/data_card.md) |
| Per-build immutable snapshot | [data/processed/archive/](../data/processed/archive) |
| Quality-test suite | [tests/test_dataset_quality.py](../tests/test_dataset_quality.py) |
| Classical baseline metrics | [results/](../results) (per-day folders) |
| FinBERT baseline metrics | same |
| CARA-Lite full-stack metrics | same |
| Leaderboard CSV + plots | `results/comparison_*.csv`, [figures/](../figures) |
| This methodology pack | [methodology/](.) |

## Compute envelope

* CPU: Intel Core 7 150U, 16 GB RAM, no GPU.
* Python 3.11.9.
* PyTorch 2.x CPU build, transformers ≥ 4.39.
* Wall-clock for the canonical pipeline:
  * dataset build (66k → 13.9k rows): ~7 min
  * classical baseline suite: ~2–4 min
  * FinBERT 3-epoch fine-tune on 12.7k rows: 6–10 h on CPU.
* All scripts are CPU-safe; no CUDA assumed.

## Headline dataset stats (build `20260430_212903`)

| | train | test |
|---|---:|---:|
| rows | 12,711 | 1,205 |
| positive | 5,087 | 378 |
| neutral | 5,083 | 483 |
| negative | 2,541 | 344 |
| imbalance ratio | 2.00 | 1.40 |
| sources | 9 | 4 (gold-only) |
| tiers | gold + silver | gold-only |
