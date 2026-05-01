# 03 · Deduplication and contamination control

Lives in [`src/cara_finsent/dedup.py`](../src/cara_finsent/dedup.py). Run as
step 3 (within-corpus dedup) and step 6 (train↔test contamination check) of
[`scripts/05_build_dataset.py`](../scripts/05_build_dataset.py).

## Why this matters

A surprising number of "FinBERT crushes the baseline" claims in the
literature evaporate once you remove duplicates and near-duplicates from
the train/test split. Two common pitfalls in the financial-NLP corpora we
fuse:

1. **Cross-source duplicates.** PhraseBank sentences sometimes appear
   verbatim in the FiQA corpus and the financial-news scrape.
2. **Near-duplicates from press-release templating.** Reuters / NORDIC
   BUSINESS REPORT etc. recycle the same sentence with a date prefix or
   ticker substitution. Levenshtein distance is small, exact match misses.

We therefore run **two stages**: exact dedup, then MinHash-LSH near-dup.

## Stage 1 — exact dedup

After preprocessing, drop rows whose `text_clean` is exactly equal across
the corpus, keeping the highest-tier row. Tier order:
`gold > silver > synthetic > bronze`.

```python
TIER_RANK = {'gold': 3, 'silver': 2, 'synthetic': 1, 'bronze': 0}
```

For build `20260430_212903`: 14,260 rows dropped (≈ 22 % of the input).
Most of these come from the FOMC / news scrape stitched up against
SemEval+TimKoornstra duplicates.

## Stage 2 — MinHash LSH near-dup

For every remaining row, build a MinHash signature with `num_perm=128` over
**word-level 5-shingles** (`_shingles(text, k=5)`). Insert into a
`MinHashLSH` index with Jaccard `threshold=0.85`. Two rows are considered
near-duplicates when their signatures collide. We then keep the
highest-tier survivor per cluster (ties broken by lower row index → stable
order).

For build `20260430_212903`: 1,015 additional rows dropped after exact
pass.

## Stage 3 — train ↔ test contamination check

After class balancing (Section 04), we recheck train against the carved
test set with the **same** MinHash machinery but a more permissive
`threshold=0.5` — i.e. we accept Jaccard 0.5 as a "leak". Any train row
that collides with any test row is dropped. The pairs are written to
`data/processed/latest/contamination_report.csv` for audit.

For build `20260430_212903`: 20 contamination pairs found and removed.

## Knobs

| Knob | Default | Where set |
|---|---|---|
| Shingle size `k` | 5 | `_shingles` |
| MinHash perms | 128 | `dedup_dataframe`, `contamination_check` |
| Within-corpus threshold | 0.85 | `dedup_dataframe(threshold=…)` |
| Train↔test threshold | 0.50 | `contamination_check(threshold=…)` |
| Tier ranking | gold > silver > synthetic > bronze | `TIER_RANK` |

## Diagnostic output

The build manifest stores 20 sample near-dup pairs under
`dedup.sample_pairs`. They are intentionally short truncations so the
file stays grep-able. Eye-balling these every build is the cheapest way
to catch a mojibake regression — e.g. `"-\u00a6"` vs `"-\""` clusters
mean ftfy is missing some characters.

## Sanity tests covering this stage

* `tests/test_dataset_quality.py::test_no_train_test_text_overlap`
* `tests/test_dataset_quality.py::test_within_source_no_exact_dupes`
