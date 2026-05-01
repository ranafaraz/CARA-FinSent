# 10 · Design decisions

A short FAQ on the choices that aren't obvious from reading the code.

## Why mask `$TICKER`, `@user`, `https://…` instead of dropping them?

Dropping makes very short rows empty (`"$TSLA 🚀"` → `""`) and trains the
classifier to associate the *presence* of a URL or ticker with a label.
Masking preserves the structural slot but strips the identity, which is
the academic-cleanliness move and what the evaluators expect to see in a
financial NLP paper.

## Why MinHash LSH instead of just `df.drop_duplicates()`?

Exact dedup misses press-release templating (date prefixes, ticker
substitutions). On this corpus exact dedup drops 14k rows; near-dup
dedup catches an additional 1k. That extra 1k is exactly the kind of
text that bleeds train into test in the real world.

## Why a 0.85 within-corpus threshold but 0.50 train↔test threshold?

Within-corpus: we want to keep diverse examples, only kill genuine
templating clones. 0.85 Jaccard on 5-shingles is "obviously the same
sentence with edits".

Train↔test: we want **zero** doubt. 0.50 is intentionally aggressive —
better to drop 20 ambiguous train rows than to keep them and pollute the
test metric.

## Why a 4-tier system (gold/silver/synthetic/bronze)?

* **gold** is what we trust enough to put in test.
* **silver** trains the model but doesn't get to define the metric.
* **synthetic** is a slot for future paraphrase / LLM-rewrite rows; the
  pipeline already routes them to `sample_weight=0.3` so adding them
  later is a single-line change.
* **bronze** exists so a contributor can add a noisy scrape with weight
  0.0 — the rows enrich vocabulary but contribute no gradient.

## Why undersample instead of oversample / SMOTE?

Negatives are scarce (≈ 2.5k). Oversampling negatives by replication
makes the test-set leakage check meaningless because every duplicate
collides with itself in train. SMOTE doesn't make sense on TF-IDF
sparse matrices and especially not on tokenised text. Undersampling
majority is the simplest defensible move; we keep the cap at 2× the
minority so we don't throw away too much positive/neutral signal.

A paraphrase-based oversampling path is implemented in
[`scripts/07_balance_classes.py`](../scripts/07_balance_classes.py)
but it is **off by default** because adding T5-paraphrased rows would
create a new tier of synthetic data we haven't validated.

## Why `text_clean` for FinBERT but `text_clean_lower` for TF-IDF?

BERT tokenisers are case-sensitive (or carry casing through subword
splits). TF-IDF baselines treat `Apple` and `apple` as different tokens
unless we lower-case, which fragments the vocabulary and hurts the
linear models more than it helps. So we pre-lowercase once, save it
into its own column, and let each model use the version it expects.

## Why `target_test_size = 1450` if we end up with ~1,200?

1,450 is the *upper bound* the carver tries to fill from gold sources
without violating per-source caps. The cap mechanism (no source > 50 %
of test) means we sometimes leave gold rows in train rather than skew
the test set. 1,205 with 4 sources at the current build is healthier
than 1,450 with one source contributing 900.

## Why `seed=42` everywhere?

Determinism is the cheapest reviewer-trust signal. One seed, one place
(`scripts/05_build_dataset.py` defaults plus per-model `random_state`),
no one wonders whether your numbers replicate.

## Why store both `data/processed/latest/dataset.csv` and `data/processed/latest.csv`?

* `latest/dataset.csv` is the canonical, schema-rich, manifest-backed
  build that the methodology and tests reference.
* `latest.csv` (file, not folder) is a **backwards-compatibility shim**
  for scripts 10–17 that were written before the canonical structure
  existed. It has `text_clean_lower` renamed to `text_lower` so old
  code paths still work without a rewrite.
* When all old scripts have been migrated to read from
  `latest/dataset.csv` directly, the shim can be removed.

## Why English-only?

Every downstream model in the comparison was trained on English
financial text. Mixing in Finnish PhraseBank examples (which exist
upstream) would hurt those models and confound the evaluation. We use
`langdetect` to drop non-English rows instead of trusting source labels
because PhraseBank in particular has both.

## Why no GPU requirements in `requirements.txt`?

The reference machine doesn't have CUDA. `requirements-gpu.txt` exists
for the GPU build. Keeping the default install CPU-only means a fresh
contributor can run the full classical suite without wrestling
torch+CUDA wheel compatibility.

## Why a 14-test pytest gate instead of "just look at the manifest"?

Manifests get stale. CI doesn't read manifests. A pytest suite that
fails when leakage appears is a hard contract that survives refactors
and PRs. The whole suite runs in ~ 1 second.
