# Phase 13 — Step 8 RAG Extension Plan (Plan-Only, Not Executed)

**Date:** 2026-05-02
**Status:** Design document only. **No code committed under this step.**
**Purpose:** Capture a concrete, runnable design for a retrieval-augmented
extension of CARA-FinSent so that a future phase can execute it without
re-deciding scope. Intentionally narrow — reliability-first, not SOTA-chasing.

---

## 1. Hypothesis

> *Retrieving 3–5 PhraseBank sentences with similar lexical/semantic
> structure, conditioning AW FinBERT on those exemplars at inference time,
> reduces both (a) neutral→positive errors on PhraseBank-test and (b) ECE@10
> on the AW probability vector — without retraining and without changing the
> training data.*

This is a **reliability hypothesis**, not an accuracy hypothesis.

---

## 2. Architecture (one paragraph)

For each test sentence $x$:
1. Encode $x$ with `sentence-transformers/all-MiniLM-L6-v2` (CPU-friendly).
2. Retrieve top-$k$ ($k = 3$ default) nearest neighbours from the PhraseBank
   **train** split using cosine similarity over pre-cached embeddings.
3. Build a context string: each neighbour's `text` and `label`, concatenated
   with separator tokens.
4. Run AW FinBERT on the **concatenated `[context] [SEP] x`** input (truncated
   to 256 tokens, prioritising $x$ in the right-truncation step).
5. Output the AW probability vector.

The AW backbone weights are **not** modified. Only the input is augmented.
This is a strict zero-training intervention.

---

## 3. Why this might work (and why it might not)

### Reasons it might help reliability
- AW errors concentrate on neutral→positive (Phase 8 / Phase 13 Step 4).
  Many of these errors are on sentences that *resemble* training-time
  positives but are factually neutral. Surrounding the test sentence with
  retrieved neutrals from train should pull the model's prior toward
  neutral.
- Calibration is bounded by the model's marginal distribution. RAG context
  effectively conditions the marginal, which can flatten over-confident
  positives.

### Reasons it might not
- FinBERT was not trained as a retrieval-aware reader. Concatenating context
  may inject noise rather than signal.
- The PhraseBank train set is small (3,353 rows) and label distribution is
  skewed (~60 % neutral). Retrieved neighbours will be neutral-biased,
  which could push every prediction toward neutral and hurt accuracy.
- 256-token truncation may drop critical context tokens.

A pre-registered abandonment criterion: **if accuracy on PhraseBank-test
drops by ≥ 1.5 pp absolute and ECE@10 does not improve by ≥ 25 % relative,
abandon the approach** and document as a negative result.

---

## 4. Concrete script skeleton (to be written in Phase 13.x)

`scripts/40_rag_inference.py` (proposed):

```text
inputs:
  --aw_model_repo  <hf-id-or-local>             # AW checkpoint to load
  --train_data     data/processed/gold/latest_gold_phrasebank_split.csv
  --test_data      data/processed/gold/latest_gold_phrasebank_split.csv
  --k              3
  --max_length     256
  --embed_model    sentence-transformers/all-MiniLM-L6-v2
  --cache_dir      data/processed/retrieval/phrasebank_train_minilm/

steps:
  1. embed train texts (cache to .npy)
  2. embed test texts
  3. cosine top-k per test row -> list of (id, text, label)
  4. build context = "Examples:\n- [pos] X\n- [neg] Y\n- [neu] Z\nQuery:"
  5. tokenise, truncate-from-left to keep query intact
  6. forward through AW (no_grad, CPU OK for 959 rows)
  7. record per-id probs + chosen neighbours

outputs:
  results/<date>/phase13_rag_predictions_<ts>.csv
  results/<date>/phase13_rag_summary_<ts>.csv
  results/<date>/phase13_rag_manifest_<ts>.json
  artifacts/phase13_extended_validation/rag_summary_<ts>.csv
  artifacts/phase13_extended_validation/rag_manifest_<ts>.json

statistical comparison vs AW non-RAG:
  reuse scripts/32_statistical_validation.py helpers
  paired bootstrap delta_macro_f1 + McNemar exact
```

---

## 5. Compute budget

| Item | Estimate |
|---|---|
| Embedding train (3,353 sentences) with MiniLM CPU | ≈ 1–2 min |
| Embedding test (959 sentences) | ≈ 30 s |
| AW forward over 959 RAG-augmented inputs at max_length 256 (CPU) | ≈ 6–10 min |
| Total | **≤ 15 minutes CPU**, no GPU required |

If the AW checkpoint is not on disk (current state — see
`PHASE13_BASELINE_SNAPSHOT.md` §4.1), a one-time CPU fine-tune of FinBERT
with agreement weighting is required first (≈ 60–90 min on CPU for 3 epochs
on 3,353 sentences) — or re-pull from Kaggle.

---

## 6. Out of scope for this plan

- LLM-based RAG (e.g. RAG-fusion, GPT-4 reader). Reason: latency, cost, and
  reproducibility — incompatible with our reliability-first framing.
- Retrieval over external corpora (FiQA, SEC, FOMC). Reason: blocked by FiQA
  polarity bug; sec/fomc pre-processing pipelines have not been audited.
- Re-training AW with retrieval-aware loss. Reason: out of "no retraining"
  envelope; would change the unit of comparison.

---

## 7. Decision required from user before execution

1. Approve the abandonment criterion above (acc drop ≥ 1.5 pp AND ECE
   improvement < 25 % → abandon)?
2. Confirm $k = 3$ neighbours and `max_length = 256` as defaults?
3. Approve a CPU-only AW re-training (≈ 90 min) if the AW checkpoint cannot
   be re-pulled from Kaggle? Or is this run gated on Kaggle availability?

This document does **not** consume any GPU budget and ships no executable
code. Execution requires explicit user approval in a follow-up turn.
