# FiQA Benchmark Decision

**Status (2026-05-02):** *Pending audit — provisionally treated as **external stress test only**.*

## Background

Phase 3 measurements of `ProsusAI/finbert` (zero-shot, no fine-tuning) on
`data/processed/gold/latest_gold_fiqa_split.csv` returned:

| Setting | Accuracy | Macro-F1 |
| --- | --- | --- |
| FinBERT zero-shot, FiQA test | ~0.13 | ~0.13 |
| PhraseBank → FiQA transfer (classical) | ~0.11 | ~0.10 |
| FinBERT zero-shot, PhraseBank test | ~0.88 | ~0.88 |

The drop is too large to be attributed to model weakness alone; it is far more
consistent with a label-semantics or domain mismatch.

## Hypothesised causes

1. The chosen FiQA source carries **continuous sentiment scores** that were
   bucketised into `negative / neutral / positive` using a project-specific
   threshold. The threshold may not align with how FinBERT was trained.
2. FiQA contains **microblog and headline** text very different from
   PhraseBank's investor-relations sentences.
3. The dataset may be **aspect-conditional** — the released sentiment is the
   sentiment of an aspect inside the text, not the global sentiment of the
   sentence.

Inspecting the standardized export
(`data/processed/2026-05-01/fiqa_standardized_20260501_193108.csv`) shows clear
sarcasm or speculative phrasing labelled positive (e.g. *"$PLUG bear raid"*),
which suggests aspect or score-thresholding effects.

## Decision

> Until `scripts/23_fiqa_semantics_audit.py` is run online and a label-semantics
> manifest is produced (currently blocked in the offline CPU environment),
> **FiQA is retained only as an external stress test**. It is not a primary
> in-domain benchmark and FinBERT FiQA results must not be presented as a
> failure of FinBERT *as such* — they reflect dataset/label mismatch.

When the audit runs:

* If `scripts/23_fiqa_semantics_audit.py` reports `n_unmapped == 0`, no class
  exceeds 85% prevalence, and the labels are not "mostly numeric", FiQA may be
  promoted to a secondary in-domain benchmark.
* Otherwise, this document remains the canonical decision and FiQA stays as a
  stress-test-only dataset.

## Implications for the paper

* All FiQA in-domain numbers are reported with the explicit caveat above.
* Cross-domain transfer numbers (PhraseBank → FiQA) are framed as evidence of
  poor cross-annotation-scheme generalisation, *not* evidence that the source
  models are broken.
* The retrieval corpus must not draw from FiQA text.
