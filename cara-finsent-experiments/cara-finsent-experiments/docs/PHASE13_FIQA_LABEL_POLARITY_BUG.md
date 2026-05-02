# Phase 13 — FiQA Gold Split Label Polarity Bug

**Discovered:** 2026-05-02, Phase 13 Step 2.
**Severity:** Blocking for any FiQA-based result that does not apply runtime correction.
**Affected files (raw):**
- `data/processed/gold/latest_gold_fiqa.csv` (1,213 rows)
- `data/processed/gold/latest_gold_fiqa_split.csv` (1,111 rows: train 777 / val 111 / test 223)

**Status:** Documented; runtime correction applied in `scripts/35_external_validation.py`.
Raw files on disk are **unchanged** to preserve audit history. A future Phase 13.x patch should fix the upstream pipeline (`scripts/00_prepare_phrasebank_fiqa.py` or its label-mapping helper) and republish the gold split.

---

## 1. Symptom

Both zero-shot `ProsusAI/finbert` and locally fine-tuned FinBERT checkpoints scored
**below random** (macro-F1 ≈ 0.13 on a 3-class task) on the FiQA gold test split,
while the same zero-shot model scores **0.884 macro-F1** on the PhraseBank gold test
split (where labels are correct). This was not consistent with FinBERT being
extensively pre-trained on financial text.

---

## 2. Root cause — verified by inspecting raw rows

Examining individual rows (script: `scripts/35_external_validation.py`, then
`pd.read_csv` on the raw gold file) showed positive and negative labels are
**systematically swapped** at the data layer.

### Examples — labelled `negative` in FiQA gold but clearly positive in text

| Text | FiQA gold label |
|---|---|
| "How Kraft-Heinz Merger Came Together in Speedy 10 Weeks" | negative |
| "$AAPL bounces off support, it seems" | negative |
| "Johnson Matthey raises prospect of investor payout" | negative |
| "Black Friday best ever for Amazon Kindle family" | negative |
| "UPDATE 1-AstraZeneca sells rare cancer drug to Sanofi for up to $300 mln" | negative |

### Examples — labelled `positive` in FiQA gold but clearly negative in text

| Text | FiQA gold label |
|---|---|
| "Still short $LNG from $11.70 area...next stop could be down through $9.00" | positive |
| "$PLUG bear raid" | positive |
| "Slump in Weir leads FTSE down from record high" | positive |
| "$TSLA Recalls 2,700 Model X Vehicles; Shares Volatile" | positive |
| "78 users on Vetr are bearish on Tesla Motors ... SELL Rating ... $TSLA" | positive |
| "Reuters: Green Mountain revenue misses, shares plunge ... bad day to disappoint" | positive |

The semantics are unambiguous: "bear raid", "SELL rating", "shares plunge",
"down through", "slump", "recall" → all labelled positive. "Bounces off
support", "raises prospect of investor payout", "best ever Black Friday",
"sells drug for $300M" → all labelled negative.

`neutral` rows appear unaffected.

---

## 3. Most likely upstream cause

FiQA-Task1 ships continuous sentiment scores in $[-1, +1]$. The pipeline that
binarised these scores (presumably in `scripts/00_prepare_phrasebank_fiqa.py`
or the underlying HuggingFace dataset wrapper) likely applied an **inverted
sign convention** — e.g., negative scores were mapped to `positive` and vice
versa. This must be confirmed and patched in a separate Phase 13.x ticket.

The FinBERT fine-tuned checkpoint `models/finbert_20260430_215910/` shows an
**MCC of −0.48** on the polarity-corrected FiQA test split, which is strong
indirect evidence that this checkpoint was trained on the **uncorrected**
(inverted) FiQA labels — i.e., it learned the bug.

---

## 4. Runtime mitigation in Phase 13 Step 2

`scripts/35_external_validation.py` accepts `--swap_pos_neg` (default **True**)
which, immediately after `load_gold_split`, applies:

```python
swap_map = {'positive': 'negative', 'negative': 'positive'}
test_df['label'] = test_df['label'].map(lambda x: swap_map.get(x, x))
```

Every output artefact (predictions CSV, summary CSV, manifest JSON) records
`label_polarity_corrected = True` and includes
`label_polarity_correction_reason` pointing to this document.

The raw gold split on disk is **not modified**.

---

## 5. Implications for prior results

| Artefact | Affected? | Notes |
|---|---|---|
| PhraseBank in-domain leaderboard | **No** | PhraseBank gold split is independent and labelled correctly (verified in Phase 8). |
| All Phase 8/9/10/11 reported numbers | **No** | All headline numbers (AW 0.8863, ZS 0.883, ECE 0.0236, etc.) are PhraseBank in-domain. |
| Any FiQA fine-tuned model on disk | **Yes (trained on inverted labels)** | `models/finbert_20260430_215910/` MCC = −0.48 on corrected FiQA test confirms it learned the inversion. These checkpoints should be re-trained or excluded from external-validation conclusions. |
| Phase 13 Step 2 external-validation report | Mitigated at runtime | All Phase 13 Step 2 numbers are reported on the polarity-corrected FiQA test split. |

---

## 6. Recommended follow-up (Phase 13.x or Phase 14)

1. Open `scripts/00_prepare_phrasebank_fiqa.py`; locate the FiQA label-mapping
   block; verify the score → {negative,neutral,positive} thresholding.
2. Patch the mapping; regenerate `data/processed/gold/latest_gold_fiqa*.csv`
   with a new timestamp; archive the inverted version under
   `data/processed/archive/inverted_fiqa_<ts>/`.
3. Re-train any FiQA-only checkpoints used for downstream analysis.
4. Re-run `scripts/35_external_validation.py` with `--no_swap_pos_neg`
   against the corrected gold split; the headline zero-shot macro-F1 should be
   approximately the same (~0.46) since the runtime swap and the data fix are
   equivalent operations.
5. Add a regression test in `tests/test_imports.py` (or a new
   `tests/test_label_polarity.py`) that asserts a known clearly-positive
   FiQA text never carries label `negative`.

---

## 7. Provenance of this finding

- Run timestamp: `20260502_135135` (uncorrected baseline showing MCC ≈ −0.25)
- Run timestamp: `20260502_140602` (polarity-corrected, all artefacts under `results/2026-05-02/`)
- Git commit SHA at discovery: `082c69df25d269f6dea1ab6a2543d38c055bbcd3`
