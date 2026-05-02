# Phase 13 — Step 6 PEFT Baseline (Deferred)

**Date:** 2026-05-02
**Status:** **Deferred** — not executed in this Phase 13 cycle.
**Decision rule applied:** Plan §6 — *"Optionally run PEFT baseline with Kaggle GPU only.
Do not start RunPod automatically. If Kaggle fails and GPU is required, ask Rana first."*

---

## 1. Why deferred (not skipped)

A parameter-efficient fine-tuning (PEFT) baseline (e.g. LoRA on FinBERT,
LLM2Vec FinBERT-large, or FinMA-7B + LoRA) was scoped as **optional** in the
Phase 13 plan. The recommended decision in the plan itself is:

> "SKIP unless user explicitly requests, document as 'deferred' in integrated report."

The Steps 1–5 results above already deliver the highest-priority Phase 13
goals (external validation, calibration improvement, error mitigation,
ensembles) without GPU, so PEFT does not gate any Phase 13 deliverable.

A PEFT run is **not justified at this point** because:

1. The in-domain ceiling on PhraseBank is already at ~0.89 macro-F1 across
   the FinBERT family (Phase 11). Adding PEFT is unlikely to break this
   ceiling without a much larger backbone (≥ 7B), which itself is out of
   scope for a CPU/Kaggle T4 budget.
2. The most useful Phase 13 finding is the **calibration improvement**
   from Step 3 (isotonic, ECE 0.065 → 0.028) and Step 5 (grid ensemble,
   ECE 0.072 → 0.016). PEFT does not address calibration directly.
3. External validation (Step 2) revealed a **data-quality bug** in FiQA gold
   (PHASE13_FIQA_LABEL_POLARITY_BUG.md) that must be patched before any
   PEFT result on FiQA could be trusted. Running PEFT now would absorb
   the same upstream bug.

---

## 2. What is required to actually run PEFT later

If the user later approves a PEFT run, the minimum requirements are:

| Requirement | Status |
|---|---|
| Kaggle GPU (T4 ×1 or P100) credentials | Available (`~/.kaggle/kaggle.json`, Phase 10 verified) |
| RunPod | **Forbidden by user** — do not start. |
| FiQA polarity bug patched at source | **Open** (PHASE13_FIQA_LABEL_POLARITY_BUG.md) |
| Backbone choice | Decision needed: FinBERT-large vs FinMA-7B vs Llama-3-8B + LoRA |
| Compute budget cap | Decision needed (e.g. ≤ 2 GPU-hours per run, ≤ 5 runs total) |
| Pre-registered hypotheses | Decision needed; current research is reliability-first, not SOTA — must justify why a PEFT run is "reliability evidence" and not just chasing macro-F1 |

A pre-flight script and a Kaggle notebook scaffold can be drafted in a
follow-up Phase 13.x ticket once the user approves.

---

## 3. Honest framing for the paper

Cite Step 6 in the limitations section approximately as:

> "Parameter-efficient fine-tuning of larger backbones (LoRA on FinBERT-large,
> instruction-tuned FinMA-7B, or general-purpose 7–8B language models) was
> scoped but deliberately deferred. Our reliability-first contribution does
> not rest on a SOTA macro-F1 claim, and our in-domain results saturate the
> FinBERT-family ceiling on PhraseBank. We leave a controlled PEFT comparison
> — gated on a corrected FiQA gold split and pre-registered reliability
> hypotheses — to future work."

This deferral is recorded with the same provenance fields (git SHA, date) as
the executed Phase 13 deliverables for audit consistency.
