# CARA-FinSent — Final Research Status

Date: 2026-05-02
Latest commit (pre-Phase-11): `d0e119faf59fc735d577b61299e64234d26320ee`.

## Headline numbers

| metric | value | source |
|---|---|---|
| best model by mean macro-F1 | `finbert_agreement_weighted` (0.8863 ± 0.00227, n=5) | [results/2026-05-02/final_leaderboard_mean_std_20260502_110922.csv](../results/2026-05-02/final_leaderboard_mean_std_20260502_110922.csv) |
| best model by calibration (ECE@10) | `finbert_base_zero_shot_ProsusAI/finbert` (0.0236) | same |
| best model by Brier | `finbert_agreement_weighted` (0.1713) | same |
| AW vs zero-shot statistical decisiveness | not decisive (paired bootstrap CI [-0.018, +0.023]; McNemar p≈0.56) | [results/2026-05-02/paired_model_comparison_*.csv](../results/2026-05-02/) |
| research gate | **PASS** (all 11 checks) | [results/2026-05-02/research_gate_report_20260502_110923.csv](../results/2026-05-02/research_gate_report_20260502_110923.csv) |

## Status

| area | status |
|---|---|
| experiment evidence | complete |
| statistical validation | complete (Phase 11 script run) |
| publication tables | complete (`paper_assets/tables/`) |
| publication figures | complete (`paper_assets/figures/`) |
| paper draft | skeleton ready (`paper/IEEE_SINGLE_COLUMN_DRAFT.md`); prose pending |
| practical demo | blueprint only (`docs/PHASE11_PRACTICAL_IMPLEMENTATION_BLUEPRINT.md`) |

## What has been achieved

- A leakage-controlled, multi-seed evaluation pipeline for financial sentiment
  with strict research-gate enforcement.
- Agreement-weighted FinBERT trained over five seeds; ranked #1 by macro-F1 and
  ~4× more seed-stable than vanilla fine-tuning.
- Calibration, abstention, and error-analysis evidence collected for the
  FinBERT family.
- Phase 11 statistical validation (per-model bootstrap CIs, paired bootstrap,
  McNemar, Cohen's d) producing
  `results/2026-05-02/statistical_validation_summary_*.csv` and
  `results/2026-05-02/paired_model_comparison_*.csv`.
- Six publication-grade tables and six publication-grade figures ready for
  paper assembly.
- IEEE single-column draft skeleton with placeholder citations.

## What remains before manuscript submission

1. Fill in citations and related-work prose in
   `paper/IEEE_SINGLE_COLUMN_DRAFT.md`.
2. Promote the methodology blueprint into formal paper prose.
3. Promote the results narrative into the results section, lifting verbatim
   numbers from the Phase 11 tables.
4. Optional: add an out-of-domain validation pass (e.g. FiQA headlines) to
   strengthen external validity. This requires GPU; if attempted, **Kaggle
   first**, RunPod only on explicit user confirmation.
5. Final IEEE LaTeX conversion and figure resolution check.

## Exact next steps for paper writing

- **Step A.** Pull verified evidence: tables 1-6 and figures 1-6.
- **Step B.** Draft Sections 1-3 (intro, related work, gap) in
  `paper/IEEE_SINGLE_COLUMN_DRAFT.md`.
- **Step C.** Replace `[CITATION: ...]` placeholders with verified references.
- **Step D.** Convert Markdown to IEEE LaTeX template; embed PNG/PDF figures
  from `paper_assets/figures/`.
- **Step E.** Run a final claim-boundary review against
  `paper_assets/tables/table5_claim_boundary_matrix.csv` before submission.

## Hard rules to keep

- Do not claim universal SOTA.
- Do not claim live-trading or investment-decision validation.
- Do not auto-start RunPod.
- Do not commit secrets, raw datasets, model checkpoints, or large prediction
  dumps.
