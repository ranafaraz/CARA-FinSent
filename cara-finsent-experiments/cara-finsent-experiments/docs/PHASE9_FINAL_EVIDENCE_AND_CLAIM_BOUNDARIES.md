# Phase 9 Final Evidence And Claim Boundaries

Date: 2026-05-02
Scope: PhraseBank in-domain benchmark, 5-seed research-grade evidence

## 1) Final 5-Seed Leaderboard Snapshot

Source: results/2026-05-02/final_leaderboard_mean_std_20260502_110922.csv

Top model families by mean macro-F1:
- agreement_weighted: macro-F1 0.8863 (+/- 0.0023), accuracy 0.8886 (+/- 0.0020)
- finbert_zero_shot: macro-F1 0.8836 (+/- 0.0000), accuracy 0.8832 (+/- 0.0000)
- finbert_finetuned: macro-F1 0.8821 (+/- 0.0092), accuracy 0.8859 (+/- 0.0073)

Best classical baseline:
- tfidf_linear_svm: macro-F1 0.6941, accuracy 0.7497

## 2) Agreement-Weighted 5-Seed Completion Evidence

Source: results/2026-05-02/seed_sweep_summary_20260502_105326.csv

Unique completed seeds for agreement_weighted:
- 13, 21, 42, 87, 101

Per-seed macro-F1:
- seed 13: 0.8860
- seed 21: 0.8901
- seed 42: 0.8859
- seed 87: 0.8853
- seed 101: 0.8841

## 3) Research Gate Final Status

Source: results/2026-05-02/research_gate_report_20260502_110923.csv

Overall: PASS

Critical seed checks:
- min_seed_runs: PASS (classical=6; finbert_zero_shot=5; finbert_finetuned=5; agreement_weighted=5)
- agreement_weighted_seed_runs: PASS (unique_seeds=5; required=5)

Quality and leakage checks:
- no_text_hash_leakage: PASS
- controlled_gold_splits: PASS
- audit_gate: PASS

## 4) Reliability-Oriented Comparison (What Evidence Supports)

From current summarized evidence:
- agreement_weighted shows the best mean macro-F1 among evaluated model families.
- agreement_weighted also improves mean ECE versus finbert_finetuned (0.0495 vs 0.0608).
- finbert_zero_shot remains strongest on mean ECE (0.0236) among FinBERT variants.

Interpretation:
- If the objective emphasizes balanced classification quality (macro-F1), agreement_weighted is currently preferred.
- If calibration strictness is primary, finbert_zero_shot remains a strong baseline.

## 5) Recommended Publication Model

Primary recommendation:
- agreement_weighted as the main reported model for PhraseBank in-domain performance.

Secondary recommendation:
- report finbert_zero_shot as calibration-oriented comparator.
- report finbert_finetuned as supervised baseline comparator.

## 6) Claims Allowed

The following claims are supported by the evidence above:
- Agreement-weighted FinBERT achieved complete 5-seed evidence on PhraseBank (seeds 13, 21, 42, 87, 101).
- Agreement-weighted FinBERT is the best model family in this run on mean macro-F1.
- Research gate passed with explicit agreement-weighted seed enforcement.
- No text-hash leakage was detected in audited outputs.

## 7) Claims Not Allowed

The following claims are not supported by current evidence:
- Generalization claims to other domains/datasets beyond the evaluated splits.
- Causal claims that agreement weighting alone guarantees calibration superiority over zero-shot.
- Claims of universal superiority across all downstream financial sentiment tasks.
- Production-readiness claims without external validation, drift analysis, and deployment constraints.

## 8) Cost Discipline Confirmation

RunPod pod w8d93um6ain3fq was stopped after evidence collection and verified as EXITED.
