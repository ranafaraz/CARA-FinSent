# AGENTS.md - Instructions for AI Coding Agents

## Mission

Help run and extend the CARA-FinSent experimental pipeline. The target research contribution is not a slightly better classifier; it is a decision-grade financial sentiment system with retrieval, structured financial signals, agreement-aware training, calibration, and abstention.

## Non-negotiable rules

1. Do not overwrite existing result files. All outputs must include a timestamp.
2. Do not commit API keys, SEC user agents with private emails, `.env`, downloaded raw datasets, model checkpoints, or large result folders unless explicitly requested.
3. Keep labels normalized to exactly: `negative`, `neutral`, `positive`.
4. Every experiment must save a summary CSV in `results/`.
5. Every experiment should save prediction-level CSVs when possible.
6. Prefer reproducibility over cleverness: fixed random seeds, explicit CLI arguments, clear manifests.
7. Do not report SOTA claims unless the results folder contains the evidence.
8. Treat StockTwits, SEC 10-K, and news weak labels as noisy unless manually verified.

## Recommended agent workflow

1. Run syntax check:

```bash
python -m compileall -q src scripts
```

2. Prepare a small smoke-test dataset:

```bash
python scripts/00_prepare_phrasebank_fiqa.py --skip_fiqa
```

3. Run small smoke experiments:

```bash
python scripts/10_run_classical_baselines.py --data <prepared_csv> --max_rows 500
python scripts/16_run_full_cara_lite_experiment.py --data <prepared_csv> --max_rows 500
```

4. If smoke tests pass, run full experiments.

5. Summarize outputs using the latest timestamped `*_summary_*.csv` files.

## Good pull request structure

- One PR for dataset collection improvements.
- One PR for model/experiment changes.
- One PR for plotting/reporting changes.

## Research reporting expectations

A useful results report should compare:

- Classical baselines vs FinBERT.
- FinBERT/classical without retrieval vs with retrieval.
- TF-IDF only vs TF-IDF + structured financial features.
- Unweighted vs agreement-weighted training.
- Uncalibrated vs calibrated models.
- Accuracy/F1 vs calibration/abstention/latency.

## Failure handling

If a script fails due to missing API keys or unavailable data provider, do not fake data. Record the failure and continue with available datasets.
