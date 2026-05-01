# CARA-FinSent Colab notebooks

One notebook per script, plus a results explorer. Designed for [Google Colab](https://colab.research.google.com/) but works in any Jupyter runtime.

## How to use

1. Push this repo to a GitHub URL you can read from Colab (public repo or with a token).
2. Open a notebook below in Colab.
3. In **Cell 1** set:
   - `USE_DRIVE = True` (default) to persist `data/processed/`, `results/`, `figures/` under `/content/drive/MyDrive/CARA-FinSent/`.
   - `REPO_URL = "https://github.com/<you>/cara-finsent-experiments.git"` the first time. The notebook reuses an existing clone on subsequent runs.
4. Run all cells. Each notebook ends with a `cara_results_<ts>.zip` download.

For FinBERT / CARA-lite / full pipeline notebooks, switch the runtime to **GPU** (`Runtime → Change runtime type → GPU`).

## Notebook map

| Notebook | Backed script | GPU |
|---|---|---|
| [00_prepare_phrasebank_fiqa.ipynb](00_prepare_phrasebank_fiqa.ipynb) | [scripts/00_prepare_phrasebank_fiqa.py](../../scripts/00_prepare_phrasebank_fiqa.py) | — |
| [01_collect_stocktwits.ipynb](01_collect_stocktwits.ipynb) | [scripts/01_collect_stocktwits.py](../../scripts/01_collect_stocktwits.py) | — |
| [02_collect_sec_10k.ipynb](02_collect_sec_10k.ipynb) | [scripts/02_collect_sec_10k.py](../../scripts/02_collect_sec_10k.py) | — |
| [03_collect_financial_news.ipynb](03_collect_financial_news.ipynb) | [scripts/03_collect_financial_news.py](../../scripts/03_collect_financial_news.py) | — |
| [10_classical_baselines.ipynb](10_classical_baselines.ipynb) | [scripts/10_run_classical_baselines.py](../../scripts/10_run_classical_baselines.py) | — |
| [11_finbert_baseline.ipynb](11_finbert_baseline.ipynb) | [scripts/11_run_finbert_baseline.py](../../scripts/11_run_finbert_baseline.py) | ✅ |
| [12_structured_features.ipynb](12_structured_features.ipynb) | [scripts/12_run_structured_features_experiment.py](../../scripts/12_run_structured_features_experiment.py) | — |
| [13_retrieval.ipynb](13_retrieval.ipynb) | [scripts/13_run_retrieval_experiment.py](../../scripts/13_run_retrieval_experiment.py) | — |
| [14_agreement_aware.ipynb](14_agreement_aware.ipynb) | [scripts/14_run_agreement_aware_experiment.py](../../scripts/14_run_agreement_aware_experiment.py) | — |
| [15_calibration.ipynb](15_calibration.ipynb) | [scripts/15_run_calibration_experiment.py](../../scripts/15_run_calibration_experiment.py) | — |
| [16_cara_lite.ipynb](16_cara_lite.ipynb) | [scripts/16_run_full_cara_lite_experiment.py](../../scripts/16_run_full_cara_lite_experiment.py) | optional |
| [90_full_pipeline.ipynb](90_full_pipeline.ipynb) | [scripts/90_run_all_classical_pipeline.py](../../scripts/90_run_all_classical_pipeline.py) | recommended |
| [99_results_explorer.ipynb](99_results_explorer.ipynb) | — | — |

The transformer dropdown in `11_finbert_baseline.ipynb` ships three pre-configured choices: `ProsusAI/finbert`, `yiyanghkust/finbert-tone`, `nlptown/bert-base-multilingual-uncased-sentiment` (free-text override allowed).

## Regenerating notebooks

If you change a script's CLI flags, re-run:

```bash
python notebooks/colab/_generate_colab_notebooks.py
```

This script is the single source of truth for the notebook layout.
