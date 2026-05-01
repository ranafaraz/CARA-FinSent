# CARA-FinSent Makefile
#
# Cross-platform (Windows / macOS / Linux). On Windows run from PowerShell or
# Git Bash with your virtual environment already activated (.venv\Scripts\activate
# on PowerShell, source .venv/bin/activate on POSIX).
#
# Override variables per-call:
#   make baseline DATA=data/processed/combined_standardized_20260101_120000.csv
#   make finbert-smoke DATA=$(DATA) MAX_ROWS=500
#   make retrieval-ext DATA=$(DATA) EXT=data/external/financial_news_headlines_*.csv
#
# `make help` lists every target with a one-line description.

PYTHON ?= python
DATA ?= $(shell $(PYTHON) -c "from pathlib import Path; p=Path('data/processed/latest.csv'); m=sorted(Path('data/processed').rglob('combined_standardized_*.csv'),key=lambda f:f.stat().st_mtime,reverse=True) if not p.exists() else []; print(str(p) if p.exists() else (str(m[0]) if m else ''))" 2>/dev/null)
DATA_FLAG := $(if $(DATA),--data $(DATA),)
EXT ?=
MAX_ROWS ?=
SEED ?= 42
EPOCHS ?= 3
BATCH_SIZE ?= 8
MODEL_NAME ?= ProsusAI/finbert
DATA_RAW ?=
RESULTS_DIR ?= results
FIGURES_DIR ?= figures

MAX_ROWS_FLAG := $(if $(MAX_ROWS),--max_rows $(MAX_ROWS),)
EXT_FLAG := $(if $(EXT),--external_corpus_csv $(EXT),)
SEED_FLAG := --seed $(SEED)

.PHONY: help setup setup-gpu test clean-pyc \
        prepare prepare-local \
        collect-sec collect-news \
        baseline baseline-smoke \
        finbert finbert-smoke \
        structured retrieval retrieval-ext \
        agreement \
        calibration calibration-iso \
        cara cara-ext cara-ablation \
        pipeline pipeline-full \
		smoke-all figures-summary pipeline pipeline-full

help:  ## Show this help.
	@$(PYTHON) -c "import re,sys; \
text=open('Makefile').read(); \
print('Available make targets:\n'); \
[print(f'  {m.group(1):<22} {m.group(2)}') for m in re.finditer(r'^([a-zA-Z][a-zA-Z0-9_-]+):.*?##\s*(.*)$$', text, re.M)]"

setup:  ## Install CPU requirements into the active environment.
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -r requirements.txt

setup-gpu:  ## Install GPU requirements (run AFTER installing the right torch+CUDA wheel).
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -r requirements-gpu.txt

test:  ## Compile-check + run pytest.
	$(PYTHON) -m compileall -q src scripts
	$(PYTHON) -m pytest -q tests

clean-pyc:  ## Remove __pycache__ folders.
	$(PYTHON) -c "import pathlib,shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"

# ---------- Data preparation ----------

prepare:  ## Download + standardize Financial PhraseBank and FiQA from Hugging Face.
	$(PYTHON) scripts/00_prepare_phrasebank_fiqa.py

prepare-local:  ## Standardize from a local CSV (set DATA_RAW=path/to.csv).
	@if [ -z "$(DATA_RAW)" ]; then echo "ERROR: set DATA_RAW=path/to/your.csv"; exit 1; fi
	$(PYTHON) scripts/00_prepare_phrasebank_fiqa.py --local_csv $(DATA_RAW) --skip_hf

# ---------- External corpora ----------

collect-sec:  ## Collect SEC 10-K filings (requires SEC_USER_AGENT env var).
	$(PYTHON) scripts/02_collect_sec_10k.py --tickers AAPL MSFT NVDA TSLA --years 3 --max_filings_per_ticker 2

collect-news:  ## Collect financial news headlines via RSS (+ optional NewsAPI / Finnhub).
	$(PYTHON) scripts/03_collect_financial_news.py --weak_label

# ---------- Experiments (require DATA=...) ----------

baseline:  ## Classical TF-IDF baselines (script 10). Data auto-detected from data/processed/latest.csv.
	$(PYTHON) scripts/10_run_classical_baselines.py $(DATA_FLAG) $(SEED_FLAG) $(MAX_ROWS_FLAG)

baseline-smoke:  ## 500-row smoke run of classical baselines.
	$(PYTHON) scripts/10_run_classical_baselines.py $(DATA_FLAG) $(SEED_FLAG) --max_rows 500

finbert:  ## FinBERT fine-tune (GPU recommended).
	$(PYTHON) scripts/11_run_finbert_baseline.py $(DATA_FLAG) --model_name $(MODEL_NAME) --epochs $(EPOCHS) --batch_size $(BATCH_SIZE) $(SEED_FLAG) $(MAX_ROWS_FLAG)

finbert-smoke:  ## 1k-row 1-epoch FinBERT smoke run.
	$(PYTHON) scripts/11_run_finbert_baseline.py $(DATA_FLAG) --model_name $(MODEL_NAME) --epochs 1 --batch_size $(BATCH_SIZE) $(SEED_FLAG) --max_rows 1000

structured:  ## TF-IDF vs TF-IDF + structured-features ablation (script 12).
	$(PYTHON) scripts/12_run_structured_features_experiment.py $(DATA_FLAG) $(SEED_FLAG) $(MAX_ROWS_FLAG)

retrieval:  ## Retrieval ablation using only training corpus (script 13).
	$(PYTHON) scripts/13_run_retrieval_experiment.py $(DATA_FLAG) $(SEED_FLAG) $(MAX_ROWS_FLAG)

retrieval-ext:  ## Retrieval ablation with external corpus (set EXT=path/to.csv).
	@if [ -z "$(EXT)" ]; then echo "ERROR: set EXT=data/external/financial_news_headlines_*.csv"; exit 1; fi
	$(PYTHON) scripts/13_run_retrieval_experiment.py $(DATA_FLAG) --external_corpus_csv $(EXT) $(SEED_FLAG) $(MAX_ROWS_FLAG)

agreement:  ## Agreement-aware training (PhraseBank only; script 14).
	$(PYTHON) scripts/14_run_agreement_aware_experiment.py $(DATA_FLAG) $(SEED_FLAG) $(MAX_ROWS_FLAG)

calibration:  ## Sigmoid calibration + abstention curves (script 15).
	$(PYTHON) scripts/15_run_calibration_experiment.py $(DATA_FLAG) --method sigmoid $(SEED_FLAG) $(MAX_ROWS_FLAG)

calibration-iso:  ## Isotonic calibration variant.
	$(PYTHON) scripts/15_run_calibration_experiment.py $(DATA_FLAG) --method isotonic $(SEED_FLAG) $(MAX_ROWS_FLAG)

cara:  ## CARA-lite integrated pipeline (retrieval + structured + agreement + calibration; script 16).
	$(PYTHON) scripts/16_run_full_cara_lite_experiment.py $(DATA_FLAG) $(SEED_FLAG) $(MAX_ROWS_FLAG)

cara-ext:  ## CARA-lite with external retrieval corpus.
	@if [ -z "$(EXT)" ]; then echo "ERROR: set EXT=data/external/financial_news_headlines_*.csv"; exit 1; fi
	$(PYTHON) scripts/16_run_full_cara_lite_experiment.py $(DATA_FLAG) --external_corpus_csv $(EXT) $(SEED_FLAG) $(MAX_ROWS_FLAG)

cara-ablation:  ## Run CARA-lite four times toggling each component off (full ablation table).
	$(PYTHON) scripts/16_run_full_cara_lite_experiment.py $(DATA_FLAG) $(SEED_FLAG) $(MAX_ROWS_FLAG)
	$(PYTHON) scripts/16_run_full_cara_lite_experiment.py $(DATA_FLAG) $(SEED_FLAG) $(MAX_ROWS_FLAG) --no_retrieval
	$(PYTHON) scripts/16_run_full_cara_lite_experiment.py $(DATA_FLAG) $(SEED_FLAG) $(MAX_ROWS_FLAG) --no_structured
	$(PYTHON) scripts/16_run_full_cara_lite_experiment.py $(DATA_FLAG) $(SEED_FLAG) $(MAX_ROWS_FLAG) --no_agreement
	$(PYTHON) scripts/16_run_full_cara_lite_experiment.py $(DATA_FLAG) $(SEED_FLAG) $(MAX_ROWS_FLAG) --no_calibration

pipeline:  ## Orchestrator: classical + structured + retrieval + calibration + CARA-lite (script 90).
	$(PYTHON) scripts/90_run_all_classical_pipeline.py $(DATA_FLAG) $(MAX_ROWS_FLAG) $(if $(EXT),--external_corpus_csv $(EXT),)

pipeline-full:  ## Pipeline + FinBERT + agreement-aware (slower).
	$(PYTHON) scripts/90_run_all_classical_pipeline.py $(DATA_FLAG) $(MAX_ROWS_FLAG) --with_finbert --with_agreement $(if $(EXT),--external_corpus_csv $(EXT),)

smoke-all:  ## Run every smoke variant on a 500-row sample for end-to-end verification.
	$(MAKE) baseline-smoke SEED=$(SEED)
	$(PYTHON) scripts/12_run_structured_features_experiment.py $(DATA_FLAG) $(SEED_FLAG) --max_rows 500
	$(PYTHON) scripts/13_run_retrieval_experiment.py $(DATA_FLAG) $(SEED_FLAG) --max_rows 500
	$(PYTHON) scripts/15_run_calibration_experiment.py $(DATA_FLAG) --method sigmoid $(SEED_FLAG) --max_rows 500
	$(PYTHON) scripts/16_run_full_cara_lite_experiment.py $(DATA_FLAG) $(SEED_FLAG) --max_rows 500

figures-summary:  ## Print the latest *_summary_*.csv files in results/.
	$(PYTHON) -c "from pathlib import Path; \
[print(p) for p in sorted(Path('$(RESULTS_DIR)').rglob('*_summary_*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)]"
