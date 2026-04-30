.PHONY: setup test prepare baseline structured retrieval calibration cara pipeline

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

test:
	python -m compileall -q src scripts

prepare:
	python scripts/00_prepare_phrasebank_fiqa.py

baseline:
	python scripts/10_run_classical_baselines.py --data $(DATA)

structured:
	python scripts/12_run_structured_features_experiment.py --data $(DATA)

retrieval:
	python scripts/13_run_retrieval_experiment.py --data $(DATA)

calibration:
	python scripts/15_run_calibration_experiment.py --data $(DATA)

cara:
	python scripts/16_run_full_cara_lite_experiment.py --data $(DATA)

pipeline:
	python scripts/90_run_all_classical_pipeline.py --data $(DATA)
