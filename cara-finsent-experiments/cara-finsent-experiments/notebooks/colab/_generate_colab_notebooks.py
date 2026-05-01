"""Generate Colab notebooks for every CARA-FinSent script.

This script is executed once at repo-prep time (or by maintainers when
scripts/CLI flags change).  Run from the repo root:

    python notebooks/colab/_generate_colab_notebooks.py

Each notebook follows the same template:
  1. Title + intro markdown.
  2. Environment check (GPU for FinBERT/CARA only).
  3. Repo clone (auto-detects current ``origin``) or %cd into mounted Drive.
  4. ``pip install -r requirements.txt`` (and ``requirements-gpu.txt`` for FinBERT).
  5. Optional Drive mount with ``USE_DRIVE`` toggle (default True).
  6. Parameter form mirroring the script CLI.
  7. ``%run`` the script.
  8. Load latest timestamped output, render head + a key chart inline.
  9. Zip ``results/`` and ``figures/`` for download.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent


def md(*lines: str) -> dict:
    return {'cell_type': 'markdown', 'metadata': {}, 'source': [l if l.endswith('\n') else l + '\n' for l in lines]}


def code(*lines: str) -> dict:
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': [l if l.endswith('\n') else l + '\n' for l in lines]}


NB_META = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
    'colab': {'provenance': [], 'toc_visible': True},
    'accelerator': 'GPU',
}


def make_notebook(cells: list[dict], gpu: bool = False) -> dict:
    meta = dict(NB_META)
    meta['accelerator'] = 'GPU' if gpu else 'None'
    return {'nbformat': 4, 'nbformat_minor': 5, 'metadata': meta, 'cells': cells}


# ---------- Reusable cell blocks ----------

def cell_setup(needs_gpu: bool, gpu_extras: bool = False) -> list[dict]:
    cells: list[dict] = []
    if needs_gpu:
        cells.append(md('## 0. Runtime check (GPU recommended)\n', 'Runtime → Change runtime type → **GPU** (T4 is free).'))
        cells.append(code('!nvidia-smi || echo "No GPU detected — script will fall back to CPU."'))
    cells.append(md('## 1. Mount Drive & clone repo\n',
                    'Set `USE_DRIVE=True` to persist `data/`, `results/`, `figures/` across Colab runtime resets. ',
                    'Set `REPO_URL` to override the auto-detected git remote.'))
    cells.append(code(
        'USE_DRIVE = True  #@param {type:"boolean"}\n',
        'DRIVE_DIR = "/content/drive/MyDrive/CARA-FinSent"  #@param {type:"string"}\n',
        'REPO_URL = ""  #@param {type:"string"}  # leave blank to auto-detect from this notebook\'s repo\n',
        '\n',
        'import os, subprocess\n',
        'from pathlib import Path\n',
        '\n',
        'if USE_DRIVE:\n',
        '    from google.colab import drive\n',
        '    drive.mount("/content/drive", force_remount=False)\n',
        '    Path(DRIVE_DIR).mkdir(parents=True, exist_ok=True)\n',
        '    WORK_DIR = Path(DRIVE_DIR)\n',
        'else:\n',
        '    WORK_DIR = Path("/content")\n',
        '\n',
        'REPO_DIR = WORK_DIR / "cara-finsent-experiments"\n',
        'if not REPO_DIR.exists():\n',
        '    if not REPO_URL:\n',
        '        # Best-effort auto-detect: try the repo this notebook lives in (works when notebook is opened from GitHub).\n',
        '        REPO_URL = os.environ.get("REPO_URL", "")\n',
        '    if not REPO_URL:\n',
        '        raise RuntimeError("Set REPO_URL above (e.g. https://github.com/<user>/cara-finsent-experiments.git)")\n',
        '    subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)\n',
        '\n',
        'os.chdir(REPO_DIR)\n',
        'print("Working in:", os.getcwd())\n',
    ))
    install_lines = ['!pip -q install -r requirements.txt\n']
    if gpu_extras:
        install_lines.append('# requirements-gpu.txt is a thin layer over requirements.txt; torch is already installed in Colab.\n')
    cells.append(md('## 2. Install dependencies'))
    cells.append(code(*install_lines))
    return cells


def cell_zip_results() -> list[dict]:
    return [
        md('## ⤓ Download results\n', 'Zip `results/` + `figures/` for sharing.'),
        code(
            'import shutil, datetime\n',
            'ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")\n',
            'archive = shutil.make_archive(f"cara_results_{ts}", "zip", root_dir=".", base_dir="results")\n',
            'print("Created:", archive)\n',
            'try:\n',
            '    from google.colab import files\n',
            '    files.download(archive)\n',
            'except Exception:\n',
            '    pass\n',
        ),
    ]


def cell_show_latest_summary(prefix: str) -> list[dict]:
    return [
        md(f'## 4. Inspect latest `{prefix}_summary_*.csv`'),
        code(
            'import pandas as pd\n',
            'from pathlib import Path\n',
            f'matches = sorted(Path("results").rglob("{prefix}_summary_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)\n',
            'assert matches, "No summary CSV found — did the script run?"\n',
            'latest = matches[0]\n',
            'print("Latest:", latest)\n',
            'df = pd.read_csv(latest)\n',
            'df\n',
        ),
    ]


# ---------- Per-notebook definitions ----------

NB_SPECS = [
    {
        'filename': '00_prepare_phrasebank_fiqa.ipynb',
        'title': 'Prepare Financial PhraseBank + FiQA',
        'gpu': False,
        'script': 'scripts/00_prepare_phrasebank_fiqa.py',
        'summary_prefix': 'dataset_inventory',
        'params_code': [
            'SKIP_HF = False  #@param {type:"boolean"}\n',
            'SKIP_FIQA = False  #@param {type:"boolean"}\n',
            'LOCAL_CSV = ""  #@param {type:"string"}\n',
            'argv = []\n',
            'if SKIP_HF: argv.append("--skip_hf")\n',
            'if SKIP_FIQA: argv.append("--skip_fiqa")\n',
            'if LOCAL_CSV: argv += ["--local_csv", LOCAL_CSV]\n',
        ],
    },
    {
        'filename': '01_collect_stocktwits.ipynb',
        'title': 'Collect StockTwits messages',
        'gpu': False,
        'script': 'scripts/01_collect_stocktwits.py',
        'summary_prefix': 'stocktwits_messages',
        'params_code': [
            'SYMBOLS = "AAPL MSFT TSLA NVDA"  #@param {type:"string"}\n',
            'LIMIT = 30  #@param {type:"integer"}\n',
            'import os\n',
            'STOCKTWITS_TOKEN = ""  #@param {type:"string"}\n',
            'if STOCKTWITS_TOKEN: os.environ["STOCKTWITS_ACCESS_TOKEN"] = STOCKTWITS_TOKEN\n',
            'argv = ["--symbols", *SYMBOLS.split(), "--limit_per_symbol", str(LIMIT)]\n',
        ],
        'no_summary': True,
    },
    {
        'filename': '02_collect_sec_10k.ipynb',
        'title': 'Collect SEC 10-K filings',
        'gpu': False,
        'script': 'scripts/02_collect_sec_10k.py',
        'summary_prefix': 'sec_10k_weak_sentiment',
        'params_code': [
            'TICKERS = "AAPL MSFT NVDA TSLA"  #@param {type:"string"}\n',
            'YEARS = 3  #@param {type:"integer"}\n',
            'MAX_FILINGS = 2  #@param {type:"integer"}\n',
            'import os\n',
            'SEC_USER_AGENT = ""  #@param {type:"string"}  # REQUIRED: e.g. "Your Name your@email.com"\n',
            'assert "@" in SEC_USER_AGENT, "Set SEC_USER_AGENT to a real name + email (SEC requires it)"\n',
            'os.environ["SEC_USER_AGENT"] = SEC_USER_AGENT\n',
            'argv = ["--tickers", *TICKERS.split(), "--years", str(YEARS), "--max_filings_per_ticker", str(MAX_FILINGS)]\n',
        ],
        'no_summary': True,
    },
    {
        'filename': '03_collect_financial_news.ipynb',
        'title': 'Collect financial news headlines',
        'gpu': False,
        'script': 'scripts/03_collect_financial_news.py',
        'summary_prefix': 'financial_news_headlines',
        'params_code': [
            'WEAK_LABEL = True  #@param {type:"boolean"}\n',
            'INCLUDE_NEWSAPI = False  #@param {type:"boolean"}\n',
            'INCLUDE_FINNHUB = False  #@param {type:"boolean"}\n',
            'import os, getpass\n',
            'if INCLUDE_NEWSAPI and not os.environ.get("NEWSAPI_KEY"):\n',
            '    os.environ["NEWSAPI_KEY"] = getpass.getpass("NEWSAPI_KEY: ")\n',
            'if INCLUDE_FINNHUB and not os.environ.get("FINNHUB_API_KEY"):\n',
            '    os.environ["FINNHUB_API_KEY"] = getpass.getpass("FINNHUB_API_KEY: ")\n',
            'argv = []\n',
            'if WEAK_LABEL: argv.append("--weak_label")\n',
            'if INCLUDE_NEWSAPI: argv.append("--include_newsapi")\n',
            'if INCLUDE_FINNHUB: argv.append("--include_finnhub")\n',
        ],
        'no_summary': True,
    },
    {
        'filename': '10_classical_baselines.ipynb',
        'title': 'Classical TF-IDF baselines',
        'gpu': False,
        'script': 'scripts/10_run_classical_baselines.py',
        'summary_prefix': 'classical_baseline',
        'needs_data': True,
        'params_code': [
            'MAX_ROWS = 0  #@param {type:"integer"}  # 0 = use all rows\n',
            'SEED = 42  #@param {type:"integer"}\n',
            'NO_XGBOOST = False  #@param {type:"boolean"}\n',
            'argv = ["--data", DATA, "--seed", str(SEED)]\n',
            'if MAX_ROWS: argv += ["--max_rows", str(MAX_ROWS)]\n',
            'if NO_XGBOOST: argv.append("--no_xgboost")\n',
        ],
    },
    {
        'filename': '11_finbert_baseline.ipynb',
        'title': 'FinBERT (and other transformer) baseline',
        'gpu': True,
        'gpu_extras': True,
        'script': 'scripts/11_run_finbert_baseline.py',
        'summary_prefix': 'finbert_baseline',
        'needs_data': True,
        'params_code': [
            'MODEL_NAME = "ProsusAI/finbert"  #@param ["ProsusAI/finbert", "yiyanghkust/finbert-tone", "nlptown/bert-base-multilingual-uncased-sentiment"] {allow-input: true}\n',
            'EPOCHS = 3  #@param {type:"number"}\n',
            'BATCH_SIZE = 16  #@param {type:"integer"}\n',
            'MAX_ROWS = 0  #@param {type:"integer"}\n',
            'SEED = 42  #@param {type:"integer"}\n',
            'argv = ["--data", DATA, "--model_name", MODEL_NAME, "--epochs", str(EPOCHS),\n',
            '        "--batch_size", str(BATCH_SIZE), "--seed", str(SEED)]\n',
            'if MAX_ROWS: argv += ["--max_rows", str(MAX_ROWS)]\n',
        ],
    },
    {
        'filename': '12_structured_features.ipynb',
        'title': 'Structured-feature ablation',
        'gpu': False,
        'script': 'scripts/12_run_structured_features_experiment.py',
        'summary_prefix': 'structured_features',
        'needs_data': True,
        'params_code': [
            'MAX_ROWS = 0  #@param {type:"integer"}\n',
            'SEED = 42  #@param {type:"integer"}\n',
            'argv = ["--data", DATA, "--seed", str(SEED)]\n',
            'if MAX_ROWS: argv += ["--max_rows", str(MAX_ROWS)]\n',
        ],
    },
    {
        'filename': '13_retrieval.ipynb',
        'title': 'Retrieval-augmented classification',
        'gpu': False,
        'script': 'scripts/13_run_retrieval_experiment.py',
        'summary_prefix': 'retrieval_experiment',
        'needs_data': True,
        'params_code': [
            'TOP_K = 3  #@param {type:"integer"}\n',
            'EXTERNAL_CORPUS = ""  #@param {type:"string"}\n',
            'MAX_ROWS = 0  #@param {type:"integer"}\n',
            'SEED = 42  #@param {type:"integer"}\n',
            'argv = ["--data", DATA, "--top_k", str(TOP_K), "--seed", str(SEED)]\n',
            'if EXTERNAL_CORPUS: argv += ["--external_corpus_csv", EXTERNAL_CORPUS]\n',
            'if MAX_ROWS: argv += ["--max_rows", str(MAX_ROWS)]\n',
        ],
    },
    {
        'filename': '14_agreement_aware.ipynb',
        'title': 'Agreement-aware training (PhraseBank)',
        'gpu': False,
        'script': 'scripts/14_run_agreement_aware_experiment.py',
        'summary_prefix': 'agreement_aware',
        'needs_data': True,
        'data_pattern': 'phrasebank_standardized_*.csv',
        'params_code': [
            'MIN_WEIGHT = 0.4  #@param {type:"number"}\n',
            'WEIGHT_SCHEDULE = "linear"  #@param ["linear","quadratic"]\n',
            'MAX_ROWS = 0  #@param {type:"integer"}\n',
            'SEED = 42  #@param {type:"integer"}\n',
            'argv = ["--data", DATA, "--min_weight", str(MIN_WEIGHT),\n',
            '        "--weight_schedule", WEIGHT_SCHEDULE, "--seed", str(SEED)]\n',
            'if MAX_ROWS: argv += ["--max_rows", str(MAX_ROWS)]\n',
        ],
    },
    {
        'filename': '15_calibration.ipynb',
        'title': 'Calibration + abstention',
        'gpu': False,
        'script': 'scripts/15_run_calibration_experiment.py',
        'summary_prefix': 'calibration',
        'needs_data': True,
        'params_code': [
            'BASE_MODEL = "linear_svm"  #@param ["linear_svm","logistic_regression"]\n',
            'METHOD = "sigmoid"  #@param ["sigmoid","isotonic"]\n',
            'N_BINS = 10  #@param {type:"integer"}\n',
            'MAX_ROWS = 0  #@param {type:"integer"}\n',
            'SEED = 42  #@param {type:"integer"}\n',
            'argv = ["--data", DATA, "--base_model", BASE_MODEL, "--method", METHOD,\n',
            '        "--n_bins", str(N_BINS), "--seed", str(SEED)]\n',
            'if MAX_ROWS: argv += ["--max_rows", str(MAX_ROWS)]\n',
        ],
    },
    {
        'filename': '16_cara_lite.ipynb',
        'title': 'CARA-lite integrated pipeline',
        'gpu': False,
        'script': 'scripts/16_run_full_cara_lite_experiment.py',
        'summary_prefix': 'cara_lite',
        'needs_data': True,
        'params_code': [
            'BASE_MODEL = "logreg"  #@param ["logreg","svm"]\n',
            'TOP_K = 3  #@param {type:"integer"}\n',
            'EXTERNAL_CORPUS = ""  #@param {type:"string"}\n',
            'NO_RETRIEVAL = False  #@param {type:"boolean"}\n',
            'NO_STRUCTURED = False  #@param {type:"boolean"}\n',
            'NO_AGREEMENT = False  #@param {type:"boolean"}\n',
            'NO_CALIBRATION = False  #@param {type:"boolean"}\n',
            'MAX_ROWS = 0  #@param {type:"integer"}\n',
            'SEED = 42  #@param {type:"integer"}\n',
            'argv = ["--data", DATA, "--base_model", BASE_MODEL, "--top_k", str(TOP_K), "--seed", str(SEED)]\n',
            'if EXTERNAL_CORPUS: argv += ["--external_corpus_csv", EXTERNAL_CORPUS]\n',
            'for flag, on in [("--no_retrieval", NO_RETRIEVAL), ("--no_structured", NO_STRUCTURED),\n',
            '                  ("--no_agreement", NO_AGREEMENT), ("--no_calibration", NO_CALIBRATION)]:\n',
            '    if on: argv.append(flag)\n',
            'if MAX_ROWS: argv += ["--max_rows", str(MAX_ROWS)]\n',
        ],
    },
    {
        'filename': '90_full_pipeline.ipynb',
        'title': 'Orchestrator pipeline (scripts 10/12/13/15/16, optionally + 11/14)',
        'gpu': True,
        'gpu_extras': True,
        'script': 'scripts/90_run_all_classical_pipeline.py',
        'summary_prefix': 'cara_lite',
        'needs_data': True,
        'params_code': [
            'EXTERNAL_CORPUS = ""  #@param {type:"string"}\n',
            'WITH_FINBERT = False  #@param {type:"boolean"}\n',
            'WITH_AGREEMENT = False  #@param {type:"boolean"}\n',
            'KEEP_GOING = True  #@param {type:"boolean"}\n',
            'MAX_ROWS = 0  #@param {type:"integer"}\n',
            'SEED = 42  #@param {type:"integer"}\n',
            'argv = ["--data", DATA, "--seed", str(SEED)]\n',
            'if EXTERNAL_CORPUS: argv += ["--external_corpus_csv", EXTERNAL_CORPUS]\n',
            'if WITH_FINBERT: argv.append("--with_finbert")\n',
            'if WITH_AGREEMENT: argv.append("--with_agreement")\n',
            'if KEEP_GOING: argv.append("--keep_going")\n',
            'if MAX_ROWS: argv += ["--max_rows", str(MAX_ROWS)]\n',
        ],
    },
]


def build_data_resolver_cell(data_pattern: str = 'combined_standardized_*.csv') -> dict:
    return code(
        '## Resolve the latest standardized CSV (run notebook 00 first if missing)\n',
        'from pathlib import Path\n',
        f'matches = sorted(Path("data/processed").glob("{data_pattern}"), key=lambda p: p.stat().st_mtime, reverse=True)\n',
        'assert matches, "No standardized CSV found in data/processed/. Run 00_prepare_phrasebank_fiqa.ipynb first."\n',
        'DATA = str(matches[0])\n',
        'print("DATA =", DATA)\n',
    )


def build_notebook(spec: dict) -> dict:
    cells: list[dict] = [
        md(f'# {spec["title"]}\n', f'Runs `{spec["script"]}` end-to-end on Google Colab.'),
        *cell_setup(spec.get('gpu', False), spec.get('gpu_extras', False)),
    ]
    if spec.get('needs_data'):
        cells.append(md('## 3a. Pick the input dataset'))
        cells.append(build_data_resolver_cell(spec.get('data_pattern', 'combined_standardized_*.csv')))
    cells.append(md('## 3b. Configure parameters'))
    cells.append(code(*spec['params_code']))
    cells.append(md(f'## 3c. Run `{Path(spec["script"]).name}`'))
    cells.append(code(
        'import sys, runpy\n',
        'sys.argv = [' + repr(spec['script']) + '] + argv\n',
        'print("Running:", " ".join(sys.argv))\n',
        f'runpy.run_path("{spec["script"]}", run_name="__main__")\n',
    ))
    if not spec.get('no_summary'):
        cells.extend(cell_show_latest_summary(spec['summary_prefix']))
    cells.extend(cell_zip_results())
    return make_notebook(cells, gpu=spec.get('gpu', False))


def build_explorer_notebook() -> dict:
    cells: list[dict] = [
        md('# CARA-FinSent results explorer\n',
           'Loads every `*_summary_*.csv` and renders a side-by-side comparison.'),
        *cell_setup(needs_gpu=False),
        md('## 3. Aggregate latest summaries'),
        code(
            'import pandas as pd\n',
            'from pathlib import Path\n',
            'rows = []\n',
            'for path in sorted(Path("results").rglob("*_summary_*.csv")):\n',
            '    try:\n',
            '        df = pd.read_csv(path)\n',
            '        df["_source_file"] = path.name\n',
            '        rows.append(df)\n',
            '    except Exception as e:\n',
            '        print("Skip", path, e)\n',
            'if not rows:\n',
            '    raise SystemExit("No *_summary_*.csv found in results/. Run some experiments first.")\n',
            'all_summary = pd.concat(rows, ignore_index=True, sort=False)\n',
            'cols_first = [c for c in ["model","accuracy","macro_f1","weighted_f1","mcc","ece_10_bins","brier_score","ablation_flags","seed","_source_file"] if c in all_summary.columns]\n',
            'all_summary = all_summary[cols_first + [c for c in all_summary.columns if c not in cols_first]]\n',
            'all_summary.sort_values("macro_f1", ascending=False, inplace=True, na_position="last")\n',
            'all_summary\n',
        ),
        md('## 4. Plot top models by macro-F1'),
        code(
            'import matplotlib.pyplot as plt\n',
            'top = all_summary.dropna(subset=["macro_f1"]).head(15)\n',
            'plt.figure(figsize=(10, 5))\n',
            'plt.barh(top["model"].astype(str)[::-1], top["macro_f1"][::-1])\n',
            'plt.xlabel("macro_f1"); plt.title("Top 15 models by macro_f1"); plt.tight_layout(); plt.show()\n',
        ),
        *cell_zip_results(),
    ]
    return make_notebook(cells, gpu=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for spec in NB_SPECS:
        nb = build_notebook(spec)
        out = OUT_DIR / spec['filename']
        out.write_text(json.dumps(nb, indent=1), encoding='utf-8')
        written.append(out.name)
    explorer = OUT_DIR / '99_results_explorer.ipynb'
    explorer.write_text(json.dumps(build_explorer_notebook(), indent=1), encoding='utf-8')
    written.append(explorer.name)
    print(f'Generated {len(written)} notebooks in {OUT_DIR}:')
    for name in written:
        print(' -', name)


if __name__ == '__main__':
    main()
