# Repository Organization

> This document is the **single source of truth** for what lives where.
> Read this before touching any folder.

Last reorganized: **2026-05-01** (after canonical dataset rebuild, build_id `20260430_212903`).

---

## TL;DR

- **Active code**: `src/`, `scripts/`, `tests/`, `methodology/`, `configs/`, `docs/`, `notebooks/`
- **Active data**: `data/raw/` and `data/processed/latest/` (+ `data/processed/latest.csv` shim)
- **New experiment outputs**: land in date-stamped folders (e.g. `results/2026-05-01/`, `figures/2026-05-01/`, `models/<name>_<timestamp>/`)
- **Anything in a folder named `_archive/`**: pre-rebuild artifacts. **Do not use**, do not delete unless you're sure.

---

## Active vs Archived

### `data/`

| Path | Purpose | Status |
|---|---|---|
| `data/raw/` | Collector outputs (FOMC, SemEval, PhraseBank, FiQA). Inputs to `scripts/05_build_dataset.py`. | **Active** |
| `data/processed/latest/dataset.csv` | **THE canonical dataset** (13,916 rows, build_id `20260430_212903`). All experiments read this. | **Active** |
| `data/processed/latest/{manifest.json,data_card.md,contamination_report.csv}` | Provenance + quality reports for the canonical build. | **Active** |
| `data/processed/latest.csv` | Backwards-compat flat file (used by `auto_detect_data()` in legacy scripts). | **Active** |
| `data/processed/archive/dataset_<build_id>.csv` | Immutable snapshots of past canonical builds. | **Active (read-only)** |
| `data/_archive/` | Pre-rebuild intermediates and failed scrapes. | **Archived** |

### `results/`

| Path | Purpose | Status |
|---|---|---|
| `results/<YYYY-MM-DD>/` | Output of experiment scripts (CSV, JSON, manifests). | **Active** — new runs land here |
| `results/logs/<YYYY-MM-DD>/` | Per-script logs (auto-rotated by `io_utils`). | **Active** |
| `results/_archive/2026-04-30/` | All ~241 result files from runs on the **pre-rebuild** dataset. | **Archived** |
| `results/_archive/logs/2026-04-30/` | Logs from those pre-rebuild runs. | **Archived** |
| `results/_archive/legacy_logs/` | Loose top-level `*.log` / `*.txt` files from ad-hoc runs. | **Archived** |

### `figures/`

| Path | Purpose | Status |
|---|---|---|
| `figures/<YYYY-MM-DD>/` | New plots (confusion matrices, reliability, comparison bars). | **Active** — new runs land here |
| `figures/_archive/` | 61 PNGs generated against the pre-rebuild dataset. | **Archived** |

### `models/`

| Path | Purpose | Status |
|---|---|---|
| `models/<name>_<timestamp>/` | Fine-tuned checkpoints (FinBERT, etc.). | **Active** |
| `models/_archive/finbert_20260430_140446/` | FinBERT fine-tuned on the **old** dataset. | **Archived** |
| `models/_archive/finbert_20260430_202719/` | Partial FinBERT run that was killed during reorganization. | **Archived** |

### Code / docs (all active)

| Path | Purpose |
|---|---|
| `src/cara_finsent/` | Library code (preprocessing, dedup, retrieval, metrics, plotting, io). |
| `scripts/` | Numbered pipeline + experiment scripts (00–17, 90). See `docs/EXPERIMENTS.md`. |
| `tests/` | Pytest suite (`test_imports.py`, `test_dataset_quality.py` — 14 quality gates). |
| `methodology/` | 12 markdown docs explaining every methodological decision. |
| `configs/default.yaml` | Pipeline hyper-parameters. |
| `docs/` | Setup, execution guide, data sources, experiments overview. |
| `notebooks/colab/` | Colab mirrors of the scripts (auto-generated). |

---

## Where will my next run output land?

Scripts use `cara_finsent.io_utils.timestamped_path(...)` which writes to:

```
results/<today>/<prefix>_<timestamp>.csv
results/logs/<today>/<script>_<timestamp>.log
figures/<today>/<prefix>_<timestamp>.png        # NEW (post-2026-05-01)
models/<name>_<timestamp>/                       # for FinBERT
```

So today's runs land under `results/2026-05-01/` and `figures/2026-05-01/` (or whichever date).

---

## Reading the canonical dataset

```python
from cara_finsent.data_utils import auto_detect_data
df = auto_detect_data()  # → data/processed/latest.csv (13,916 rows)
```

Or directly:

```python
import pandas as pd
df = pd.read_csv("data/processed/latest/dataset.csv")
```

Both files are byte-identical except the `latest.csv` shim renames `text_clean_lower → text_lower` for backwards compat.

---

## Cleanup policy

- `_archive/` folders are **kept indefinitely** so the dissertation has a paper trail.
- If disk space is tight, delete `models/_archive/` first (~2.9 GB of obsolete FinBERT checkpoints).
- Never delete `data/processed/archive/` — those are the immutable canonical snapshots referenced by manifests.
- Never delete `data/raw/` — re-collecting is slow and rate-limited.

---

## Pre-rebuild vs post-rebuild markers

Anything with timestamp `20260430_2113xx` or later is **post-rebuild** (new canonical dataset).
Anything with timestamp `20260430_104xxx`, `20260430_12xxxx`, `20260430_140xxx`, `20260430_142xxx`, `20260430_143xxx`, `20260430_144xxx`, or `20260430_2026xx` is **pre-rebuild** and lives in `_archive/`.
