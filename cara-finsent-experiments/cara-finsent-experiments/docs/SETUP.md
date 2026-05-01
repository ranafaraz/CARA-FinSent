# Setup Guide

## Option A: Local Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python -m compileall -q src scripts
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
python -m compileall -q src scripts
```

## Option B: Docker

```bash
docker build -t cara-finsent .
docker run -it --rm -v "$PWD:/app" cara-finsent bash
```

Inside container:

```bash
python -m compileall -q src scripts
```

## Option C: Google Colab

```python
!git clone <your-repo-url>
%cd cara-finsent-experiments
!pip install -r requirements.txt
```

For FinBERT, choose `Runtime > Change runtime type > GPU`.

## Environment variables

Create `.env` from `.env.example` or export variables manually.

Required only for SEC 10-K collection:

```bash
export SEC_USER_AGENT="Your Name your_email@example.com"
```

Optional:

```bash
export NEWSAPI_KEY="..."
export FINNHUB_API_KEY="..."
```
