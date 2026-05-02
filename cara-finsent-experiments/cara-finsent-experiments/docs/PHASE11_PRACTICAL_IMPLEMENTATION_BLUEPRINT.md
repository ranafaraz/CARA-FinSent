# Phase 11 — Practical Implementation Blueprint

Date: 2026-05-02
Scope: Translate the CARA-FinSent research artifact into a deployable product
concept. This is a **blueprint only** — no application code is implemented in
Phase 11. Implementation belongs to a later phase.

## Goal

Provide a decision-grade financial sentiment service that a finance analyst,
investor-relations team, or research desk can interact with via a hosted demo,
an API, and a model card on Hugging Face. The system must surface confidence,
calibration, and abstention, and must communicate uncertainty rather than hide
it.

## Deployment surfaces

### 1. Hugging Face publishing plan

- **Repo name (proposed):** `ranafarazahmed/cara-finsent-aw-finbert`
- **Artifact:** agreement-weighted FinBERT checkpoint (best macro-F1 seed).
- **Tokenizer:** mirror of `ProsusAI/finbert`.
- **Card sections:** model summary, training data, training procedure, intended
  use, out-of-scope use, calibration report (ECE@10 = 0.0495), seed-stability
  table, abstention recommendations, paired statistical validation against
  zero-shot, license, citation.
- **Files included:** `config.json`, `pytorch_model.bin`/`model.safetensors`,
  tokenizer files, `inference_example.py`, `model_card.md`.
- **Files excluded:** raw training predictions, raw datasets, environment
  files, secrets.

### 2. Gradio / Streamlit demo plan

- **Framework:** Gradio first (simpler share link), Streamlit as alternative
  for richer dashboards.
- **Inputs:** free-text financial sentence; optional ticker filter; optional
  abstention threshold slider (default 0.7).
- **Outputs:** predicted label, full softmax bar chart, calibrated probability
  via temperature scaling, abstention flag, confidence ribbon.
- **Disclaimers:** persistent banner — *"Research demo. Not financial advice.
  Probabilities reflect model uncertainty and may be miscalibrated on
  out-of-distribution text."*

### 3. FastAPI inference endpoint

- **Routes:**
  - `POST /predict` — single text in, label + probabilities + abstention flag
    out.
  - `POST /predict/batch` — list of texts.
  - `GET  /healthz` — container liveness.
  - `GET  /modelcard` — JSON of the model card.
- **Auth:** bearer token.
- **Rate limit:** simple token-bucket per IP.
- **Telemetry:** prediction latency histogram, confidence histogram, abstention
  rate; **no input text logging** unless explicitly opted-in.

### 4. Dashboard architecture

- **Frontend:** React + Recharts (or Streamlit alt).
- **Backend:** the FastAPI service above + a small aggregation layer that
  groups predictions by company / sector / time window.
- **Storage:** SQLite for the demo, Postgres for production.

### 5. RAG finance assistant architecture (forward-looking)

- **Retriever:** dense retriever over indexed SEC 10-K and recent news headlines.
- **Generator:** a guard-railed LLM that cites the sentiment classifier
  output and the retrieved evidence.
- **Sentiment fusion:** the AW FinBERT classifier provides per-document and
  per-snippet sentiment, with the abstention flag explicitly surfaced to the
  generator so the LLM can decline to summarise low-confidence inputs.

## Data sources for demos and dashboards

- Financial headlines (RSS).
- SEC 10-K filings (EDGAR).
- Company news endpoints (compliant providers only).
- Public Kaggle datasets for static demos and reproducibility examples.

## Charts to expose

- Company sentiment trend (line chart by day).
- Sector sentiment trend (stacked area).
- Confidence over time (line + abstention overlay).
- Positive / neutral / negative distribution (donut).
- Abstention rate over time (line).

## Cautions and disclosures

- **Not financial advice.**
- **Delayed or limited data.**
- **Model uncertainty is real:** zero-shot FinBERT is better calibrated than
  AW FinBERT; downstream consumers should consider temperature scaling or
  calibrated abstention.
- **Explainability limitations:** the system reports probabilities and an
  abstention flag, not causal explanations of market moves.
- **Out-of-distribution risk:** PhraseBank-trained models may degrade on
  social-media or earnings-call transcripts.

## Implementation phasing (out of scope for Phase 11)

1. Phase A — package the model and publish the Hugging Face card.
2. Phase B — ship the Gradio demo behind a Hugging Face Space.
3. Phase C — ship the FastAPI service with a calibrated head.
4. Phase D — build the dashboard.
5. Phase E — integrate the RAG assistant.

Phase 11 stops at the blueprint level.
