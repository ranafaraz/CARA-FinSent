#!/usr/bin/env python3
"""Collect additional financial sentiment datasets from HuggingFace to enrich the training corpus.

Target datasets (all public, no auth required):
  1. zeroshot/twitter-financial-news-sentiment  ~11k  Bearish/Bullish/Neutral
  2. nickmuchi/financial-classification          ~6k   headlines (0=neg,1=neu,2=pos)
  3. FIN-NLP/fingpt-sentiment-train              ~76k  sampled 8k  strong/mod/weak labels
  4. TheFinAI/flare-fiqasa                       ~1k   FiQA-SA variant
  5. pauri32/fiqa-2018                           ~1k   FiQA opinions
  6. FinanceInc/auditor_sentiment                ~10k  auditor text sentiment
  7. TimKoornstra/financial-phrasebank-sentiment extra  phrasebank alt version
  8. oliverguhr/financial-news-german            skip  German – out of scope
  9. StephanAkkerman/stock-market-tweets-data    ~5k   tweet polarity

Outputs:
  data/processed/<DATE>/extra_datasets_standardized_<TS>.csv
  data/processed/latest.csv  (MERGED with existing latest.csv)
  results/<DATE>/extra_datasets_inventory_<TS>.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

import os
import traceback

_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    try:
        import huggingface_hub
        huggingface_hub.login(token=_hf_token, add_to_git_credential=False)
        print("[INFO] HuggingFace authenticated via HF_TOKEN")
    except Exception as e:
        print(f"[WARN] HF login failed: {e}")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse
import pandas as pd
from datasets import load_dataset, Dataset

from cara_finsent.data_utils import STANDARD_LABELS, set_global_seeds
from cara_finsent.io_utils import save_dataframe, timestamp

LABEL_MAP_BEARISH_BULLISH = {
    "Bearish": "negative",
    "Bullish": "positive",
    "Neutral": "neutral",
    "bearish": "negative",
    "bullish": "positive",
    "neutral": "neutral",
    "BEARISH": "negative",
    "BULLISH": "positive",
    "NEUTRAL": "neutral",
    "negative": "negative",
    "positive": "positive",
    "0": "negative",
    "1": "neutral",
    "2": "positive",
    0: "negative",
    1: "neutral",
    2: "positive",
    -1: "negative",
}

FINGPT_MAP = {
    "strong negative": "negative",
    "moderately negative": "negative",
    "mildly negative": "negative",
    "strong positive": "positive",
    "moderately positive": "positive",
    "mildly positive": "positive",
    "neutral": "neutral",
    "negative": "negative",
    "positive": "positive",
}


def normalize_label(raw) -> str | None:
    if raw is None:
        return None
    mapped = LABEL_MAP_BEARISH_BULLISH.get(raw) or LABEL_MAP_BEARISH_BULLISH.get(str(raw).lower())
    if mapped:
        return mapped
    s = str(raw).strip().lower()
    if "neg" in s:
        return "negative"
    if "pos" in s or "bull" in s:
        return "positive"
    if "neu" in s:
        return "neutral"
    return None


def rows_from_dataset(ds, text_field: str, label_field: str, source_name: str,
                      label_mapper=None, max_rows: int | None = None) -> list[dict]:
    rows = []
    for i, item in enumerate(ds):
        if max_rows and i >= max_rows:
            break
        text = str(item.get(text_field, "") or "").strip()
        if not text:
            continue
        raw_label = item.get(label_field)
        if label_mapper:
            label = label_mapper(raw_label)
        else:
            label = normalize_label(raw_label)
        if label not in STANDARD_LABELS:
            continue
        rows.append({
            "id": f"{source_name}_{i}",
            "text": text,
            "label": label,
            "agreement": 1.0,
            "agreement_config": "external",
            "source_dataset": source_name,
            "split": "train",
        })
    return rows


def try_load(name: str, *args, **kwargs):
    """Load a HuggingFace dataset with graceful failure."""
    try:
        ds = load_dataset(*args, **kwargs, trust_remote_code=True)
        print(f"[OK] Loaded {name}")
        return ds
    except Exception as e:
        print(f"[SKIP] {name}: {e}")
        return None


def collect_twitter_financial_news(max_rows=None):
    """zeroshot/twitter-financial-news-sentiment — ~11k rows, Bearish/Bullish/Neutral"""
    ds = try_load("twitter_fin_news", "zeroshot/twitter-financial-news-sentiment")
    if ds is None:
        return []
    rows = []
    for split in ["train", "validation", "test"]:
        if split in ds:
            rows += rows_from_dataset(ds[split], "text", "label", "twitter_fin_news", max_rows=max_rows)
    return rows


def collect_financial_classification(max_rows=None):
    """nickmuchi/financial-classification — headlines, field='text', label_field='labels', values 0=neg/1=neu/2=pos"""
    ds = try_load("financial_classification", "nickmuchi/financial-classification")
    if ds is None:
        return []
    rows = []
    for split in ["train", "test"]:
        if split not in ds:
            continue
        rows += rows_from_dataset(ds[split], "text", "labels", "financial_classification", max_rows=max_rows)
    return rows


def collect_fingpt_sentiment(max_rows=8000):
    """FinGPT/fingpt-sentiment-train — large, sample up to max_rows. Try multiple known repo names."""
    candidates = [
        ("FinGPT/fingpt-sentiment-train", "input", "output"),
        ("FinGPT/fingpt_sentiment_train", "input", "output"),
        ("oliverwang15/FinGPT", "input", "output"),
    ]
    for repo, tf, lf in candidates:
        ds = try_load("fingpt_sentiment", repo)
        if ds is None:
            continue

        def mapper(raw):
            if raw is None:
                return None
            return FINGPT_MAP.get(str(raw).strip().lower())

        rows = []
        for split in ["train"]:
            if split in ds:
                rows += rows_from_dataset(ds[split], tf, lf, "fingpt_sentiment",
                                          label_mapper=mapper, max_rows=max_rows)
        if rows:
            return rows
    return []


def collect_flare_fiqasa(max_rows=None):
    """TheFinAI/flare-fiqasa — FiQA-SA from FLARE benchmark"""
    ds = try_load("flare_fiqasa", "TheFinAI/flare-fiqasa")
    if ds is None:
        return []
    rows = []
    for split in ["train", "test", "validation"]:
        if split in ds:
            rows += rows_from_dataset(ds[split], "query", "answer", "flare_fiqasa", max_rows=max_rows)
    return rows


def collect_auditor_sentiment(max_rows=None):
    """FinanceInc/auditor_sentiment — auditor texts with sentiment labels"""
    ds = try_load("auditor_sentiment", "FinanceInc/auditor_sentiment")
    if ds is None:
        return []
    rows = []
    for split in ["train", "test"]:
        if split in ds:
            rows += rows_from_dataset(ds[split], "sentence", "label", "auditor_sentiment", max_rows=max_rows)
    return rows


def collect_stock_market_tweets(max_rows=None):
    """Try multiple tweet/news sentiment datasets with actual labels."""
    candidates = [
        # (repo, text_field, label_field, source_name, label_mapper)
        ("zeroshot/twitter-financial-news-topic", "text", "label", "twitter_fin_topic", None),
        ("Jean-Baptiste/financial_news_sentiment", "title", "label", "jb_fin_news_sentiment", None),
        ("nickmuchi/sec-filing-sentiment-analysis", "text", "label", "sec_filing_sentiment", None),
        ("Birkir/financial_sentiment", "sentence", "label", "birkir_fin_sentiment", None),
        ("bhallaakshit/financial_news_sentiment_analysis", "News Headline", "Sentiment", "akshit_fin_news", None),
    ]
    rows = []
    for repo, tf, lf, name, mapper in candidates:
        ds = try_load(name, repo)
        if ds is None:
            continue
        for split in ["train", "test", "validation"]:
            if split in ds:
                rows += rows_from_dataset(ds[split], tf, lf, name,
                                          label_mapper=mapper, max_rows=max_rows)
    return rows


def collect_financial_sentiment_labelled(max_rows=None):
    """Try multiple small labelled financial sentiment datasets."""
    candidates = [
        ("Sygil/financial-sentiment-analysis", "sentence", "label"),
        ("nickmuchi/financial-text-classification", "sentence", "label"),
        ("simonzhu97/financial_sentiment_classification", "text", "label"),
        ("pauri32/fiqa-2018", "sentence", "sentiment_score", None),  # continuous, handled below
        ("FinancialSupport/FinancialDatasets", "text", "label"),
        ("BerkeleyHaas/firm-news-nlp-dataset", "headline", "sentiment"),
    ]
    rows = []
    for item in candidates:
        if len(item) == 3:
            repo, tf, lf = item
            mapper = None
        else:
            repo, tf, lf, mapper = item
        ds = try_load(repo, repo)
        if ds is None:
            continue
        # For continuous sentiment scores, binarize
        if lf == "sentiment_score":
            for split in ["train", "test", "validation"]:
                if split not in ds:
                    continue
                for i, item_row in enumerate(ds[split]):
                    if max_rows and i >= max_rows:
                        break
                    text = str(item_row.get(tf, "") or "").strip()
                    score = item_row.get(lf)
                    if not text or score is None:
                        continue
                    try:
                        s = float(score)
                        label = "positive" if s > 0.1 else ("negative" if s < -0.1 else "neutral")
                    except (ValueError, TypeError):
                        continue
                    rows.append({"id": f"{repo.replace('/','_')}_{i}", "text": text, "label": label,
                                 "agreement": 1.0, "agreement_config": "external",
                                 "source_dataset": repo.replace("/", "_"), "split": "train"})
        else:
            for split in ["train", "test", "validation"]:
                if split in ds:
                    rows += rows_from_dataset(ds[split], tf, lf, repo.replace("/", "_"),
                                              label_mapper=mapper, max_rows=max_rows)
    return rows


def collect_semeval2017_task5(max_rows=None):
    """Various SemEval / other financial sentiment datasets on HF"""
    candidates = [
        ("yiyanghkust/finbert-tone", "text", "label"),
        ("mjain97/financial_sentiment", "Sentence", "Sentiment"),
        ("AdaptLLM/finance-tasks", "text", "label"),
        ("gtfintechlab/fomc-communication", "sentence", "label"),
        ("financial_phrasebank", "sentence", "label"),   # second call is harmless – dedup handles it
    ]
    rows = []
    for repo, tf, lf in candidates:
        ds = try_load(repo, repo)
        if ds is None:
            continue
        for split in ["train", "test", "validation"]:
            if split in ds:
                rows += rows_from_dataset(ds[split], tf, lf, repo.replace("/", "_"), max_rows=max_rows)
    return rows


def collect_extra_phrasebank(max_rows=None):
    """Try alternate phrasebank and financial headline repositories."""
    candidates = [
        ("TimKoornstra/financial-phrasebank-sentiment", "sentence", "label"),
        ("Rohanpai/financial_phrasebank", "sentence", "label"),
        ("chibi9999/financial_phrasebank_allAgreement", "sentence", "label"),
        ("Pablito1ive/financial_phrase_bank_translated", "sentence", "label"),
        ("winddude/reddit_finance_43_250k", "body", "label"),
        ("mattboggess/financial_news_headline_sentiment", "headline", "sentiment"),
        ("tner/fin_sentiment", "tokens", "tags"),   # token-level, likely skipped
        ("Sygil/financial_phrasebank", "sentence", "label"),
    ]
    rows = []
    for repo, tf, lf in candidates:
        ds = try_load(repo, repo)
        if ds is None:
            continue
        for split in ["train", "test", "validation"]:
            if split in ds:
                rows += rows_from_dataset(ds[split], tf, lf, repo.replace("/", "_"), max_rows=max_rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Collect extra financial sentiment datasets from HuggingFace.")
    parser.add_argument("--max_fingpt", type=int, default=8000, help="Max rows from FinGPT (large dataset).")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_merge", action="store_true", help="Skip merging with existing latest.csv.")
    args = parser.parse_args()

    set_global_seeds(args.seed)
    ts = timestamp()

    all_rows = []
    inventory = []

    collectors = [
        ("Twitter Financial News", collect_twitter_financial_news),
        ("Financial Classification (nickmuchi)", collect_financial_classification),
        ("FinGPT Sentiment", lambda: collect_fingpt_sentiment(max_rows=args.max_fingpt)),
        ("FLARE FiQA-SA", collect_flare_fiqasa),
        ("Auditor Sentiment", collect_auditor_sentiment),
        ("Multi-source Tweet/News Sentiment", collect_stock_market_tweets),
        ("Financial Sentiment Labelled (misc)", collect_financial_sentiment_labelled),
        ("SemEval/FOMC/FinBERT-tone", collect_semeval2017_task5),
        ("Extra PhraseBank & Reddit Finance", collect_extra_phrasebank),
    ]

    for name, fn in collectors:
        print(f"\n>>> Collecting: {name}")
        try:
            rows = fn()
            all_rows.extend(rows)
            label_counts = pd.Series([r["label"] for r in rows]).value_counts().to_dict() if rows else {}
            print(f"    -> {len(rows)} rows | {label_counts}")
            inventory.append({"source": name, "rows": len(rows), **label_counts, "status": "ok"})
        except Exception as exc:
            print(f"    -> FAILED: {exc}")
            traceback.print_exc()
            inventory.append({"source": name, "rows": 0, "status": f"error: {exc}"})

    if not all_rows:
        print("[ERROR] No data collected from any source.")
        sys.exit(1)

    new_df = pd.DataFrame(all_rows).drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"\n=== NEW DATA: {len(new_df)} unique rows ===")
    print(new_df["label"].value_counts())
    print(new_df["source_dataset"].value_counts())

    # Save new data standalone
    new_path = save_dataframe(new_df, args.output_dir, "extra_datasets_standardized", ts)
    print(f"[SAVED] New data -> {new_path}")

    # Merge with existing latest.csv
    if not args.no_merge:
        latest_path = Path("data/processed/latest.csv")
        if latest_path.exists():
            existing = pd.read_csv(latest_path)
            print(f"\n[MERGE] Existing: {len(existing)} rows  +  New: {len(new_df)} rows")
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
            print(f"[MERGE] Combined (dedup): {len(combined)} rows")
            print(combined["label"].value_counts())
            print(combined["source_dataset"].value_counts())
        else:
            combined = new_df

        combined_path = save_dataframe(combined, args.output_dir, "combined_standardized", ts)
        # Update latest.csv
        import shutil
        shutil.copy(str(combined_path), str(latest_path))
        print(f"[DONE] Combined -> {combined_path}")
        print(f"[DONE] latest.csv updated ({len(combined)} rows)")

    # Save inventory
    inv_df = pd.DataFrame(inventory).fillna(0)
    inv_path = save_dataframe(inv_df, args.results_dir, "extra_datasets_inventory", ts)
    print(f"[DONE] Inventory -> {inv_path}")


if __name__ == "__main__":
    main()
