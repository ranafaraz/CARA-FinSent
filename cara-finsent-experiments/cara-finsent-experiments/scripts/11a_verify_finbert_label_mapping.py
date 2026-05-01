#!/usr/bin/env python3
"""Phase 2 Step A: FinBERT label mapping sanity check.
Verify that the pretrained FinBERT model's label order is correctly understood
before running any large-scale evaluation or fine-tuning.

Outputs:
  results/YYYY-MM-DD/finbert_label_mapping_sanity_<ts>.csv
  results/YYYY-MM-DD/finbert_label_mapping_manifest_<ts>.json
"""
from __future__ import annotations

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs before import

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from cara_finsent.io_utils import save_dataframe, save_json, timestamp  # noqa: E402
from cara_finsent.label_mapping import (  # noqa: E402
    CANONICAL_LABELS,
    model_label_remap,
    remap_probs,
)


# 15 hand-written sanity examples covering positive, negative, neutral
SANITY_EXAMPLES = [
    # Positive examples
    ("Revenue increased by 25 percent year-over-year.", "positive"),
    ("The company announced record profits this quarter.", "positive"),
    ("Strong earnings growth exceeded analyst expectations.", "positive"),
    ("We are pleased to report excellent financial results.", "positive"),
    ("Market share gained significantly in the key segment.", "positive"),
    # Negative examples
    ("Revenue declined sharply due to market headwinds.", "negative"),
    ("The company reported significant losses this quarter.", "negative"),
    ("Earnings fell below analyst estimates dramatically.", "negative"),
    ("We face serious challenges in the current environment.", "negative"),
    ("Market share lost substantially in core business.", "negative"),
    # Neutral examples
    ("The company operates in the financial sector.", "neutral"),
    ("We have 50,000 employees across 30 countries.", "neutral"),
    ("Our headquarters is located in New York.", "neutral"),
    ("The board met on Tuesday to discuss strategy.", "neutral"),
    ("Revenue in 2025 was reported at $5 billion.", "neutral"),
]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', default='ProsusAI/finbert')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output_dir', default='results')
    args = ap.parse_args()

    # Lazy import to avoid slow transformers init on Windows
    print("[INFO] Loading transformers...", file=sys.stderr, flush=True)
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    print("[INFO] Transformers loaded.", file=sys.stderr, flush=True)

    ts = timestamp()
    print(f"[INFO] FinBERT Label Mapping Sanity Check (seed={args.seed})")
    print(f"[INFO] Model: {args.model_name}")

    # Load model config
    config = AutoConfig.from_pretrained(args.model_name)
    id2label = {int(k): str(v) for k, v in dict(config.id2label).items()}
    print(f"[INFO] Native id2label: {id2label}")

    # Compute canonical remap
    try:
        remap_cols = model_label_remap(id2label)
        print(f"[INFO] Canonical remap columns: {remap_cols}")
    except Exception as e:
        print(f"[ERROR] Failed to compute remap: {e}")
        return

    # Load model & tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    model.eval()

    # Run sanity examples
    texts = [t for t, _ in SANITY_EXAMPLES]
    expected_labels = [l for _, l in SANITY_EXAMPLES]

    enc = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits.cpu().numpy()

    probs_native = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    probs_canon = remap_probs(probs_native, id2label)

    pred_native_labels = [id2label[int(i)] for i in probs_native.argmax(axis=1)]
    pred_canon_labels = [CANONICAL_LABELS[int(i)] for i in probs_canon.argmax(axis=1)]

    results = []
    correct_native = 0
    correct_canon = 0
    for i, (text, expected) in enumerate(SANITY_EXAMPLES):
        pred_native = pred_native_labels[i]
        pred_canon = pred_canon_labels[i]
        match_native = 1 if pred_native == expected else 0
        match_canon = 1 if pred_canon == expected else 0
        correct_native += match_native
        correct_canon += match_canon
        results.append({
            'example_id': i,
            'text': text,
            'expected_label': expected,
            'pred_native': pred_native,
            'pred_canonical': pred_canon,
            'match_native': match_native,
            'match_canonical': match_canon,
            'p_negative': float(probs_canon[i, 0]),
            'p_neutral': float(probs_canon[i, 1]),
            'p_positive': float(probs_canon[i, 2]),
        })

    df = pd.DataFrame(results)
    path = save_dataframe(df, args.output_dir, 'finbert_label_mapping_sanity', ts)

    # Status check
    accuracy_native = correct_native / len(SANITY_EXAMPLES)
    accuracy_canon = correct_canon / len(SANITY_EXAMPLES)
    status = 'PASS' if accuracy_canon >= 0.80 else 'WARN' if accuracy_canon >= 0.60 else 'FAIL'

    manifest = {
        'timestamp_utc': ts,
        'model_name': args.model_name,
        'seed': args.seed,
        'n_examples': len(SANITY_EXAMPLES),
        'native_id2label': id2label,
        'canonical_labels': CANONICAL_LABELS,
        'remap_columns': remap_cols,
        'accuracy_native_order': float(accuracy_native),
        'accuracy_canonical_order': float(accuracy_canon),
        'sanity_check_status': status,
        'sanity_csv': str(path),
    }
    save_json(manifest, args.output_dir, 'finbert_label_mapping_manifest', ts)

    print(f"[OK] Sanity check complete")
    print(f"     Native order accuracy:    {accuracy_native:.2%}")
    print(f"     Canonical order accuracy: {accuracy_canon:.2%}")
    print(f"     Status: {status}")
    print(f"[OK] Results -> {path}")
    if status != 'PASS':
        print(f"[WARN] Sanity check status={status}. Review label mapping before proceeding.")
        sys.exit(1)


if __name__ == '__main__':
    main()
