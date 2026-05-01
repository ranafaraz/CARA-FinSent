#!/usr/bin/env python3
"""Validate FinBERT (or any HF) model's native label mapping and run a 20-row
sanity check before any full evaluation.

Outputs:
  data/audit/finbert_label_mapping_<ts>.json
  data/audit/finbert_label_sanity_<ts>.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from cara_finsent.io_utils import save_dataframe, save_json, timestamp  # noqa: E402
from cara_finsent.label_mapping import CANONICAL_LABELS, model_label_remap, remap_probs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', default='ProsusAI/finbert')
    ap.add_argument('--sample_file', required=True, help='CSV with text,label columns to draw 20 examples from.')
    ap.add_argument('--n_samples', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output_dir', default='data/audit')
    args = ap.parse_args()

    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    import torch

    ts = timestamp()
    config = AutoConfig.from_pretrained(args.model_name)
    id2label = {int(k): str(v) for k, v in dict(config.id2label).items()}
    cols = model_label_remap(id2label)
    label2id = {str(v): int(k) for k, v in id2label.items()}
    print(f'[INFO] model id2label = {id2label}')
    print(f'[INFO] canonical column order in native space = {cols}')

    df = pd.read_csv(args.sample_file)
    if 'text' not in df.columns:
        raise SystemExit('sample_file must have a text column')
    sample = df.sample(min(args.n_samples, len(df)), random_state=args.seed).reset_index(drop=True)

    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    model.eval()
    enc = tok(sample['text'].astype(str).tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        logits = model(**enc).logits.cpu().numpy()
    probs_native = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    probs_canon = remap_probs(probs_native, id2label)
    pred_canon_idx = probs_canon.argmax(axis=1)
    sample['model_native_pred'] = [id2label[int(i)] for i in probs_native.argmax(axis=1)]
    sample['canonical_pred'] = [CANONICAL_LABELS[int(i)] for i in pred_canon_idx]
    for i, lbl in enumerate(CANONICAL_LABELS):
        sample[f'p_{lbl}'] = probs_canon[:, i].round(4)

    sanity_path = save_dataframe(sample, args.output_dir, 'finbert_label_sanity', ts)
    manifest = {
        'timestamp_utc': ts,
        'model_name': args.model_name,
        'native_id2label': id2label,
        'native_label2id': label2id,
        'canonical_labels': CANONICAL_LABELS,
        'native_columns_to_canonical': cols,
        'sanity_csv': str(sanity_path),
        'n_samples': int(len(sample)),
    }
    save_json(manifest, args.output_dir, 'finbert_label_mapping', ts)
    print(f'[OK] sanity sample -> {sanity_path}')
    print('--- first 5 rows ---')
    print(sample.head().to_string(index=False))


if __name__ == '__main__':
    main()
