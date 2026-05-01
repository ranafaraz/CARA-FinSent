#!/usr/bin/env python3
"""Phase 2 Step E: Evaluate models trained on PhraseBank, using FiQA as external validation.
Assesses generalization by testing models on the independent FiQA dataset.

Outputs:
  results/YYYY-MM-DD/fiqa_external_validation_summary_<ts>.csv
  results/YYYY-MM-DD/fiqa_external_validation_predictions_<ts>.csv
  results/YYYY-MM-DD/fiqa_external_validation_manifest_<ts>.json
"""
from __future__ import annotations

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from cara_finsent.data_utils import STANDARD_LABELS, load_standardized_csv, set_global_seeds  # noqa: E402
from cara_finsent.io_utils import save_dataframe, save_json, timestamp  # noqa: E402
from cara_finsent.label_mapping import remap_probs, model_label_remap  # noqa: E402
from cara_finsent.metrics import metrics_with_optional_proba  # noqa: E402


STANDARD_LABEL2ID = {label: i for i, label in enumerate(STANDARD_LABELS)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint_dir', required=True, help='Directory of fine-tuned model checkpoint')
    ap.add_argument('--fiqa_data', required=True, help='FiQA split CSV with pre-split column')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--max_length', type=int, default=128)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output_dir', default='results')
    args = ap.parse_args()

    set_global_seeds(args.seed)
    ts = timestamp()
    
    print(f"[INFO] FiQA External Validation (seed={args.seed})", flush=True)
    print(f"[INFO] Checkpoint: {args.checkpoint_dir}", flush=True)
    
    # Load FiQA test data
    df = load_standardized_csv(args.fiqa_data)
    if 'split' in df.columns:
        print(f'[INFO] Using pre-existing FiQA splits', flush=True)
        test_df = df[df['split'] == 'test'].reset_index(drop=True)
    else:
        print(f'[ERROR] No pre-split column in FiQA data', flush=True)
        sys.exit(1)
    
    print(f"[INFO] FiQA test size: {len(test_df)}", flush=True)
    
    # Load checkpoint model
    print("[INFO] Loading checkpoint model...", flush=True)
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_dir)
    model.eval()
    
    config = model.config
    id2label = {int(k): str(v) for k, v in dict(config.id2label).items()}
    print(f"[INFO] Model id2label: {id2label}", flush=True)
    remap_cols = model_label_remap(id2label)
    
    # Batch predict
    texts = test_df['text'].values
    labels = test_df['label'].values
    
    all_probs = []
    start_time = time.perf_counter()
    
    for i in range(0, len(texts), args.batch_size):
        batch_texts = texts[i:i+args.batch_size]
        enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits.cpu().numpy()
        probs_native = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        probs_canon = remap_probs(probs_native, id2label)
        all_probs.append(probs_canon)
        if (i // args.batch_size + 1) % 5 == 0:
            print(f"[INFO] Processed {min(i+args.batch_size, len(texts))}/{len(texts)} examples", flush=True)
    
    elapsed = time.perf_counter() - start_time
    all_probs = np.concatenate(all_probs, axis=0)
    
    # Predictions
    y_pred = np.array([STANDARD_LABELS[int(i)] for i in all_probs.argmax(axis=1)])
    
    # Metrics
    metrics = metrics_with_optional_proba(labels, y_pred, all_probs, model_name='fiqa_external_validation')
    metrics['latency_seconds'] = elapsed
    metrics['latency_per_example_ms'] = 1000.0 * elapsed / len(test_df)
    metrics['test_rows'] = len(test_df)
    metrics['seed'] = args.seed
    metrics['checkpoint_dir'] = args.checkpoint_dir
    
    summary_df = pd.DataFrame([metrics])
    summary_path = save_dataframe(summary_df, args.output_dir, 'fiqa_external_validation_summary', ts)
    print(f"[OK] Summary -> {summary_path}", flush=True)
    print(f"     Accuracy: {metrics['accuracy']:.4f}")
    print(f"     Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"     Latency:  {metrics['latency_seconds']:.2f}s ({metrics['latency_per_example_ms']:.2f}ms/ex)")
    
    # Predictions CSV
    pred_df = test_df[['id', 'text', 'label']].copy()
    pred_df['prediction'] = y_pred
    pred_df['confidence'] = all_probs.max(axis=1)
    for i, cls in enumerate(STANDARD_LABELS):
        pred_df[f'proba_{cls}'] = all_probs[:, i]
    pred_path = save_dataframe(pred_df, args.output_dir, 'fiqa_external_validation_predictions', ts)
    print(f"[OK] Predictions -> {pred_path}", flush=True)
    
    # Manifest
    manifest = {
        'timestamp_utc': ts,
        'checkpoint_dir': args.checkpoint_dir,
        'validation_dataset': 'FiQA',
        'seed': args.seed,
        'test_rows': len(test_df),
        'accuracy': float(metrics['accuracy']),
        'macro_f1': float(metrics['macro_f1']),
        'latency_seconds': float(metrics['latency_seconds']),
        'summary_csv': str(summary_path),
        'predictions_csv': str(pred_path),
    }
    save_json(manifest, args.output_dir, 'fiqa_external_validation_manifest', ts)
    
    print(f"[DONE] External validation complete", flush=True)


if __name__ == '__main__':
    main()
