#!/usr/bin/env python3
"""Phase 2 Step D: FinBERT fine-tuning on PhraseBank for 3 epochs.
Performs supervised fine-tuning on the canonical training set.

Outputs:
  results/YYYY-MM-DD/finbert_finetuned_summary_<ts>.csv
  results/YYYY-MM-DD/finbert_finetuned_predictions_<ts>.csv
  models/finbert_finetuned_<ts>/  (checkpoint directory)
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
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

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
    ap.add_argument('--model_name', default='ProsusAI/finbert')
    ap.add_argument('--data', default=None)
    ap.add_argument('--text_col', default=None)
    ap.add_argument('--label_col', default=None)
    ap.add_argument('--num_epochs', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--learning_rate', type=float, default=2e-5)
    ap.add_argument('--max_length', type=int, default=128)
    ap.add_argument('--warmup_steps', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output_dir', default='results')
    ap.add_argument('--models_dir', default='models')
    args = ap.parse_args()

    set_global_seeds(args.seed)
    ts = timestamp()
    
    print(f"[INFO] FinBERT Fine-Tuning on PhraseBank (seed={args.seed})", flush=True)
    print(f"[INFO] Model: {args.model_name}, Epochs: {args.num_epochs}", flush=True)
    
    # Load data
    if not args.data:
        from cara_finsent.data_utils import auto_detect_data
        args.data = str(auto_detect_data())
        print(f'[AUTO] data = {args.data}', flush=True)
    
    df = load_standardized_csv(args.data, args.text_col, args.label_col)
    
    # Extract splits
    if 'split' in df.columns:
        print(f'[INFO] Using pre-existing splits', flush=True)
        train_df = df[df['split'] == 'train'].reset_index(drop=True)
        test_df = df[df['split'] == 'test'].reset_index(drop=True)
    else:
        print(f'[ERROR] No pre-split column found. Fine-tuning requires explicit splits.', flush=True)
        sys.exit(1)
    
    print(f"[INFO] Train: {len(train_df)}, Test: {len(test_df)}", flush=True)
    
    # Load model & tokenizer
    print("[INFO] Loading transformers...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        ignore_mismatched_sizes=True,  # Reinitialize classifier head
    )
    
    config = model.config
    id2label = {int(k): str(v) for k, v in dict(config.id2label).items()}
    print(f"[INFO] Native id2label: {id2label}", flush=True)
    remap_cols = model_label_remap(id2label)
    print(f"[INFO] Remap columns: {remap_cols}", flush=True)
    
    # Prepare datasets
    def tokenize(examples):
        return tokenizer(
            examples['text'],
            max_length=args.max_length,
            truncation=True,
            padding='max_length',
        )
    
    train_df = train_df.copy()
    train_df['label_id'] = train_df['label'].map(STANDARD_LABEL2ID)
    train_ds = Dataset.from_pandas(train_df[['text', 'label_id']]).map(
        tokenize, batched=True, remove_columns=['text']
    )
    train_ds = train_ds.rename_column('label_id', 'labels')
    
    test_df_copy = test_df.copy()
    test_df_copy['label_id'] = test_df_copy['label'].map(STANDARD_LABEL2ID)
    test_ds = Dataset.from_pandas(test_df_copy[['text', 'label_id']]).map(
        tokenize, batched=True, remove_columns=['text']
    )
    test_ds = test_ds.rename_column('label_id', 'labels')
    
    # Training arguments
    checkpoint_dir = str(Path(args.models_dir) / f'finbert_finetuned_{ts}')
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=500,
        eval_strategy='no',  # No validation during training
        seed=args.seed,
        fp16=False,
        report_to='none',
        disable_tqdm=False,
    )
    
    # Train
    print(f"[INFO] Starting training for {args.num_epochs} epochs...", flush=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
    )
    
    train_start = time.perf_counter()
    trainer.train()
    train_elapsed = time.perf_counter() - train_start
    print(f"[OK] Training complete ({train_elapsed:.1f}s)", flush=True)
    
    # Evaluate on test set
    print(f"[INFO] Evaluating on test set...", flush=True)
    eval_start = time.perf_counter()
    pred_output = trainer.predict(test_ds)
    eval_elapsed = time.perf_counter() - eval_start
    
    # Process predictions
    logits = pred_output.predictions
    probs_native = torch.softmax(torch.tensor(logits), dim=1).numpy()
    probs_canon = remap_probs(probs_native, id2label)
    
    y_pred = np.array([STANDARD_LABELS[int(i)] for i in probs_canon.argmax(axis=1)])
    y_true = test_df['label'].values
    
    # Metrics
    metrics = metrics_with_optional_proba(y_true, y_pred, probs_canon, model_name='finbert_finetuned')
    metrics['train_seconds'] = train_elapsed
    metrics['eval_seconds'] = eval_elapsed
    metrics['train_rows'] = len(train_df)
    metrics['test_rows'] = len(test_df)
    metrics['seed'] = args.seed
    metrics['model_name'] = args.model_name
    metrics['num_epochs'] = args.num_epochs
    metrics['checkpoint_dir'] = checkpoint_dir
    
    summary_df = pd.DataFrame([metrics])
    summary_path = save_dataframe(summary_df, args.output_dir, 'finbert_finetuned_summary', ts)
    print(f"[OK] Summary -> {summary_path}", flush=True)
    print(f"     Accuracy: {metrics['accuracy']:.4f}")
    print(f"     Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"     Train time: {metrics['train_seconds']:.1f}s")
    
    # Predictions CSV
    pred_df = test_df[['id', 'text', 'label']].copy()
    pred_df['prediction'] = y_pred
    pred_df['confidence'] = probs_canon.max(axis=1)
    for i, cls in enumerate(STANDARD_LABELS):
        pred_df[f'proba_{cls}'] = probs_canon[:, i]
    pred_path = save_dataframe(pred_df, args.output_dir, 'finbert_finetuned_predictions', ts)
    print(f"[OK] Predictions -> {pred_path}", flush=True)
    
    # Manifest
    manifest = {
        'timestamp_utc': ts,
        'model_name': args.model_name,
        'seed': args.seed,
        'training_type': 'supervised_finetuning',
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'train_rows': len(train_df),
        'test_rows': len(test_df),
        'accuracy': float(metrics['accuracy']),
        'macro_f1': float(metrics['macro_f1']),
        'train_seconds': float(metrics['train_seconds']),
        'eval_seconds': float(metrics['eval_seconds']),
        'checkpoint_dir': checkpoint_dir,
        'summary_csv': str(summary_path),
        'predictions_csv': str(pred_path),
    }
    save_json(manifest, args.output_dir, 'finbert_finetuned_manifest', ts)
    
    print(f"[DONE] Fine-tuning complete", flush=True)


if __name__ == '__main__':
    main()
