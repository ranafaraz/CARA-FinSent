#!/usr/bin/env python3
"""Phase 2 Step F: FinBERT agreement-aware fine-tuning on PhraseBank.
Uses agreement scores from PhraseBank to weight training examples.

Weight schedules:
  - all_equal: 1.0 for all examples
  - linear: 0.50, 0.66, 0.75, 1.00 for agreement quartiles
  - strong: 0.25, 0.50, 0.75, 1.00 for agreement quartiles
  - high_only: train on 75+ only, test on all

Outputs:
  results/YYYY-MM-DD/finbert_agreement_weighted_summary_<ts>.csv
  results/YYYY-MM-DD/finbert_agreement_weighted_predictions_<ts>.csv
  models/finbert_agreement_weighted_<ts>/  (checkpoint directory)
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


def compute_sample_weights(agreement_scores, weight_schedule='all_equal'):
    """Compute sample weights based on agreement and schedule."""
    if weight_schedule == 'all_equal':
        return np.ones_like(agreement_scores, dtype=np.float32)
    elif weight_schedule == 'linear':
        # Linear: 0.50, 0.66, 0.75, 1.00 for quartiles 25%, 50%, 75%, 100%
        q1, q2, q3 = np.percentile(agreement_scores, [25, 50, 75])
        weights = np.where(agreement_scores < q1, 0.50,
                  np.where(agreement_scores < q2, 0.66,
                  np.where(agreement_scores < q3, 0.75, 1.00)))
        return weights.astype(np.float32)
    elif weight_schedule == 'strong':
        # Strong: 0.25, 0.50, 0.75, 1.00 for quartiles
        q1, q2, q3 = np.percentile(agreement_scores, [25, 50, 75])
        weights = np.where(agreement_scores < q1, 0.25,
                  np.where(agreement_scores < q2, 0.50,
                  np.where(agreement_scores < q3, 0.75, 1.00)))
        return weights.astype(np.float32)
    elif weight_schedule == 'high_only':
        # Binary: 0.0 for <75%, 1.0 for >=75%
        return (agreement_scores >= 75).astype(np.float32)
    else:
        raise ValueError(f"Unknown weight schedule: {weight_schedule}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', default='ProsusAI/finbert')
    ap.add_argument('--data', default=None)
    ap.add_argument('--weight_schedule', default='linear', choices=['all_equal', 'linear', 'strong', 'high_only'])
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
    
    print(f"[INFO] FinBERT Agreement-Aware Fine-Tuning (seed={args.seed}, schedule={args.weight_schedule})", flush=True)
    print(f"[INFO] Model: {args.model_name}, Epochs: {args.num_epochs}", flush=True)
    
    # Load data
    if not args.data:
        from cara_finsent.data_utils import auto_detect_data
        args.data = str(auto_detect_data())
        print(f'[AUTO] data = {args.data}', flush=True)
    
    df = load_standardized_csv(args.data)
    
    # Extract splits
    if 'split' in df.columns:
        print(f'[INFO] Using pre-existing splits', flush=True)
        train_df = df[df['split'] == 'train'].reset_index(drop=True)
        test_df = df[df['split'] == 'test'].reset_index(drop=True)
    else:
        print(f'[ERROR] No pre-split column found', flush=True)
        sys.exit(1)
    
    # Check for agreement column
    if 'agreement' not in train_df.columns:
        print(f'[WARN] No agreement column; falling back to all_equal weights', flush=True)
        args.weight_schedule = 'all_equal'
    
    # Compute weights
    if args.weight_schedule == 'high_only':
        # Filter to high-agreement examples
        train_df_filtered = train_df[train_df['agreement'] >= 75].reset_index(drop=True)
        print(f"[INFO] high_only: {len(train_df_filtered)}/{len(train_df)} high-agreement examples retained", flush=True)
        train_df = train_df_filtered
        sample_weights = None
    else:
        sample_weights = compute_sample_weights(train_df['agreement'].values, args.weight_schedule)
        weight_stats = {
            'min': float(sample_weights.min()),
            'max': float(sample_weights.max()),
            'mean': float(sample_weights.mean()),
        }
        print(f"[INFO] Sample weights ({args.weight_schedule}): min={weight_stats['min']:.2f}, max={weight_stats['max']:.2f}, mean={weight_stats['mean']:.2f}", flush=True)
    
    print(f"[INFO] Train: {len(train_df)}, Test: {len(test_df)}", flush=True)
    
    # Load model & tokenizer
    print("[INFO] Loading transformers...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        ignore_mismatched_sizes=True,
    )
    
    config = model.config
    id2label = {int(k): str(v) for k, v in dict(config.id2label).items()}
    print(f"[INFO] Native id2label: {id2label}", flush=True)
    remap_cols = model_label_remap(id2label)
    
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
    if sample_weights is not None:
        train_df['sample_weight'] = sample_weights
    
    train_ds = Dataset.from_pandas(train_df[['text', 'label_id']]).map(
        tokenize, batched=True, remove_columns=['text']
    )
    train_ds = train_ds.rename_column('label_id', 'labels')
    
    # Add weights column if needed
    if sample_weights is not None:
        def add_weights(examples, indices):
            examples['sample_weight'] = [sample_weights[i] for i in indices]
            return examples
        train_ds = train_ds.map(add_weights, with_indices=True)
    
    test_df_copy = test_df.copy()
    test_df_copy['label_id'] = test_df_copy['label'].map(STANDARD_LABEL2ID)
    test_ds = Dataset.from_pandas(test_df_copy[['text', 'label_id']]).map(
        tokenize, batched=True, remove_columns=['text']
    )
    test_ds = test_ds.rename_column('label_id', 'labels')
    
    # Training arguments
    checkpoint_dir = str(Path(args.models_dir) / f'finbert_agreement_weighted_{args.weight_schedule}_{ts}')
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
        eval_strategy='no',
        seed=args.seed,
        fp16=False,
        report_to='none',
        disable_tqdm=False,
    )
    
    # Custom trainer for weighted loss
    from transformers.trainer import Trainer
    
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            if 'sample_weight' in inputs:
                sample_weight = inputs.pop("sample_weight")
            else:
                sample_weight = None
            
            outputs = model(**inputs)
            logits = outputs.logits
            
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fn(logits, labels)
            
            if sample_weight is not None:
                loss = loss * torch.tensor(sample_weight, device=loss.device)
                loss = loss.mean()
            else:
                loss = loss.mean()
            
            return (loss, outputs) if return_outputs else loss
    
    # Train
    print(f"[INFO] Starting training for {args.num_epochs} epochs...", flush=True)
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
    ) if sample_weights is not None else Trainer(
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
    metrics = metrics_with_optional_proba(y_true, y_pred, probs_canon, model_name='finbert_agreement_weighted')
    metrics['train_seconds'] = train_elapsed
    metrics['eval_seconds'] = eval_elapsed
    metrics['train_rows'] = len(train_df)
    metrics['test_rows'] = len(test_df)
    metrics['seed'] = args.seed
    metrics['model_name'] = args.model_name
    metrics['num_epochs'] = args.num_epochs
    metrics['weight_schedule'] = args.weight_schedule
    metrics['checkpoint_dir'] = checkpoint_dir
    
    summary_df = pd.DataFrame([metrics])
    summary_path = save_dataframe(summary_df, args.output_dir, 'finbert_agreement_weighted_summary', ts)
    print(f"[OK] Summary -> {summary_path}", flush=True)
    print(f"     Accuracy: {metrics['accuracy']:.4f}")
    print(f"     Macro-F1: {metrics['macro_f1']:.4f}")
    
    # Predictions CSV
    pred_df = test_df[['id', 'text', 'label']].copy()
    pred_df['prediction'] = y_pred
    pred_df['confidence'] = probs_canon.max(axis=1)
    for i, cls in enumerate(STANDARD_LABELS):
        pred_df[f'proba_{cls}'] = probs_canon[:, i]
    pred_path = save_dataframe(pred_df, args.output_dir, 'finbert_agreement_weighted_predictions', ts)
    print(f"[OK] Predictions -> {pred_path}", flush=True)
    
    # Manifest
    manifest = {
        'timestamp_utc': ts,
        'model_name': args.model_name,
        'seed': args.seed,
        'training_type': 'agreement_weighted_finetuning',
        'weight_schedule': args.weight_schedule,
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
    save_json(manifest, args.output_dir, 'finbert_agreement_weighted_manifest', ts)
    
    print(f"[DONE] Agreement-aware fine-tuning complete", flush=True)


if __name__ == '__main__':
    main()
