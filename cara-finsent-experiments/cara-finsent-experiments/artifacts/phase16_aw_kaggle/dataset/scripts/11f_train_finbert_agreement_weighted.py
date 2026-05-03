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

import json
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

from cara_finsent.data_utils import (  # noqa: E402
    STANDARD_LABELS,
    auto_detect_gold_split,
    load_gold_split,
    set_global_seeds,
    split_label_distribution,
    text_hash_leakage_count,
)
from cara_finsent.io_utils import git_commit_sha, save_dataframe, save_json, timestamp  # noqa: E402
from cara_finsent.label_mapping import encode_labels_for_model, remap_probs, model_label_remap  # noqa: E402
from cara_finsent.metrics import metrics_with_optional_proba  # noqa: E402


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
        # Binary: 0.0 below the high-agreement threshold, 1.0 at/above it.
        threshold = high_agreement_threshold(agreement_scores)
        return (np.asarray(agreement_scores) >= threshold).astype(np.float32)
    else:
        raise ValueError(f"Unknown weight schedule: {weight_schedule}")


def high_agreement_threshold(agreement_scores) -> float:
    """Return the high-agreement threshold matching the agreement scale.

    PhraseBank stores agreement as decimals (0.50, 0.66, 0.75, 1.00). Some
    upstream sources expose the same values multiplied by 100. This helper
    inspects the score range and returns 0.75 or 75 accordingly so that the
    `high_only` schedule never silently filters out the entire training set.
    """
    arr = np.asarray(agreement_scores, dtype=float)
    if arr.size == 0:
        return 0.75
    return 0.75 if float(arr.max()) <= 1.0 else 75.0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', default='ProsusAI/finbert')
    ap.add_argument('--data', default=None)
    ap.add_argument('--dataset_name', default='phrasebank', choices=['phrasebank', 'fiqa'])
    ap.add_argument('--weight_schedule', default='linear', choices=['all_equal', 'linear', 'strong', 'high_only'])
    ap.add_argument('--num_epochs', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--learning_rate', type=float, default=2e-5)
    ap.add_argument('--max_length', type=int, default=128)
    ap.add_argument('--warmup_steps', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output_dir', default='results')
    # Aliases so the seed-sweep orchestrator (scripts/19_run_seed_sweep.py)
    # can pass --results_dir / --figures_dir uniformly across all experiments.
    ap.add_argument('--results_dir', default=None,
                    help='Alias for --output_dir; takes precedence when set.')
    ap.add_argument('--figures_dir', default='figures',
                    help='Accepted for orchestrator compatibility; this script writes no figures.')
    ap.add_argument('--models_dir', default='models')
    args = ap.parse_args()
    if args.results_dir:
        args.output_dir = args.results_dir

    set_global_seeds(args.seed, enable_deep_learning=True)
    ts = timestamp()
    
    print(f"[INFO] FinBERT Agreement-Aware Fine-Tuning (seed={args.seed}, schedule={args.weight_schedule})", flush=True)
    print(f"[INFO] Model: {args.model_name}, Epochs: {args.num_epochs}", flush=True)
    
    # Load data
    if not args.data:
        args.data = str(auto_detect_gold_split(args.dataset_name))
        print(f'[AUTO] data = {args.data}', flush=True)
    
    train_df, val_df, test_df = load_gold_split(args.data)
    print('[INFO] Using controlled gold split', flush=True)

    split_source = 'controlled_gold_split'
    benchmark_mode = f'{args.dataset_name}_in_domain'
    label_dist = split_label_distribution(train_df, val_df, test_df)
    leakage_count = text_hash_leakage_count(train_df, val_df, test_df)
    git_sha = git_commit_sha(PROJECT_ROOT)
    
    # Check for agreement column
    if 'agreement' not in train_df.columns:
        print('[WARN] No agreement column; falling back to all_equal weights', flush=True)
        args.weight_schedule = 'all_equal'
    
    # Compute weights
    if args.weight_schedule == 'high_only':
        # Filter to high-agreement examples using a scale-aware threshold.
        threshold = high_agreement_threshold(train_df['agreement'].values)
        train_df_filtered = train_df[train_df['agreement'] >= threshold].reset_index(drop=True)
        print(
            f"[INFO] high_only (threshold={threshold}): "
            f"{len(train_df_filtered)}/{len(train_df)} high-agreement examples retained",
            flush=True,
        )
        if len(train_df_filtered) == 0:
            raise SystemExit(
                f'high_only filter produced 0 training rows (threshold={threshold}). '
                'Check that the agreement column is on the expected scale.'
            )
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
    
    print(f"[INFO] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}", flush=True)
    
    # Load model & tokenizer
    print("[INFO] Loading transformers...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    
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
    train_df['label_id'] = encode_labels_for_model(train_df['label'], id2label)
    if sample_weights is not None:
        train_df['sample_weight'] = sample_weights
    
    train_ds = Dataset.from_pandas(train_df[['text', 'label_id']]).map(
        tokenize, batched=True, remove_columns=['text']
    )
    train_ds = train_ds.rename_column('label_id', 'labels')
    
    # Add weights column if needed
    if sample_weights is not None:
        def add_weights(examples, indices):
            # batched=True so `indices` is always a list/range; this avoids the
            # 'int object is not iterable' error seen with the unbatched form.
            examples['sample_weight'] = [float(sample_weights[i]) for i in indices]
            return examples
        train_ds = train_ds.map(add_weights, with_indices=True, batched=True)
    
    val_df_copy = val_df.copy()
    val_df_copy['label_id'] = encode_labels_for_model(val_df_copy['label'], id2label)
    val_ds = Dataset.from_pandas(val_df_copy[['text', 'label_id']]).map(
        tokenize, batched=True, remove_columns=['text']
    )
    val_ds = val_ds.rename_column('label_id', 'labels')

    test_df_copy = test_df.copy()
    test_df_copy['label_id'] = encode_labels_for_model(test_df_copy['label'], id2label)
    test_ds = Dataset.from_pandas(test_df_copy[['text', 'label_id']]).map(
        tokenize, batched=True, remove_columns=['text']
    )
    test_ds = test_ds.rename_column('label_id', 'labels')

    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score, f1_score
        logits, labels = eval_pred
        probs_native_eval = torch.softmax(torch.tensor(logits), dim=1).numpy()
        probs_canon_eval = remap_probs(probs_native_eval, id2label)
        y_pred_eval = [STANDARD_LABELS[int(i)] for i in probs_canon_eval.argmax(axis=1)]
        from cara_finsent.label_mapping import canonical_label as _cl
        y_true_eval = [_cl(id2label[int(i)]) for i in labels]
        return {
            'accuracy': accuracy_score(y_true_eval, y_pred_eval),
            'macro_f1': f1_score(y_true_eval, y_pred_eval, average='macro', zero_division=0),
        }

    # Training arguments: validation-driven model selection + early stopping
    checkpoint_dir = str(Path(args.models_dir) / f'finbert_agreement_weighted_{args.weight_schedule}_{ts}')
    training_kwargs = dict(
        output_dir=checkpoint_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        greater_is_better=True,
        seed=args.seed,
        fp16=False,
        report_to='none',
        disable_tqdm=False,
    )
    try:
        training_args = TrainingArguments(eval_strategy='epoch', **training_kwargs)
    except TypeError:
        training_args = TrainingArguments(evaluation_strategy='epoch', **training_kwargs)
    
    # Custom trainer for weighted loss
    from transformers.trainer import Trainer
    from transformers import EarlyStoppingCallback

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Newer transformers versions pass extra kwargs (e.g. num_items_in_batch).
            labels = inputs.pop("labels")
            sample_weight = inputs.pop("sample_weight", None)

            outputs = model(**inputs)
            logits = outputs.logits

            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fn(logits, labels)

            if sample_weight is not None:
                if not isinstance(sample_weight, torch.Tensor):
                    sample_weight = torch.as_tensor(sample_weight)
                sample_weight = sample_weight.to(loss.device).float()
                loss = (loss * sample_weight).mean()
            else:
                loss = loss.mean()

            return (loss, outputs) if return_outputs else loss

    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]

    # Train
    print(f"[INFO] Starting training for {args.num_epochs} epochs...", flush=True)
    trainer_cls = WeightedTrainer if sample_weights is not None else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    
    train_start = time.perf_counter()
    trainer.train()
    train_elapsed = time.perf_counter() - train_start
    print(f"[OK] Training complete ({train_elapsed:.1f}s)", flush=True)
    
    # Evaluate on test set
    print('[INFO] Evaluating on test set...', flush=True)
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
    metrics['dataset_name'] = args.dataset_name
    metrics['dataset_file'] = str(Path(args.data))
    metrics['split_source'] = split_source
    metrics['benchmark_mode'] = benchmark_mode
    metrics['train_seconds'] = train_elapsed
    metrics['eval_seconds'] = eval_elapsed
    metrics['train_rows'] = len(train_df)
    metrics['val_rows'] = len(val_df)
    metrics['test_rows'] = len(test_df)
    metrics['seed'] = args.seed
    metrics['model_name'] = args.model_name
    metrics['native_id2label'] = json.dumps(id2label, sort_keys=True)
    metrics['canonical_remap'] = json.dumps(remap_cols)
    metrics['label_distribution_train'] = json.dumps(label_dist['train'], sort_keys=True)
    metrics['label_distribution_val'] = json.dumps(label_dist['val'], sort_keys=True)
    metrics['label_distribution_test'] = json.dumps(label_dist['test'], sort_keys=True)
    metrics['text_hash_leakage_count'] = leakage_count
    metrics['git_commit_sha'] = git_sha
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
    pred_df['dataset_name'] = args.dataset_name
    pred_df['split_source'] = split_source
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
        'dataset_name': args.dataset_name,
        'dataset_file': str(Path(args.data)),
        'split_source': split_source,
        'benchmark_mode': benchmark_mode,
        'seed': args.seed,
        'training_type': 'agreement_weighted_finetuning',
        'weight_schedule': args.weight_schedule,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'train_rows': len(train_df),
        'val_rows': len(val_df),
        'test_rows': len(test_df),
        'label_distribution': label_dist,
        'text_hash_leakage_count': leakage_count,
        'native_id2label': id2label,
        'canonical_remap': remap_cols,
        'git_commit_sha': git_sha,
        'accuracy': float(metrics['accuracy']),
        'macro_f1': float(metrics['macro_f1']),
        'train_seconds': float(metrics['train_seconds']),
        'eval_seconds': float(metrics['eval_seconds']),
        'checkpoint_dir': checkpoint_dir,
        'summary_csv': str(summary_path),
        'predictions_csv': str(pred_path),
    }
    save_json(manifest, args.output_dir, 'finbert_agreement_weighted_manifest', ts)
    
    print('[DONE] Agreement-aware fine-tuning complete', flush=True)


if __name__ == '__main__':
    main()
