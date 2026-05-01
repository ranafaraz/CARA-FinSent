#!/usr/bin/env python3
"""Improved FinBERT baseline with confidence diagnostics on controlled gold splits."""
from __future__ import annotations

import json
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / '.env', override=False)
except ImportError:
    pass

import os
_hf_token = os.environ.get('HF_TOKEN')
if _hf_token:
    try:
        from huggingface_hub import login as _hf_login
        _hf_login(token=_hf_token, add_to_git_credential=False)
        print('[INFO] HuggingFace authenticated via HF_TOKEN')
    except Exception as _e:
        print(f'[WARN] HF login failed: {_e}')

os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

import argparse
import time
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments

from cara_finsent.data_utils import (
    STANDARD_LABELS,
    auto_detect_gold_split,
    infer_dataset_name,
    load_gold_split,
    set_global_seeds,
    split_label_distribution,
    text_hash_leakage_count,
)
from cara_finsent.io_utils import git_commit_sha, save_dataframe, save_json, timestamp, write_manifest
from cara_finsent.label_mapping import canonical_label, encode_labels_for_model, model_label_remap, remap_probs
from cara_finsent.metrics import classwise_metrics, confusion_matrix_df, metrics_with_optional_proba
from cara_finsent.plotting import save_confusion_matrix_plot


def main():
    parser = argparse.ArgumentParser(description='Improved FinBERT with confidence scoring.')
    parser.add_argument('--data', default=None, help='Controlled gold split CSV.')
    parser.add_argument('--dataset_name', default=None, choices=['phrasebank', 'fiqa'])
    parser.add_argument('--text_col', default=None)
    parser.add_argument('--label_col', default=None)
    parser.add_argument('--model_name', default='ProsusAI/finbert')
    parser.add_argument('--max_rows', type=int, default=None)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--epochs', type=float, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--early_stopping_patience', type=int, default=2)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--figures_dir', default='figures')
    parser.add_argument('--models_dir', default='models')
    args = parser.parse_args()

    set_global_seeds(args.seed, enable_deep_learning=True)
    dataset_name = args.dataset_name or infer_dataset_name(args.data)
    if not args.data:
        if dataset_name == 'unknown':
            raise SystemExit('Provide --dataset_name when --data is omitted. Allowed values: phrasebank, fiqa.')
        args.data = str(auto_detect_gold_split(dataset_name))
        print(f'[AUTO] data = {args.data}')
    if dataset_name == 'unknown':
        dataset_name = infer_dataset_name(args.data)
    if dataset_name == 'unknown':
        raise SystemExit('Could not infer dataset_name from --data. Pass --dataset_name explicitly.')
    
    has_cuda = torch.cuda.is_available()
    use_fp16 = args.fp16 if args.fp16 else has_cuda
    print(f'[INFO] device={torch.cuda.get_device_name(0) if has_cuda else "cpu"} seed={args.seed}')

    ts = timestamp()
    train_df, val_df, test_df = load_gold_split(args.data)
    if args.max_rows:
        print('[WARN] --max_rows is ignored for controlled gold split inputs.')

    split_source = 'controlled_gold_split'
    benchmark_mode = f'{dataset_name}_in_domain'
    label_dist = split_label_distribution(train_df, val_df, test_df)
    leakage_count = text_hash_leakage_count(train_df, val_df, test_df)
    git_sha = git_commit_sha(PROJECT_ROOT)

    print(f'[INFO] train/val/test: {len(train_df)}/{len(val_df)}/{len(test_df)}')
    
    # Log uncertainty statistics if available
    if 'is_uncertain' in train_df.columns:
        uncertain_count = train_df['is_uncertain'].sum()
        print(f'[INFO] Uncertain rows in training: {uncertain_count}/{len(train_df)} ({100*uncertain_count/len(train_df):.1f}%)')
    
    # Log agreement statistics if available
    if 'agreement' in train_df.columns:
        agreement_mean = train_df['agreement'].mean()
        print(f'[INFO] Mean agreement level: {agreement_mean:.2f}')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    native_id2label = {int(k): str(v) for k, v in dict(model.config.id2label).items()}
    canonical_remap = model_label_remap(native_id2label)
    print(f'[INFO] Native id2label: {native_id2label}')
    print(f'[INFO] Canonical remap: {canonical_remap}')

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs_native = torch.softmax(torch.tensor(logits), dim=1).numpy()
        probs_canon = remap_probs(probs_native, native_id2label)
        y_pred = [STANDARD_LABELS[int(i)] for i in probs_canon.argmax(axis=1)]
        y_true = [canonical_label(native_id2label[int(i)]) for i in labels]
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'mcc': matthews_corrcoef(y_true, y_pred),
        }

    def tokenize(examples):
        return tokenizer(examples['text'], max_length=args.max_length, truncation=True, padding='max_length')

    # Encode labels in the model native order so config.id2label remains truthful.
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df['label_id'] = encode_labels_for_model(train_df['label'], native_id2label)
    val_df['label_id'] = encode_labels_for_model(val_df['label'], native_id2label)
    test_df['label_id'] = encode_labels_for_model(test_df['label'], native_id2label)

    train_ds = Dataset.from_pandas(train_df[['text', 'label_id']]).map(tokenize, batched=True, remove_columns=['text'])
    train_ds = train_ds.rename_column('label_id', 'labels')
    val_ds = Dataset.from_pandas(val_df[['text', 'label_id']]).map(tokenize, batched=True, remove_columns=['text'])
    val_ds = val_ds.rename_column('label_id', 'labels')
    test_ds = Dataset.from_pandas(test_df[['text', 'label_id']]).map(tokenize, batched=True, remove_columns=['text'])
    test_ds = test_ds.rename_column('label_id', 'labels')

    out_dir = f'{args.models_dir}/finbert_{ts}'
    training_kwargs = dict(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=use_fp16,
        seed=args.seed,
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
    )
    try:
        training_args = TrainingArguments(eval_strategy='epoch', **training_kwargs)
    except TypeError:
        training_args = TrainingArguments(evaluation_strategy='epoch', **training_kwargs)
    
    callbacks = [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if args.early_stopping_patience > 0 else []
    
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_ds, 
        eval_dataset=val_ds, 
        compute_metrics=compute_metrics, 
        callbacks=callbacks
    )

    start = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - start
    pred_output = trainer.predict(test_ds)
    probs_native = torch.softmax(torch.tensor(pred_output.predictions), dim=1).numpy()
    probs = remap_probs(probs_native, native_id2label)
    pred_ids = probs.argmax(axis=1)
    y_pred = [STANDARD_LABELS[int(i)] for i in pred_ids]
    y_true = test_df['label'].values

    # Compute confidence scores and abstention metrics
    confidence = probs.max(axis=1)
    entropy = -(probs * np.log(probs + 1e-10)).sum(axis=1)  # Shannon entropy of predictions
    
    summary = metrics_with_optional_proba(y_true, y_pred, probs, model_name=f'finbert_{args.model_name}')
    summary['dataset_name'] = dataset_name
    summary['dataset_file'] = str(Path(args.data))
    summary['split_source'] = split_source
    summary['benchmark_mode'] = benchmark_mode
    summary['train_plus_infer_seconds'] = elapsed
    summary['train_rows'] = len(train_df)
    summary['val_rows'] = len(val_df)
    summary['test_rows'] = len(test_df)
    summary['seed'] = args.seed
    summary['device'] = 'cuda' if has_cuda else 'cpu'
    summary['native_id2label'] = json.dumps(native_id2label, sort_keys=True)
    summary['canonical_remap'] = json.dumps(canonical_remap)
    summary['label_distribution_train'] = json.dumps(label_dist['train'], sort_keys=True)
    summary['label_distribution_val'] = json.dumps(label_dist['val'], sort_keys=True)
    summary['label_distribution_test'] = json.dumps(label_dist['test'], sort_keys=True)
    summary['text_hash_leakage_count'] = leakage_count
    summary['git_commit_sha'] = git_sha
    summary['mean_confidence'] = float(confidence.mean())
    summary['mean_entropy'] = float(entropy.mean())
    summary_path = save_dataframe(pd.DataFrame([summary]), args.results_dir, 'finbert_improved_summary', ts)

    # Persist checkpoint
    try:
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)
    except Exception as exc:
        print(f'[WARN] Failed to save: {exc}')

    # Save predictions with confidence scores
    pred_df = test_df[['id', 'text', 'label']].copy()
    pred_df['dataset_name'] = dataset_name
    pred_df['split_source'] = split_source
    pred_df['prediction'] = y_pred
    pred_df['confidence'] = confidence
    pred_df['entropy'] = entropy
    
    # Add abstention decision: abstain if confidence < 0.60
    ABSTAIN_THRESHOLD = 0.60
    pred_df['abstain'] = confidence < ABSTAIN_THRESHOLD
    
    for i, cls in enumerate(STANDARD_LABELS):
        pred_df[f'proba_{cls}'] = probs[:, i]
    
    # Compute accuracy on non-abstained predictions
    non_abstained_mask = ~pred_df['abstain'].values
    if non_abstained_mask.sum() > 0:
        accuracy_non_abstained = (pred_df.loc[non_abstained_mask, 'prediction'] == pred_df.loc[non_abstained_mask, 'label']).mean()
        coverage = non_abstained_mask.sum() / len(pred_df)
        print(f'[ABSTENTION] Coverage: {coverage:.1%}, Accuracy (non-abstained): {accuracy_non_abstained:.3f}')
        summary['abstention_coverage'] = float(coverage)
        summary['abstention_accuracy'] = float(accuracy_non_abstained)
    
    pred_path = save_dataframe(pred_df, args.results_dir, 'finbert_improved_predictions', ts)
    cm = confusion_matrix_df(y_true, y_pred)
    cm_path = save_dataframe(cm.reset_index().rename(columns={'index': 'actual'}), args.results_dir, 'finbert_confusion_matrix', ts)
    cw_path = save_dataframe(classwise_metrics(y_true, y_pred), args.results_dir, 'finbert_classwise_metrics', ts)
    fig_path = f'{args.figures_dir}/finbert_confusion_matrix_{ts}.png'
    save_confusion_matrix_plot(cm, fig_path, title='FinBERT Confusion Matrix')
    metrics_path = save_json(
        {
            'trainer_metrics': pred_output.metrics,
            'native_id2label': native_id2label,
            'canonical_remap': canonical_remap,
        },
        args.results_dir,
        'finbert_trainer_metrics',
        ts,
    )
    manifest = write_manifest(
        args.results_dir,
        'finbert_improved',
        {
            'summary': str(summary_path),
            'predictions': str(pred_path),
            'confusion_matrix': str(cm_path),
            'classwise': str(cw_path),
            'trainer_metrics': str(metrics_path),
            'figure': fig_path,
            'model_dir': out_dir,
        },
        {
            'dataset_name': dataset_name,
            'dataset_file': str(Path(args.data)),
            'split_source': split_source,
            'benchmark_mode': benchmark_mode,
            'train_rows': len(train_df),
            'val_rows': len(val_df),
            'test_rows': len(test_df),
            'label_distribution': label_dist,
            'text_hash_leakage_count': leakage_count,
            'seed': args.seed,
            'model_name': args.model_name,
            'native_id2label': native_id2label,
            'canonical_remap': canonical_remap,
            'git_commit_sha': git_sha,
        },
        ts,
    )
    
    print(pd.DataFrame([summary]))
    print(f'[DONE] Manifest -> {manifest}')


if __name__ == '__main__':
    main()
