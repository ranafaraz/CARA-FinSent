#!/usr/bin/env python3
"""Evaluate a saved FinBERT checkpoint on a controlled gold split."""
from __future__ import annotations

import json
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / '.env', override=False)
except ImportError:
    pass

import os
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

import argparse
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

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
from cara_finsent.label_mapping import encode_labels_for_model, model_label_remap, remap_probs
from cara_finsent.metrics import classwise_metrics, confusion_matrix_df, metrics_with_optional_proba
from cara_finsent.plotting import save_confusion_matrix_plot


def main():
    parser = argparse.ArgumentParser(description='Evaluate saved FinBERT checkpoint on test split.')
    parser.add_argument('--checkpoint', required=True, help='Path to saved checkpoint directory.')
    parser.add_argument('--data', default=None)
    parser.add_argument('--dataset_name', default=None, choices=['phrasebank', 'fiqa'])
    parser.add_argument('--text_col', default=None)
    parser.add_argument('--label_col', default=None)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--figures_dir', default='figures')
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
    print(f'[INFO] device={"cuda" if has_cuda else "cpu"}, checkpoint={args.checkpoint}')

    ts = timestamp()
    train_df, val_df, test_df = load_gold_split(args.data)
    split_source = 'controlled_gold_split'
    benchmark_mode = f'{dataset_name}_in_domain'
    label_dist = split_label_distribution(train_df, val_df, test_df)
    leakage_count = text_hash_leakage_count(train_df, val_df, test_df)
    git_sha = git_commit_sha(PROJECT_ROOT)
    print(f'[INFO] Controlled gold split rows: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}')

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
    native_id2label = {int(k): str(v) for k, v in dict(model.config.id2label).items()}
    canonical_remap = model_label_remap(native_id2label)
    print(f'[INFO] Native id2label: {native_id2label}')
    print(f'[INFO] Canonical remap: {canonical_remap}')

    def tokenize(examples):
        return tokenizer(examples['text'], max_length=args.max_length, truncation=True, padding='max_length')

    test_df = test_df.copy()
    test_df['label_id'] = encode_labels_for_model(test_df['label'], native_id2label)
    test_ds = Dataset.from_pandas(test_df[['text', 'label_id']]).map(tokenize, batched=True, remove_columns=['text'])
    test_ds = test_ds.rename_column('label_id', 'labels')

    try:
        training_args = TrainingArguments(
            output_dir='_tmp_eval',
            per_device_eval_batch_size=args.batch_size,
            fp16=False,
            seed=args.seed,
            no_cuda=not has_cuda,
        )
    except TypeError:
        training_args = TrainingArguments(
            output_dir='_tmp_eval',
            per_device_eval_batch_size=args.batch_size,
            fp16=False,
            seed=args.seed,
        )

    trainer = Trainer(model=model, args=training_args)
    pred_output = trainer.predict(test_ds)
    probs_native = torch.softmax(torch.tensor(pred_output.predictions), dim=1).numpy()
    probs = remap_probs(probs_native, native_id2label)
    pred_ids = probs.argmax(axis=1)
    y_pred = [STANDARD_LABELS[int(i)] for i in pred_ids]
    y_true = test_df['label'].values

    confidence = probs.max(axis=1)
    entropy = -(probs * np.log(probs + 1e-10)).sum(axis=1)

    summary = metrics_with_optional_proba(y_true, y_pred, probs, model_name=f'finbert_v2_epoch1_{args.checkpoint}')
    summary['dataset_name'] = dataset_name
    summary['dataset_file'] = str(Path(args.data))
    summary['split_source'] = split_source
    summary['benchmark_mode'] = benchmark_mode
    summary['train_rows'] = len(train_df)
    summary['val_rows'] = len(val_df)
    summary['test_rows'] = len(test_df)
    summary['seed'] = args.seed
    summary['device'] = 'cuda' if has_cuda else 'cpu'
    summary['checkpoint'] = args.checkpoint
    summary['native_id2label'] = json.dumps(native_id2label, sort_keys=True)
    summary['canonical_remap'] = json.dumps(canonical_remap)
    summary['label_distribution_train'] = json.dumps(label_dist['train'], sort_keys=True)
    summary['label_distribution_val'] = json.dumps(label_dist['val'], sort_keys=True)
    summary['label_distribution_test'] = json.dumps(label_dist['test'], sort_keys=True)
    summary['text_hash_leakage_count'] = leakage_count
    summary['git_commit_sha'] = git_sha
    summary['mean_confidence'] = float(confidence.mean())
    summary['mean_entropy'] = float(entropy.mean())

    ABSTAIN_THRESHOLD = 0.60
    pred_df = test_df[['id', 'text', 'label']].copy()
    pred_df['dataset_name'] = dataset_name
    pred_df['split_source'] = split_source
    pred_df['prediction'] = y_pred
    pred_df['confidence'] = confidence
    pred_df['entropy'] = entropy
    pred_df['abstain'] = confidence < ABSTAIN_THRESHOLD

    for i, cls in enumerate(STANDARD_LABELS):
        pred_df[f'proba_{cls}'] = probs[:, i]

    non_abstained_mask = ~pred_df['abstain'].values
    if non_abstained_mask.sum() > 0:
        accuracy_non_abstained = (pred_df.loc[non_abstained_mask, 'prediction'] == pred_df.loc[non_abstained_mask, 'label']).mean()
        coverage = non_abstained_mask.sum() / len(pred_df)
        print(f'[ABSTENTION] Coverage: {coverage:.1%}, Accuracy (non-abstained): {accuracy_non_abstained:.3f}')
        summary['abstention_coverage'] = float(coverage)
        summary['abstention_accuracy'] = float(accuracy_non_abstained)

    summary_path = save_dataframe(pd.DataFrame([summary]), args.results_dir, 'finbert_improved_summary', ts)
    pred_path = save_dataframe(pred_df, args.results_dir, 'finbert_improved_predictions', ts)
    cm = confusion_matrix_df(y_true, y_pred)
    cm_path = save_dataframe(cm.reset_index().rename(columns={'index': 'actual'}), args.results_dir, 'finbert_confusion_matrix', ts)
    cw_path = save_dataframe(classwise_metrics(y_true, y_pred), args.results_dir, 'finbert_classwise_metrics', ts)
    fig_path = f'{args.figures_dir}/finbert_confusion_matrix_{ts}.png'
    save_confusion_matrix_plot(cm, fig_path, title='FinBERT v2 Confusion Matrix')
    metrics_path = save_json(pred_output.metrics, args.results_dir, 'finbert_trainer_metrics', ts)
    manifest = write_manifest(
        args.results_dir, 'finbert_improved',
        {'summary': str(summary_path), 'predictions': str(pred_path), 'confusion_matrix': str(cm_path),
         'classwise': str(cw_path), 'trainer_metrics': str(metrics_path), 'figure': fig_path,
         'checkpoint': args.checkpoint},
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
            'model_name': str(args.checkpoint),
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
