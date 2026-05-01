#!/usr/bin/env python3
"""
11b_eval_finbert_checkpoint.py
---
Evaluate a saved FinBERT checkpoint on the test split.
Produces the same output format as 11_run_finbert_improved_v2.py.

Usage:
    python scripts/11b_eval_finbert_checkpoint.py --checkpoint models/finbert_20260501_092839/checkpoint-1005
"""
from __future__ import annotations

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

from cara_finsent.data_utils import STANDARD_LABELS, decode_labels, load_standardized_csv, set_global_seeds, split_dataframe
from cara_finsent.io_utils import save_dataframe, save_json, timestamp, write_manifest
from cara_finsent.metrics import classwise_metrics, confusion_matrix_df, metrics_with_optional_proba
from cara_finsent.plotting import save_confusion_matrix_plot

LABEL2ID = {label: i for i, label in enumerate(STANDARD_LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def main():
    parser = argparse.ArgumentParser(description='Evaluate saved FinBERT checkpoint on test split.')
    parser.add_argument('--checkpoint', required=True, help='Path to saved checkpoint directory.')
    parser.add_argument('--data', default=None)
    parser.add_argument('--text_col', default=None)
    parser.add_argument('--label_col', default=None)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--figures_dir', default='figures')
    args = parser.parse_args()

    set_global_seeds(args.seed, enable_deep_learning=True)

    if not args.data:
        from cara_finsent.data_utils import auto_detect_data
        args.data = str(auto_detect_data())
        print(f'[AUTO] data = {args.data}')

    has_cuda = torch.cuda.is_available()
    print(f'[INFO] device={"cuda" if has_cuda else "cpu"}, checkpoint={args.checkpoint}')

    ts = timestamp()
    df = load_standardized_csv(args.data, args.text_col, args.label_col)
    _train_df, _val_df, test_df = split_dataframe(df, seed=args.seed)
    print(f'[INFO] Test rows: {len(test_df)}')

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    def tokenize(examples):
        return tokenizer(examples['text'], max_length=args.max_length, truncation=True, padding='max_length')

    mapping = {label: i for i, label in enumerate(STANDARD_LABELS)}
    test_df = test_df.copy()
    test_df['label_id'] = test_df['label'].map(mapping)
    test_ds = Dataset.from_pandas(test_df[['text', 'label_id']]).map(tokenize, batched=True, remove_columns=['text'])
    test_ds = test_ds.rename_column('label_id', 'labels')

    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint, num_labels=len(STANDARD_LABELS), id2label=ID2LABEL, label2id=LABEL2ID)

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
    logits = pred_output.predictions
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    pred_ids = probs.argmax(axis=1)
    y_pred = decode_labels(pred_ids)
    y_true = test_df['label'].values

    confidence = probs.max(axis=1)
    entropy = -(probs * np.log(probs + 1e-10)).sum(axis=1)

    summary = metrics_with_optional_proba(y_true, y_pred, probs, model_name=f'finbert_v2_epoch1_{args.checkpoint}')
    summary['test_rows'] = len(test_df)
    summary['seed'] = args.seed
    summary['device'] = 'cuda' if has_cuda else 'cpu'
    summary['checkpoint'] = args.checkpoint
    summary['mean_confidence'] = float(confidence.mean())
    summary['mean_entropy'] = float(entropy.mean())

    ABSTAIN_THRESHOLD = 0.60
    pred_df = test_df[['id', 'text', 'label']].copy()
    pred_df['prediction'] = y_pred
    pred_df['confidence'] = confidence
    pred_df['entropy'] = entropy
    pred_df['abstain'] = confidence < ABSTAIN_THRESHOLD

    for i, cls in ID2LABEL.items():
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
        {'data': args.data}, ts
    )

    print(pd.DataFrame([summary]))
    print(f'[DONE] Manifest -> {manifest}')


if __name__ == '__main__':
    main()
