#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import argparse
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from cara_finsent.data_utils import STANDARD_LABELS, apply_max_rows, decode_labels, encode_labels, load_standardized_csv, split_dataframe
from cara_finsent.io_utils import save_dataframe, save_json, timestamp, write_manifest
from cara_finsent.metrics import classwise_metrics, confusion_matrix_df, metrics_with_optional_proba
from cara_finsent.plotting import save_confusion_matrix_plot

LABEL2ID = {label: i for i, label in enumerate(STANDARD_LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    y_true = decode_labels(labels)
    y_pred = decode_labels(preds)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
    }


def main():
    parser = argparse.ArgumentParser(description='Fine-tune/evaluate FinBERT-style transformer baseline.')
    parser.add_argument('--data', required=True)
    parser.add_argument('--text_col', default=None)
    parser.add_argument('--label_col', default=None)
    parser.add_argument('--model_name', default='ProsusAI/finbert')
    parser.add_argument('--max_rows', type=int, default=None)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--epochs', type=float, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--figures_dir', default='figures')
    parser.add_argument('--models_dir', default='models')
    args = parser.parse_args()

    ts = timestamp()
    df = load_standardized_csv(args.data, args.text_col, args.label_col)
    df = apply_max_rows(df, args.max_rows)
    train_df, val_df, test_df = split_dataframe(df)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=3, label2id=LABEL2ID, id2label=ID2LABEL, ignore_mismatched_sizes=True)

    def to_dataset(frame):
        ds = Dataset.from_pandas(pd.DataFrame({'text': frame['text'].tolist(), 'label': encode_labels(frame['label'])}))
        def tok(batch):
            return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=args.max_length)
        return ds.map(tok, batched=True).remove_columns(['text']).with_format('torch')

    train_ds, val_ds, test_ds = to_dataset(train_df), to_dataset(val_df), to_dataset(test_df)
    out_dir = f'{args.models_dir}/finbert_{ts}'
    training_kwargs = dict(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        logging_steps=25,
        report_to='none',
    )
    try:
        training_args = TrainingArguments(eval_strategy='epoch', **training_kwargs)
    except TypeError:
        training_args = TrainingArguments(evaluation_strategy='epoch', **training_kwargs)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_metrics)

    start = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - start
    pred_output = trainer.predict(test_ds)
    logits = pred_output.predictions
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    pred_ids = probs.argmax(axis=1)
    y_pred = decode_labels(pred_ids)
    y_true = test_df['label'].values

    summary = metrics_with_optional_proba(y_true, y_pred, probs, model_name=f'finbert_finetuned_{args.model_name}')
    summary['train_plus_infer_seconds'] = elapsed
    summary['train_rows'] = len(train_df)
    summary['test_rows'] = len(test_df)
    summary_path = save_dataframe(pd.DataFrame([summary]), args.results_dir, 'finbert_baseline_summary', ts)

    pred_df = test_df[['id', 'text', 'label']].copy()
    pred_df['prediction'] = y_pred
    for i, cls in ID2LABEL.items():
        pred_df[f'proba_{cls}'] = probs[:, i]
    pred_path = save_dataframe(pred_df, args.results_dir, 'finbert_baseline_predictions', ts)
    cm = confusion_matrix_df(y_true, y_pred)
    cm_path = save_dataframe(cm.reset_index().rename(columns={'index': 'actual'}), args.results_dir, 'finbert_confusion_matrix', ts)
    cw_path = save_dataframe(classwise_metrics(y_true, y_pred), args.results_dir, 'finbert_classwise_metrics', ts)
    fig_path = f'{args.figures_dir}/finbert_confusion_matrix_{ts}.png'
    save_confusion_matrix_plot(cm, fig_path, title='FinBERT Confusion Matrix')
    metrics_path = save_json(pred_output.metrics, args.results_dir, 'finbert_trainer_metrics', ts)
    manifest = write_manifest(args.results_dir, 'finbert_baseline', {'summary': str(summary_path), 'predictions': str(pred_path), 'confusion_matrix': str(cm_path), 'classwise': str(cw_path), 'trainer_metrics': str(metrics_path), 'figure': fig_path, 'model_dir': out_dir}, {'data': args.data}, ts)
    print(pd.DataFrame([summary]))
    print(f'[DONE] Manifest -> {manifest}')


if __name__ == '__main__':
    main()
