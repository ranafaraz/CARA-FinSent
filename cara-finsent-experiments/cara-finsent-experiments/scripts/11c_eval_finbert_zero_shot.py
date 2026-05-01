#!/usr/bin/env python3
"""
11c_eval_finbert_zero_shot.py
---
Zero-shot evaluation of the base ProsusAI/finbert model on the canonical test split.
No fine-tuning — uses the pretrained model directly with correct label remapping.

ProsusAI/finbert native label order: {0:'positive', 1:'negative', 2:'neutral'}
Our STANDARD_LABELS order:           {0:'negative', 1:'neutral',  2:'positive'}
This script remaps predictions to STANDARD_LABELS before computing metrics.

Output files use the 'finbert_baseline_summary' prefix so script 17 picks them up.

Usage:
    python scripts/11c_eval_finbert_zero_shot.py
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
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from cara_finsent.data_utils import STANDARD_LABELS, decode_labels, load_standardized_csv, set_global_seeds, split_dataframe
from cara_finsent.io_utils import save_dataframe, save_json, timestamp, write_manifest
from cara_finsent.metrics import classwise_metrics, confusion_matrix_df, metrics_with_optional_proba
from cara_finsent.plotting import save_confusion_matrix_plot

# ProsusAI/finbert native label order (pretrained)
PROSUS_ID2LABEL = {0: 'positive', 1: 'negative', 2: 'neutral'}

# Our standard label order
STANDARD_LABEL2ID = {label: i for i, label in enumerate(STANDARD_LABELS)}
STANDARD_ID2LABEL = {i: label for label, i in STANDARD_LABEL2ID.items()}

# Remap: ProsusAI output index → STANDARD_LABELS index
# prosus[0]=positive → standard[2], prosus[1]=negative → standard[0], prosus[2]=neutral → standard[1]
PROSUS_TO_STANDARD = np.array([2, 0, 1])  # prosus_probs[:, [1,2,0]] gives [neg,neu,pos]


def main():
    parser = argparse.ArgumentParser(description='Zero-shot evaluation of base ProsusAI/finbert.')
    parser.add_argument('--model_name', default='ProsusAI/finbert')
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
    device = 'cuda' if has_cuda else 'cpu'
    print(f'[INFO] model={args.model_name}, device={device}, seed={args.seed}')

    ts = timestamp()
    df = load_standardized_csv(args.data, args.text_col, args.label_col)
    
    # Check if pre-split column exists; if so, use it (Phase 1 controlled splits)
    if 'split' in df.columns:
        print(f'[INFO] Using pre-existing splits from {args.data}', flush=True)
        test_df = df[df['split'] == 'test'].reset_index(drop=True)
        print(f'[INFO] Pre-split test rows: {len(test_df)}', flush=True)
    else:
        print(f'[INFO] No pre-split column; creating splits with seed={args.seed}', flush=True)
        _train_df, _val_df, test_df = split_dataframe(df, seed=args.seed)
        print(f'[INFO] Test rows: {len(test_df)}')

    # Load base model with its ORIGINAL ProsusAI label mapping (no override)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    print(f'[INFO] Native id2label: {model.config.id2label}')

    def tokenize(examples):
        return tokenizer(examples['text'], max_length=args.max_length, truncation=True, padding='max_length')

    # Build test dataset (labels encoded in STANDARD_LABELS order for metric computation)
    test_df = test_df.copy()
    test_df['label_id'] = test_df['label'].map(STANDARD_LABEL2ID)
    test_ds = Dataset.from_pandas(test_df[['text', 'label_id']]).map(tokenize, batched=True, remove_columns=['text'])
    test_ds = test_ds.rename_column('label_id', 'labels')

    eval_args = TrainingArguments(
        output_dir='_tmp_eval_zero_shot',
        per_device_eval_batch_size=args.batch_size,
        fp16=False,
        seed=args.seed,
        report_to='none',
    )

    trainer = Trainer(model=model, args=eval_args)
    pred_output = trainer.predict(test_ds)

    # Logits are in ProsusAI's native order: [positive, negative, neutral]
    logits = pred_output.predictions
    prosus_probs = torch.softmax(torch.tensor(logits), dim=1).numpy()

    # Remap to STANDARD_LABELS order: [negative, neutral, positive]
    # prosus_probs[:, 1] = P(negative), prosus_probs[:, 2] = P(neutral), prosus_probs[:, 0] = P(positive)
    probs = prosus_probs[:, [1, 2, 0]]  # shape (N, 3) in [neg, neu, pos] order

    pred_ids = probs.argmax(axis=1)
    y_pred = decode_labels(pred_ids)
    y_true = test_df['label'].values

    confidence = probs.max(axis=1)
    entropy = -(probs * np.log(probs + 1e-10)).sum(axis=1)

    summary = metrics_with_optional_proba(y_true, y_pred, probs, model_name=f'finbert_base_zero_shot_{args.model_name}')
    summary['test_rows'] = len(test_df)
    summary['seed'] = args.seed
    summary['device'] = device
    summary['model_name'] = args.model_name
    summary['mean_confidence'] = float(confidence.mean())
    summary['mean_entropy'] = float(entropy.mean())

    ABSTAIN_THRESHOLD = 0.60
    pred_df = test_df[['id', 'text', 'label']].copy()
    pred_df['prediction'] = y_pred
    pred_df['confidence'] = confidence
    pred_df['entropy'] = entropy
    pred_df['abstain'] = confidence < ABSTAIN_THRESHOLD

    for i, cls in STANDARD_ID2LABEL.items():
        pred_df[f'proba_{cls}'] = probs[:, i]

    non_abstained_mask = ~pred_df['abstain'].values
    if non_abstained_mask.sum() > 0:
        accuracy_non_abstained = (pred_df.loc[non_abstained_mask, 'prediction'] == pred_df.loc[non_abstained_mask, 'label']).mean()
        coverage = non_abstained_mask.sum() / len(pred_df)
        print(f'[ABSTENTION] Coverage: {coverage:.1%}, Accuracy (non-abstained): {accuracy_non_abstained:.3f}')
        summary['abstention_coverage'] = float(coverage)
        summary['abstention_accuracy'] = float(accuracy_non_abstained)

    # Use 'finbert_baseline_summary' prefix so script 17 picks it up
    summary_path = save_dataframe(pd.DataFrame([summary]), args.results_dir, 'finbert_baseline_summary', ts)
    pred_path = save_dataframe(pred_df, args.results_dir, 'finbert_baseline_predictions', ts)
    cm = confusion_matrix_df(y_true, y_pred)
    cm_path = save_dataframe(cm.reset_index().rename(columns={'index': 'actual'}), args.results_dir, 'finbert_confusion_matrix', ts)
    cw_path = save_dataframe(classwise_metrics(y_true, y_pred), args.results_dir, 'finbert_classwise_metrics', ts)
    fig_path = f'{args.figures_dir}/finbert_confusion_matrix_{ts}.png'
    save_confusion_matrix_plot(cm, fig_path, title='FinBERT Zero-Shot Confusion Matrix')
    metrics_path = save_json({'native_id2label': model.config.id2label, 'test_rows': len(test_df)},
                             args.results_dir, 'finbert_trainer_metrics', ts)
    manifest = write_manifest(
        args.results_dir, 'finbert_baseline',
        {'summary': str(summary_path), 'predictions': str(pred_path), 'confusion_matrix': str(cm_path),
         'classwise': str(cw_path), 'trainer_metrics': str(metrics_path), 'figure': fig_path,
         'model': args.model_name},
        {'data': args.data}, ts
    )

    print(pd.DataFrame([summary]))
    print(f'[DONE] Manifest -> {manifest}')


if __name__ == '__main__':
    main()
