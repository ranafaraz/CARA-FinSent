#!/usr/bin/env python3
"""Phase 14 Step 2 — Generate validation + test prediction artefacts.

Produces per-id probability vectors on BOTH the validation split and the
test split for one of:

    --model_family zero_shot         (ProsusAI/finbert, no training)
    --model_family fine_tuned        (local checkpoint dir)
    --model_family agreement_weighted (local checkpoint dir)

Output schema (Phase 14 mandate):
    text_hash, split, true_label, pred_label,
    proba_negative, proba_neutral, proba_positive, confidence,
    model_family, model_name, checkpoint_path,
    git_commit_sha, generated_at_utc

Two CSVs per run (val + test) plus a manifest. Files are written under
``results/<date>/phase14_predictions/`` and mirrored to
``artifacts/phase14_corrected_validation/``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from cara_finsent.data_utils import STANDARD_LABELS, load_gold_split, set_global_seeds
from cara_finsent.io_utils import git_commit_sha, timestamp
from cara_finsent.label_mapping import model_label_remap, remap_probs

STANDARD_LABEL2ID = {label: i for i, label in enumerate(STANDARD_LABELS)}


def _build_ds(df, tokenizer, max_length):
    df = df.copy()
    df['label_id'] = df['label'].map(STANDARD_LABEL2ID)
    cols = ['text', 'label_id']
    ds = Dataset.from_pandas(df[cols]).map(
        lambda b: tokenizer(b['text'], max_length=max_length, truncation=True, padding='max_length'),
        batched=True, remove_columns=['text'],
    )
    return ds.rename_column('label_id', 'labels')


def _predict(model, ds, batch_size, seed):
    """Plain torch eval loop — bypasses HF Trainer to avoid silent loss-eval bugs."""
    torch.manual_seed(int(seed))
    model.eval()
    keep_cols = ['input_ids', 'attention_mask']
    if 'token_type_ids' in ds.column_names:
        keep_cols.append('token_type_ids')
    drop = [c for c in ds.column_names if c not in keep_cols]
    ds_eval = ds.remove_columns(drop)
    ds_eval.set_format(type='torch', columns=keep_cols)
    out_chunks = []
    n = len(ds_eval)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = {k: ds_eval[k][start:start + batch_size] for k in keep_cols}
            logits = model(**batch).logits
            out_chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out_chunks, axis=0)


def _to_csv_frame(df_split, probs, split_name, model_family, model_name, checkpoint_path, git_sha, ts):
    pred_ids = probs.argmax(axis=1)
    out = pd.DataFrame({
        'text_hash': df_split['text_hash'].astype(str).values,
        'split': split_name,
        'true_label': df_split['label'].astype(str).values,
        'pred_label': [STANDARD_LABELS[int(i)] for i in pred_ids],
        'proba_negative': probs[:, 0],
        'proba_neutral': probs[:, 1],
        'proba_positive': probs[:, 2],
        'confidence': probs.max(axis=1),
        'model_family': model_family,
        'model_name': model_name,
        'checkpoint_path': str(checkpoint_path) if checkpoint_path else '',
        'git_commit_sha': git_sha,
        'generated_at_utc': ts,
    })
    return out


def main():
    parser = argparse.ArgumentParser(description='Phase 14 Step 2 - generate val/test predictions.')
    parser.add_argument('--model_family', required=True,
                        choices=['zero_shot', 'fine_tuned', 'agreement_weighted'])
    parser.add_argument('--checkpoint', default=None,
                        help='Local checkpoint dir (required for fine_tuned and agreement_weighted) '
                             'or HF id (only for zero_shot).')
    parser.add_argument('--data', default='data/processed/gold/latest_gold_phrasebank_split.csv')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--results_dir', default='results')
    args = parser.parse_args()

    set_global_seeds(args.seed, enable_deep_learning=True)
    ts = timestamp()
    git_sha = git_commit_sha(PROJECT_ROOT)
    date_subdir = f'{ts[:4]}-{ts[4:6]}-{ts[6:8]}'

    if args.model_family == 'zero_shot':
        model_id = args.checkpoint or 'ProsusAI/finbert'
        model_name = model_id
        checkpoint_path = ''
        family_label = 'finbert_zero_shot'
    elif args.model_family == 'fine_tuned':
        if not args.checkpoint:
            raise SystemExit('--checkpoint is required for --model_family fine_tuned')
        model_id = args.checkpoint
        model_name = 'finbert_fine_tuned'
        checkpoint_path = args.checkpoint
        family_label = 'finbert_fine_tuned'
    else:  # agreement_weighted
        if not args.checkpoint:
            raise SystemExit('--checkpoint is required for --model_family agreement_weighted')
        model_id = args.checkpoint
        model_name = 'finbert_agreement_weighted'
        checkpoint_path = args.checkpoint
        family_label = 'finbert_agreement_weighted'

    train_df, val_df, test_df = load_gold_split(args.data)
    print(f'[INFO] data={args.data} val_n={len(val_df)} test_n={len(test_df)}')

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Sanity check: a checkpoint dir without vocab files silently returns
        # a degenerate BertTokenizer with vocab_size=5, which produces all-pad
        # token IDs and degenerate predictions. Detect and fall back.
        if tokenizer.vocab_size < 1000:
            raise ValueError(
                f'tokenizer at {model_id} has vocab_size={tokenizer.vocab_size}; '
                'falling back to ProsusAI/finbert.'
            )
    except (OSError, ValueError) as exc:
        print(f'[WARN] tokenizer not usable from {model_id} ({exc.__class__.__name__}: {exc}); falling back to ProsusAI/finbert')
        tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    native_id2label = {int(k): str(v) for k, v in dict(model.config.id2label).items()}
    canonical_remap = model_label_remap(native_id2label)
    print(f'[INFO] native_id2label={native_id2label} canonical_remap={canonical_remap}')

    val_ds = _build_ds(val_df, tokenizer, args.max_length)
    test_ds = _build_ds(test_df, tokenizer, args.max_length)

    probs_val_native = _predict(model, val_ds, args.batch_size, args.seed)
    probs_test_native = _predict(model, test_ds, args.batch_size, args.seed)

    probs_val = remap_probs(probs_val_native, native_id2label)
    probs_test = remap_probs(probs_test_native, native_id2label)

    out_dir = PROJECT_ROOT / args.results_dir / date_subdir / 'phase14_predictions'
    out_dir.mkdir(parents=True, exist_ok=True)
    val_path = out_dir / f'{family_label}_val_predictions_{ts}.csv'
    test_path = out_dir / f'{family_label}_test_predictions_{ts}.csv'

    val_frame = _to_csv_frame(val_df, probs_val, 'val', family_label, model_name, checkpoint_path, git_sha, ts)
    test_frame = _to_csv_frame(test_df, probs_test, 'test', family_label, model_name, checkpoint_path, git_sha, ts)
    val_frame.to_csv(val_path, index=False)
    test_frame.to_csv(test_path, index=False)
    print(f'[OUT] {val_path}')
    print(f'[OUT] {test_path}')

    val_acc = float((val_frame['pred_label'] == val_frame['true_label']).mean())
    test_acc = float((test_frame['pred_label'] == test_frame['true_label']).mean())
    print(f'[INFO] val_acc={val_acc:.4f} test_acc={test_acc:.4f}')

    manifest = {
        'phase': 'phase14_step2_val_test_predictions',
        'model_family': args.model_family,
        'model_id': model_id,
        'checkpoint_path': str(checkpoint_path),
        'data': args.data,
        'val_rows': int(len(val_df)),
        'test_rows': int(len(test_df)),
        'native_id2label': native_id2label,
        'canonical_remap': canonical_remap,
        'val_accuracy': val_acc,
        'test_accuracy': test_acc,
        'seed': int(args.seed),
        'max_length': int(args.max_length),
        'batch_size': int(args.batch_size),
        'val_predictions': str(val_path),
        'test_predictions': str(test_path),
        'git_commit_sha': git_sha,
        'generated_at_utc': ts,
    }
    manifest_path = out_dir / f'prediction_manifest_{family_label}_{ts}.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f'[OUT] {manifest_path}')

    artifacts_dir = PROJECT_ROOT / 'artifacts' / 'phase14_corrected_validation' / 'predictions'
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    val_frame.to_csv(artifacts_dir / val_path.name, index=False)
    test_frame.to_csv(artifacts_dir / test_path.name, index=False)
    (artifacts_dir / manifest_path.name).write_text(json.dumps(manifest, indent=2))
    print(f'[OUT] safe artefacts in {artifacts_dir}')


if __name__ == '__main__':
    main()
