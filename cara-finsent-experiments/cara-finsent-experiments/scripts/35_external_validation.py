#!/usr/bin/env python3
"""Phase 13 Step 2 — External validation on FiQA gold split.

Runs FinBERT zero-shot and one or more local fine-tuned FinBERT checkpoints
against the FiQA controlled gold test split, reporting accuracy, macro_f1,
weighted_f1, mcc, ECE@10, brier, and mean confidence. Outputs predictions,
classwise metrics, confusion matrices, a summary CSV, and a manifest.

Honest caveats (recorded in the manifest):
  * Agreement-Weighted FinBERT is NOT evaluated here because no AW checkpoint
    is on disk (Kaggle-trained, weights not synced back).
  * Vanilla FT results below are per-checkpoint, not seed-averaged in the same
    way as the in-domain leaderboard.

Usage (CPU OK):
    python scripts/35_external_validation.py \\
        --data data/processed/gold/latest_gold_fiqa_split.csv \\
        --ft_ckpts models/finbert_20260430_215910,models/finbert_20260501_090136,models/finbert_20260501_193637
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from cara_finsent.data_utils import (
    STANDARD_LABELS,
    load_gold_split,
    set_global_seeds,
    text_hash_leakage_count,
)
from cara_finsent.io_utils import (
    git_commit_sha,
    save_dataframe,
    save_json,
    timestamp,
    write_manifest,
)
from cara_finsent.label_mapping import model_label_remap, remap_probs
from cara_finsent.metrics import (
    classwise_metrics,
    confusion_matrix_df,
    metrics_with_optional_proba,
)
from cara_finsent.plotting import save_confusion_matrix_plot


def predict_logits(model_path: str, texts, batch_size: int, max_length: int):
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    native_id2label = {int(k): str(v) for k, v in dict(model.config.id2label).items()}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    all_logits = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(device)
            logits = model(**enc).logits.detach().cpu().numpy()
            all_logits.append(logits)
    logits = np.concatenate(all_logits, axis=0)
    return logits, native_id2label


def evaluate_one(model_label: str, model_path: str, test_df, args, ts: str, results_dir: str, figures_dir: str, dataset_name: str, leakage_count: int, git_sha: str):
    print(f'\n[INFO] Evaluating {model_label} ({model_path})')
    start = time.perf_counter()
    logits, native_id2label = predict_logits(model_path, test_df['text'].tolist(), args.batch_size, args.max_length)
    elapsed = time.perf_counter() - start

    canonical_remap = model_label_remap(native_id2label)
    probs_native = torch.softmax(torch.tensor(logits), dim=1).numpy()
    probs = remap_probs(probs_native, native_id2label)
    pred_ids = probs.argmax(axis=1)
    y_pred = [STANDARD_LABELS[int(i)] for i in pred_ids]
    y_true = test_df['label'].astype(str).values

    summary = metrics_with_optional_proba(y_true, y_pred, probs, model_name=model_label)
    summary['dataset_name'] = dataset_name
    summary['eval_split'] = 'test'
    summary['n_samples'] = int(len(test_df))
    summary['model_path'] = str(model_path)
    summary['inference_seconds'] = float(elapsed)
    summary['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    summary['native_id2label'] = json.dumps(native_id2label, sort_keys=True)
    summary['canonical_remap'] = json.dumps(canonical_remap)
    summary['text_hash_leakage_count'] = int(leakage_count)
    summary['git_commit_sha'] = git_sha
    summary['generated_at_utc'] = ts
    summary['label_polarity_corrected'] = bool(getattr(args, 'swap_pos_neg', False))

    pred_df = test_df[['id', 'text', 'label']].copy()
    pred_df['dataset_name'] = dataset_name
    pred_df['split_source'] = 'controlled_gold_split'
    pred_df['model_label'] = model_label
    pred_df['prediction'] = y_pred
    for i, cls in enumerate(STANDARD_LABELS):
        pred_df[f'proba_{cls}'] = probs[:, i]

    safe_label = model_label.replace('/', '_').replace(' ', '_')
    pred_path = save_dataframe(pred_df, results_dir, f'phase13_external_{safe_label}_predictions', ts)
    cm = confusion_matrix_df(y_true, y_pred)
    cm_path = save_dataframe(cm.reset_index().rename(columns={'index': 'actual'}), results_dir, f'phase13_external_{safe_label}_confusion_matrix', ts)
    cw_path = save_dataframe(classwise_metrics(y_true, y_pred), results_dir, f'phase13_external_{safe_label}_classwise', ts)
    fig_path = f'{figures_dir}/phase13_external_{safe_label}_confusion_matrix_{ts}.png'
    try:
        save_confusion_matrix_plot(cm, fig_path, title=f'External (FiQA) — {model_label}')
    except Exception as exc:
        print(f'[WARN] Could not write confusion matrix figure: {exc}')
        fig_path = ''

    return summary, {
        'predictions': pred_path,
        'confusion_matrix': cm_path,
        'classwise': cw_path,
        'figure': fig_path,
    }


def main():
    parser = argparse.ArgumentParser(description='Phase 13 — External validation on FiQA.')
    parser.add_argument('--data', default='data/processed/gold/latest_gold_fiqa_split.csv',
                        help='Controlled gold split CSV (FiQA).')
    parser.add_argument('--zero_shot_model', default='ProsusAI/finbert',
                        help='HuggingFace model name for the zero-shot baseline.')
    parser.add_argument('--ft_ckpts', default='',
                        help='Comma-separated list of local fine-tuned checkpoint dirs to evaluate.')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--figures_dir', default='figures')
    parser.add_argument('--out_subdir', default='', help='Optional date subdir under results/ and figures/.')
    parser.add_argument('--swap_pos_neg', dest='swap_pos_neg', action='store_true',
                        help='Swap positive<->negative gold labels before scoring. '
                             'REQUIRED for FiQA gold split which is known to have inverted polarity.')
    parser.add_argument('--no_swap_pos_neg', dest='swap_pos_neg', action='store_false')
    parser.set_defaults(swap_pos_neg=True)
    args = parser.parse_args()

    set_global_seeds(args.seed, enable_deep_learning=True)
    ts = timestamp()
    git_sha = git_commit_sha(PROJECT_ROOT)

    # io_utils.save_dataframe already prepends a YYYY-MM-DD subdir from ts,
    # so we pass the plain top-level dirs and only mirror the date for figures.
    results_dir = args.results_dir
    date_part = ts[:8]
    date_subdir = f'{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}'
    figures_dir = str(Path(args.figures_dir) / date_subdir)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_gold_split(args.data)
    dataset_name = 'fiqa'
    leakage_count = text_hash_leakage_count(train_df, val_df, test_df)

    if args.swap_pos_neg:
        # Phase 13 finding: FiQA gold split has positive/negative inverted at source
        # (see docs/PHASE13_FIQA_LABEL_POLARITY_BUG.md). We swap on-the-fly so that
        # FinBERT's natural sentiment polarity is scored against the corrected labels.
        # The polarity correction is recorded in every output artefact.
        swap_map = {'positive': 'negative', 'negative': 'positive'}
        before = dict(test_df['label'].value_counts())
        test_df = test_df.copy()
        test_df['label'] = test_df['label'].map(lambda x: swap_map.get(x, x))
        after = dict(test_df['label'].value_counts())
        print(f'[POLARITY] Applied positive<->negative swap on FiQA test labels. '
              f'before={before} after={after}')
    print(f'[INFO] dataset={dataset_name} test_n={len(test_df)} leakage_count={leakage_count} '
          f'polarity_corrected={args.swap_pos_neg}')
    print(f'[INFO] label distribution (test, post-correction): {dict(test_df["label"].value_counts())}')

    summaries = []
    files_index = {}

    # Zero-shot
    zs_label = f'finbert_zero_shot_{args.zero_shot_model}'
    zs_summary, zs_files = evaluate_one(
        zs_label, args.zero_shot_model, test_df, args, ts, results_dir, figures_dir, dataset_name, leakage_count, git_sha,
    )
    zs_summary['model_family'] = 'finbert_zero_shot'
    summaries.append(zs_summary)
    files_index[zs_label] = zs_files

    # Fine-tuned checkpoints
    ckpt_paths = [p.strip() for p in args.ft_ckpts.split(',') if p.strip()]
    for ck in ckpt_paths:
        ck_id = Path(ck).name
        ft_label = f'finbert_finetuned_{ck_id}'
        ft_summary, ft_files = evaluate_one(
            ft_label, ck, test_df, args, ts, results_dir, figures_dir, dataset_name, leakage_count, git_sha,
        )
        ft_summary['model_family'] = 'finbert_finetuned'
        summaries.append(ft_summary)
        files_index[ft_label] = ft_files

    summary_df = pd.DataFrame(summaries)
    summary_path = save_dataframe(summary_df, results_dir, 'phase13_external_validation_summary', ts)
    print(f'\n[OUT] {summary_path}')
    print(summary_df[['model', 'model_family', 'n_samples', 'accuracy', 'macro_f1', 'weighted_f1', 'mcc', 'ece_10_bins', 'brier_score', 'mean_confidence']].to_string(index=False))

    # Manifest
    manifest_meta = {
        'phase': 'phase13_step2_external_validation',
        'dataset_name': dataset_name,
        'dataset_file': args.data,
        'eval_split': 'test',
        'n_samples_test': int(len(test_df)),
        'label_distribution_test': {str(k): int(v) for k, v in test_df['label'].value_counts().items()},
        'zero_shot_model': args.zero_shot_model,
        'finetuned_checkpoints': ckpt_paths,
        'agreement_weighted_evaluated': False,
        'agreement_weighted_skipped_reason': (
            'No AW checkpoint on disk (Kaggle-trained; weights not synced back). '
            'Documented in docs/PHASE13_BASELINE_SNAPSHOT.md.'
        ),
        'text_hash_leakage_count': int(leakage_count),
        'git_commit_sha': git_sha,
        'generated_at_utc': ts,
        'label_polarity_corrected': bool(args.swap_pos_neg),
        'label_polarity_correction_reason': (
            'FiQA gold split has positive<->negative inverted at the source data layer. '
            'See docs/PHASE13_FIQA_LABEL_POLARITY_BUG.md.'
        ) if args.swap_pos_neg else 'no correction applied',
        'caveats': [
            'Vanilla fine-tuned results are per-checkpoint, not 5-seed averaged.',
            'FiQA gold split is the only external dataset evaluated in this run.',
            'No claim of SOTA is made on FiQA; this run is for reliability validation only.',
            'FiQA labels were polarity-corrected at runtime; raw gold remains uncorrected on disk.',
        ],
    }
    manifest_files = {'summary': str(summary_path)}
    for k, v in files_index.items():
        manifest_files[k] = str(v.get('predictions', ''))
    manifest_path = write_manifest(
        results_dir,
        run_name='phase13_external_validation',
        files=manifest_files,
        metadata=manifest_meta,
        ts=ts,
    )
    print(f'[OUT] manifest = {manifest_path}')

    # Also drop a copy of the summary under artifacts/ so it can be safely committed.
    artifacts_dir = PROJECT_ROOT / 'artifacts' / 'phase13_extended_validation'
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    safe_summary = artifacts_dir / f'external_validation_summary_{ts}.csv'
    summary_df.to_csv(safe_summary, index=False)
    save_json(manifest_meta, str(artifacts_dir), 'external_validation_manifest', ts)
    print(f'[OUT] safe artefact = {safe_summary}')


if __name__ == '__main__':
    main()
