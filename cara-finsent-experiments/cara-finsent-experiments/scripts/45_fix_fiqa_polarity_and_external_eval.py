#!/usr/bin/env python3
"""Phase 14 Step 7 — Materialise polarity-corrected FiQA derived dataset and
re-run zero-shot external validation on it.

Two sub-tasks:

  A. Read raw `data/processed/gold/latest_gold_fiqa_split.csv` and emit
     ``data/processed/gold/latest_gold_fiqa_split_polarity_corrected.csv``
     with positive↔negative labels swapped (neutral untouched). Raw stays
     intact.
  B. Run zero-shot ProsusAI/finbert on the corrected file's test split.
     Optionally run a local FT or AW checkpoint via ``--ft_checkpoint`` /
     ``--aw_checkpoint``.

Outputs land under ``results/<date>/phase14_external/`` and are mirrored to
``artifacts/phase14_corrected_validation/external/``.
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
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from cara_finsent.data_utils import STANDARD_LABELS, set_global_seeds
from cara_finsent.io_utils import git_commit_sha, timestamp
from cara_finsent.label_mapping import model_label_remap, remap_probs
from cara_finsent.metrics import expected_calibration_error, multiclass_brier_score

LABEL2IDX = {c: i for i, c in enumerate(STANDARD_LABELS)}


def materialise_corrected(raw_path: Path, out_path: Path, manifest_path: Path, git_sha: str, ts: str) -> dict:
    raw = pd.read_csv(raw_path)
    if 'label' not in raw.columns:
        raise SystemExit(f'{raw_path} has no `label` column')
    swap_map = {'positive': 'negative', 'negative': 'positive', 'neutral': 'neutral'}
    pre_counts = raw['label'].value_counts().to_dict()
    corrected = raw.copy()
    corrected['label'] = corrected['label'].map(swap_map)
    if corrected['label'].isna().any():
        bad = raw.loc[corrected['label'].isna(), 'label'].unique().tolist()
        raise SystemExit(f'Unrecognised label values in raw FiQA: {bad}')
    post_counts = corrected['label'].value_counts().to_dict()
    n_swapped = int(((raw['label'] != 'neutral') & (raw['label'] != corrected['label'])).sum())
    corrected.to_csv(out_path, index=False)
    manifest = {
        'phase': 'phase14_step7_fiqa_polarity_correction',
        'source_raw': str(raw_path),
        'output_corrected': str(out_path),
        'rows': int(len(raw)),
        'rows_swapped_pos_neg': n_swapped,
        'rows_unchanged_neutral': int((raw['label'] == 'neutral').sum()),
        'swap_rule': 'positive <-> negative; neutral unchanged',
        'pre_label_counts': pre_counts,
        'post_label_counts': post_counts,
        'reason': ('PHASE13_FIQA_LABEL_POLARITY_BUG.md confirmed FiQA gold split '
                   'has positive/negative inverted at source. This file is the '
                   'cleaned derivative; raw is preserved unchanged.'),
        'git_commit_sha': git_sha,
        'generated_at_utc': ts,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def _eval_model(model_id, model_name, ckpt_path, df_test, max_length, batch_size, seed, family_label):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.vocab_size < 1000:
            raise ValueError(f'tokenizer at {model_id} has vocab_size={tokenizer.vocab_size}')
    except (OSError, ValueError) as exc:
        print(f'[WARN] tokenizer not usable from {model_id} ({exc.__class__.__name__}: {exc}); falling back to ProsusAI/finbert')
        tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    native = {int(k): str(v) for k, v in dict(model.config.id2label).items()}
    df_test = df_test.copy()
    df_test['label_id'] = df_test['label'].map(LABEL2IDX)
    ds = Dataset.from_pandas(df_test[['text', 'label_id']]).map(
        lambda b: tokenizer(b['text'], max_length=max_length, truncation=True, padding='max_length'),
        batched=True, remove_columns=['text'])
    ds = ds.rename_column('label_id', 'labels')
    args = TrainingArguments(output_dir='_tmp_phase14_ext', per_device_eval_batch_size=batch_size,
                             fp16=False, seed=seed, report_to='none')
    trainer = Trainer(model=model, args=args)
    out = trainer.predict(ds)
    probs_native = torch.softmax(torch.tensor(out.predictions), dim=1).numpy()
    probs = remap_probs(probs_native, native)
    pred = np.array([STANDARD_LABELS[int(i)] for i in probs.argmax(axis=1)])
    y_true = df_test['label'].astype(str).to_numpy()
    metrics = {
        'family': family_label,
        'model_id': model_id,
        'checkpoint_path': str(ckpt_path) if ckpt_path else '',
        'n': int(len(df_test)),
        'accuracy': float(accuracy_score(y_true, pred)),
        'macro_f1': float(f1_score(y_true, pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, pred, average='weighted', zero_division=0)),
        'mcc': float(matthews_corrcoef(y_true, pred)),
        'ece_10_bins': float(expected_calibration_error(y_true, pred, probs, n_bins=10)),
        'brier_score': float(multiclass_brier_score(y_true, probs)),
        'native_id2label': json.dumps(native, sort_keys=True),
        'canonical_remap': json.dumps(model_label_remap(native)),
    }
    pred_df = df_test[['text', 'label']].copy()
    pred_df['pred_label'] = pred
    pred_df['proba_negative'] = probs[:, 0]
    pred_df['proba_neutral'] = probs[:, 1]
    pred_df['proba_positive'] = probs[:, 2]
    pred_df['confidence'] = probs.max(axis=1)
    pred_df['family'] = family_label
    return metrics, pred_df


def main():
    parser = argparse.ArgumentParser(description='Phase 14 Step 7 - FiQA polarity correction + external eval.')
    parser.add_argument('--raw_fiqa', default='data/processed/gold/latest_gold_fiqa_split.csv')
    parser.add_argument('--out_corrected', default='data/processed/gold/latest_gold_fiqa_split_polarity_corrected.csv')
    parser.add_argument('--out_manifest', default='data/processed/gold/latest_gold_fiqa_split_polarity_corrected_manifest.json')
    parser.add_argument('--zs_model', default='ProsusAI/finbert')
    parser.add_argument('--ft_checkpoint', default=None)
    parser.add_argument('--aw_checkpoint', default=None)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--skip_eval', action='store_true', help='Only materialise the corrected CSV; skip model runs.')
    args = parser.parse_args()

    set_global_seeds(args.seed, enable_deep_learning=True)
    ts = timestamp()
    git_sha = git_commit_sha(PROJECT_ROOT)
    date_subdir = f'{ts[:4]}-{ts[4:6]}-{ts[6:8]}'

    raw = Path(args.raw_fiqa)
    out = Path(args.out_corrected)
    out_manifest = Path(args.out_manifest)
    correction_manifest = materialise_corrected(raw, out, out_manifest, git_sha, ts)
    print(f'[OK] corrected derived file -> {out}')
    print(f'  pre  : {correction_manifest["pre_label_counts"]}')
    print(f'  post : {correction_manifest["post_label_counts"]}')

    if args.skip_eval:
        print('[INFO] --skip_eval set; not running models.')
        return

    df = pd.read_csv(out)
    if 'split' in df.columns:
        df_test = df[df['split'].astype(str).str.lower() == 'test'].copy()
        if df_test.empty:
            df_test = df.copy()
    else:
        df_test = df.copy()
    print(f'[INFO] external eval on {len(df_test)} polarity-corrected FiQA rows')

    out_dir = PROJECT_ROOT / args.results_dir / date_subdir / 'phase14_external'
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    pred_paths = {}

    metrics_zs, pred_zs = _eval_model(args.zs_model, args.zs_model, '', df_test,
                                      args.max_length, args.batch_size, args.seed, 'finbert_zero_shot')
    p = out_dir / f'phase14_external_zs_predictions_{ts}.csv'
    pred_zs.to_csv(p, index=False); pred_paths['finbert_zero_shot'] = str(p)
    summary_rows.append(metrics_zs)
    print(f'[ZS] macro_f1={metrics_zs["macro_f1"]:.4f} acc={metrics_zs["accuracy"]:.4f} ece={metrics_zs["ece_10_bins"]:.4f}')

    if args.ft_checkpoint:
        metrics_ft, pred_ft = _eval_model(args.ft_checkpoint, 'finbert_fine_tuned', args.ft_checkpoint, df_test,
                                          args.max_length, args.batch_size, args.seed, 'finbert_fine_tuned')
        p = out_dir / f'phase14_external_ft_predictions_{ts}.csv'
        pred_ft.to_csv(p, index=False); pred_paths['finbert_fine_tuned'] = str(p)
        summary_rows.append(metrics_ft)
        print(f'[FT] macro_f1={metrics_ft["macro_f1"]:.4f} acc={metrics_ft["accuracy"]:.4f}')

    if args.aw_checkpoint:
        metrics_aw, pred_aw = _eval_model(args.aw_checkpoint, 'finbert_agreement_weighted', args.aw_checkpoint, df_test,
                                          args.max_length, args.batch_size, args.seed, 'finbert_agreement_weighted')
        p = out_dir / f'phase14_external_aw_predictions_{ts}.csv'
        pred_aw.to_csv(p, index=False); pred_paths['finbert_agreement_weighted'] = str(p)
        summary_rows.append(metrics_aw)
        print(f'[AW] macro_f1={metrics_aw["macro_f1"]:.4f} acc={metrics_aw["accuracy"]:.4f}')

    summary_df = pd.DataFrame(summary_rows)
    summary_df['polarity_corrected_source'] = str(out)
    summary_df['git_commit_sha'] = git_sha
    summary_df['generated_at_utc'] = ts
    summary_path = out_dir / f'phase14_external_summary_{ts}.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f'\n[OUT] {summary_path}\n')
    print(summary_df[['family', 'n', 'accuracy', 'macro_f1', 'mcc', 'ece_10_bins']].to_string(index=False))

    manifest = {
        'phase': 'phase14_step7_external_validation',
        'polarity_corrected_source': str(out),
        'eval_n': int(len(df_test)),
        'predictions': pred_paths,
        'summary': str(summary_path),
        'git_commit_sha': git_sha,
        'generated_at_utc': ts,
    }
    manifest_path = out_dir / f'phase14_external_manifest_{ts}.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))

    artifacts_dir = PROJECT_ROOT / 'artifacts' / 'phase14_corrected_validation' / 'external'
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(artifacts_dir / summary_path.name, index=False)
    (artifacts_dir / manifest_path.name).write_text(json.dumps(manifest, indent=2))
    print(f'[OUT] safe artefacts in {artifacts_dir}')


if __name__ == '__main__':
    main()
