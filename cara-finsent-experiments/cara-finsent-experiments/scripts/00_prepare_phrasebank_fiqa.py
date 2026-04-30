#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from cara_finsent.data_utils import normalize_label, load_standardized_csv
from cara_finsent.io_utils import save_dataframe, timestamp, write_manifest

PHRASEBANK_CONFIGS = [
    ('sentences_50agree', 0.50),
    ('sentences_66agree', 0.66),
    ('sentences_75agree', 0.75),
    ('sentences_allagree', 1.00),
]


def _label_from_hf(row, features=None):
    value = row.get('label')
    if features is not None and hasattr(features.get('label'), 'names'):
        names = features['label'].names
        if isinstance(value, int) and value < len(names):
            return normalize_label(names[value])
    return normalize_label(value)


def load_phrasebank_hf() -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError('Install datasets: pip install datasets') from exc

    rows = []
    errors = []
    dataset_names = ['financial_phrasebank', 'takala/financial_phrasebank']
    for dataset_name in dataset_names:
        rows.clear()
        errors.clear()
        for config, agreement in PHRASEBANK_CONFIGS:
            try:
                ds = load_dataset(dataset_name, config, split='train', trust_remote_code=True)
                for i, row in enumerate(ds):
                    text = row.get('sentence') or row.get('text') or row.get('Sentence')
                    if not text:
                        continue
                    rows.append({
                        'id': f'phrasebank_{config}_{i}',
                        'text': str(text),
                        'label': _label_from_hf(row, ds.features),
                        'agreement': agreement,
                        'agreement_config': config,
                        'source_dataset': 'financial_phrasebank',
                    })
            except Exception as exc:
                errors.append(f'{dataset_name}/{config}: {exc}')
        if rows:
            df = pd.DataFrame(rows)
            # Deduplicate repeated sentences across agreement subsets. Keep highest agreement.
            df = df.sort_values('agreement', ascending=False).drop_duplicates(subset=['text'], keep='first')
            df = df[df['label'].isin(['negative', 'neutral', 'positive'])].reset_index(drop=True)
            return df
    raise RuntimeError('Could not load Financial PhraseBank from Hugging Face. Errors: ' + ' | '.join(errors))


def load_fiqa_hf() -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError('Install datasets: pip install datasets') from exc

    candidate_names = [
        'ChanceFocus/fiqa-sentiment-classification',
        'pauri32/fiqa-2018',
        'TheFinAI/fiqa-sentiment',
    ]
    errors = []
    for name in candidate_names:
        try:
            ds_dict = load_dataset(name, trust_remote_code=True)
            rows = []
            for split_name, ds in ds_dict.items():
                for i, row in enumerate(ds):
                    text = row.get('sentence') or row.get('text') or row.get('query') or row.get('headline') or row.get('comment')
                    value = row.get('label', row.get('sentiment', row.get('score', row.get('sentiment_score'))))
                    if not text or value is None:
                        continue
                    rows.append({
                        'id': f'fiqa_{split_name}_{i}',
                        'text': str(text),
                        'label': normalize_label(value),
                        'source_dataset': 'fiqa',
                        'split': split_name,
                    })
            df = pd.DataFrame(rows)
            df = df[df['label'].isin(['negative', 'neutral', 'positive'])].reset_index(drop=True)
            if len(df):
                return df
        except Exception as exc:
            errors.append(f'{name}: {exc}')
    raise RuntimeError('Could not load FiQA from known HF datasets. Provide --fiqa_csv. Errors: ' + ' | '.join(errors))


def main():
    parser = argparse.ArgumentParser(description='Prepare Financial PhraseBank and FiQA into standardized CSV files.')
    parser.add_argument('--local_csv', type=str, default=None, help='Optional local CSV with text/sentence and label/sentiment columns.')
    parser.add_argument('--fiqa_csv', type=str, default=None, help='Optional local FiQA CSV with text and label/score columns.')
    parser.add_argument('--skip_hf', action='store_true', help='Skip Hugging Face downloads; use only local CSVs.')
    parser.add_argument('--skip_fiqa', action='store_true', help='Prepare PhraseBank/local only; skip FiQA.')
    parser.add_argument('--output_dir', type=str, default='data/processed')
    args = parser.parse_args()

    ts = timestamp()
    outputs = {}
    frames: List[pd.DataFrame] = []

    if args.local_csv:
        local = load_standardized_csv(args.local_csv)
        local['source_dataset'] = local.get('source_dataset', 'local')
        path = save_dataframe(local, args.output_dir, 'local_standardized', ts)
        outputs['local_standardized'] = str(path)
        frames.append(local)

    if not args.skip_hf:
        try:
            phrase = load_phrasebank_hf()
            path = save_dataframe(phrase, args.output_dir, 'phrasebank_standardized', ts)
            outputs['phrasebank_standardized'] = str(path)
            frames.append(phrase)
            print(f'[OK] PhraseBank rows: {len(phrase)} -> {path}')
        except Exception as exc:
            print(f'[WARN] PhraseBank download failed: {exc}')

    if args.fiqa_csv:
        fiqa = load_standardized_csv(args.fiqa_csv)
        fiqa['source_dataset'] = 'fiqa'
        path = save_dataframe(fiqa, args.output_dir, 'fiqa_standardized', ts)
        outputs['fiqa_standardized'] = str(path)
        frames.append(fiqa)
    elif not args.skip_hf and not args.skip_fiqa:
        try:
            fiqa = load_fiqa_hf()
            path = save_dataframe(fiqa, args.output_dir, 'fiqa_standardized', ts)
            outputs['fiqa_standardized'] = str(path)
            frames.append(fiqa)
            print(f'[OK] FiQA rows: {len(fiqa)} -> {path}')
        except Exception as exc:
            print(f'[WARN] FiQA download failed: {exc}')

    if not frames:
        raise SystemExit('No data prepared. Provide --local_csv or enable HF download.')

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=['text', 'label']).reset_index(drop=True)
    combined_path = save_dataframe(combined, args.output_dir, 'combined_standardized', ts)
    outputs['combined_standardized'] = str(combined_path)

    inventory = combined.groupby(['source_dataset', 'label']).size().reset_index(name='rows')
    inventory_path = save_dataframe(inventory, 'results', 'dataset_inventory', ts)
    outputs['dataset_inventory'] = str(inventory_path)

    manifest = write_manifest('results', 'dataset_preparation', outputs, metadata={'rows_total': int(len(combined))}, ts=ts)
    print(f'[DONE] Combined rows: {len(combined)} -> {combined_path}')
    print(f'[DONE] Manifest -> {manifest}')


if __name__ == '__main__':
    main()
