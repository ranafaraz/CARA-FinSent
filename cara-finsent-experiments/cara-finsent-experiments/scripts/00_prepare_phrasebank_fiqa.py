#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import argparse
import shutil
import zipfile
from pathlib import Path
from typing import List

import pandas as pd

from cara_finsent.data_utils import normalize_label, load_standardized_csv
from cara_finsent.io_utils import save_dataframe, timestamp, write_manifest
from cara_finsent.label_mapping import canonical_label, text_hash


def _to_clean_gold(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Project a source-specific dataframe to the Phase 1 clean-gold schema."""
    out = df.copy()
    out['text'] = out['text'].astype(str)
    out['label'] = out['label'].apply(canonical_label)
    out = out[out['label'].isin({'negative', 'neutral', 'positive'})].copy()
    out = out[out['text'].str.strip().str.len() > 0].copy()
    out['source_dataset'] = out.get('source_dataset', source_name)
    if 'id' not in out.columns:
        out['id'] = [f'{source_name}_{i}' for i in range(len(out))]
    out['text_hash'] = out['text'].apply(text_hash)
    cols = ['id', 'text', 'label', 'source_dataset', 'text_hash']
    if 'agreement' in out.columns:
        cols.append('agreement')
    if 'agreement_config' in out.columns:
        cols.append('agreement_config')
    extra = [c for c in out.columns if c not in cols]
    return out[cols + extra].reset_index(drop=True)

PHRASEBANK_CONFIGS = [
    ('sentences_50agree', 0.50),
    ('sentences_66agree', 0.66),
    ('sentences_75agree', 0.75),
    ('sentences_allagree', 1.00),
]


def _parse_phrasebank_line(line: str):
    text = line.strip()
    if not text:
        return None, None
    labels = {'negative', 'neutral', 'positive'}
    if '@' in text:
        left, right = text.split('@', 1)
        left_n = normalize_label(left.strip())
        right_n = normalize_label(right.strip())
        if left_n in labels:
            return right.strip(), left_n
        if right_n in labels:
            return left.strip(), right_n
    return None, None


def _load_phrasebank_from_repo_files(dataset_name: str) -> pd.DataFrame:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except Exception as exc:
        raise RuntimeError('Install huggingface_hub for PhraseBank raw-file fallback.') from exc

    agreement_map = {
        '50': ('sentences_50agree', 0.50),
        '66': ('sentences_66agree', 0.66),
        '75': ('sentences_75agree', 0.75),
        'all': ('sentences_allagree', 1.00),
    }
    files = list_repo_files(dataset_name, repo_type='dataset')

    # Preferred fallback: official zip archive from the HF dataset repo.
    zip_name = 'data/FinancialPhraseBank-v1.0.zip'
    if zip_name in files:
        local_zip = hf_hub_download(repo_id=dataset_name, repo_type='dataset', filename=zip_name)
        rows = []
        with zipfile.ZipFile(local_zip) as zf:
            zip_map = {
                'sentences_50agree': ('sentences_50agree', 0.50),
                'sentences_66agree': ('sentences_66agree', 0.66),
                'sentences_75agree': ('sentences_75agree', 0.75),
                'sentences_allagree': ('sentences_allagree', 1.00),
            }
            for member in zf.namelist():
                lower = member.lower()
                if not lower.endswith('.txt') or 'sentences_' not in lower:
                    continue
                selected_cfg = None
                for key, cfg in zip_map.items():
                    if key in lower:
                        selected_cfg = cfg
                        break
                if selected_cfg is None:
                    continue
                cfg_name, agreement = selected_cfg
                with zf.open(member, 'r') as fh:
                    for i, raw_line in enumerate(fh.read().decode('latin-1', errors='ignore').splitlines()):
                        text, label = _parse_phrasebank_line(raw_line)
                        if not text or label not in {'negative', 'neutral', 'positive'}:
                            continue
                        rows.append({
                            'id': f'phrasebank_{cfg_name}_{i}',
                            'text': text,
                            'label': label,
                            'agreement': agreement,
                            'agreement_config': cfg_name,
                            'source_dataset': 'financial_phrasebank',
                        })
        if rows:
            df = pd.DataFrame(rows)
            df = df.sort_values('agreement', ascending=False).drop_duplicates(subset=['text'], keep='first')
            df = df[df['label'].isin(['negative', 'neutral', 'positive'])].reset_index(drop=True)
            return df

    rows = []
    for file_name in files:
        lower = file_name.lower()
        if not lower.endswith('.txt'):
            continue
        match_key = None
        if '50' in lower:
            match_key = '50'
        elif '66' in lower:
            match_key = '66'
        elif '75' in lower:
            match_key = '75'
        elif 'all' in lower:
            match_key = 'all'
        if match_key is None:
            continue
        cfg_name, agreement = agreement_map[match_key]
        local_path = hf_hub_download(repo_id=dataset_name, repo_type='dataset', filename=file_name)
        with Path(local_path).open('r', encoding='utf-8', errors='ignore') as fh:
            for i, line in enumerate(fh):
                text, label = _parse_phrasebank_line(line)
                if not text or label not in {'negative', 'neutral', 'positive'}:
                    continue
                rows.append({
                    'id': f'phrasebank_{cfg_name}_{i}',
                    'text': text,
                    'label': label,
                    'agreement': agreement,
                    'agreement_config': cfg_name,
                    'source_dataset': 'financial_phrasebank',
                })
    if not rows:
        raise RuntimeError('No supported PhraseBank .txt files found in dataset repository.')
    df = pd.DataFrame(rows)
    df = df.sort_values('agreement', ascending=False).drop_duplicates(subset=['text'], keep='first')
    df = df[df['label'].isin(['negative', 'neutral', 'positive'])].reset_index(drop=True)
    return df


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
                ds = load_dataset(dataset_name, config, split='train')
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
    for dataset_name in dataset_names:
        try:
            return _load_phrasebank_from_repo_files(dataset_name)
        except Exception as exc:
            errors.append(f'{dataset_name}/raw_files: {exc}')
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
            ds_dict = load_dataset(name)
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
            # Phase 1 clean-gold output (separate from combined).
            gold_phrase = _to_clean_gold(phrase, 'financial_phrasebank')
            gp_path = save_dataframe(gold_phrase, Path(args.output_dir) / 'gold', 'phrasebank_clean', ts)
            shutil.copy(str(gp_path), str(Path(args.output_dir) / 'gold' / 'latest_gold_phrasebank.csv'))
            outputs['phrasebank_clean'] = str(gp_path)
            print(f'[OK] PhraseBank rows: {len(phrase)} -> {path}')
            print(f'[OK] PhraseBank gold rows: {len(gold_phrase)} -> {gp_path}')
        except Exception as exc:
            print(f'[WARN] PhraseBank download failed: {exc}')

    if args.fiqa_csv:
        fiqa = load_standardized_csv(args.fiqa_csv)
        fiqa['source_dataset'] = 'fiqa'
        path = save_dataframe(fiqa, args.output_dir, 'fiqa_standardized', ts)
        outputs['fiqa_standardized'] = str(path)
        frames.append(fiqa)
        gold_fiqa = _to_clean_gold(fiqa, 'fiqa')
        gf_path = save_dataframe(gold_fiqa, Path(args.output_dir) / 'gold', 'fiqa_clean', ts)
        shutil.copy(str(gf_path), str(Path(args.output_dir) / 'gold' / 'latest_gold_fiqa.csv'))
        outputs['fiqa_clean'] = str(gf_path)
        print(f'[OK] FiQA gold rows: {len(gold_fiqa)} -> {gf_path}')
    elif not args.skip_hf and not args.skip_fiqa:
        try:
            fiqa = load_fiqa_hf()
            path = save_dataframe(fiqa, args.output_dir, 'fiqa_standardized', ts)
            outputs['fiqa_standardized'] = str(path)
            frames.append(fiqa)
            gold_fiqa = _to_clean_gold(fiqa, 'fiqa')
            gf_path = save_dataframe(gold_fiqa, Path(args.output_dir) / 'gold', 'fiqa_clean', ts)
            shutil.copy(str(gf_path), str(Path(args.output_dir) / 'gold' / 'latest_gold_fiqa.csv'))
            outputs['fiqa_clean'] = str(gf_path)
            print(f'[OK] FiQA rows: {len(fiqa)} -> {path}')
            print(f'[OK] FiQA gold rows: {len(gold_fiqa)} -> {gf_path}')
        except Exception as exc:
            print(f'[WARN] FiQA download failed: {exc}')

    if not frames:
        raise SystemExit('No data prepared. Provide --local_csv or enable HF download.')

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=['text', 'label']).reset_index(drop=True)
    combined_path = save_dataframe(combined, args.output_dir, 'combined_standardized', ts)
    outputs['combined_standardized'] = str(combined_path)

    # Write a fixed-name alias so downstream scripts can locate data without DATA= override.
    latest_link = Path(args.output_dir) / 'latest.csv'
    shutil.copy(str(combined_path), str(latest_link))
    outputs['latest'] = str(latest_link)

    inventory = combined.groupby(['source_dataset', 'label']).size().reset_index(name='rows')
    inventory_path = save_dataframe(inventory, 'results', 'dataset_inventory', ts)
    outputs['dataset_inventory'] = str(inventory_path)

    manifest = write_manifest('results', 'dataset_preparation', outputs, metadata={'rows_total': int(len(combined))}, ts=ts)
    print(f'[DONE] Combined rows: {len(combined)} -> {combined_path}')
    print(f'[DONE] Manifest -> {manifest}')


if __name__ == '__main__':
    main()
