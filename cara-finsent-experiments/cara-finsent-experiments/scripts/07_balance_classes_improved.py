#!/usr/bin/env python3
"""
07_balance_classes_improved.py
---
Improved class balancing via undersampling + synthetic negative generation.

Features:
- Undersampling of majority classes (with tier preference: drop silver before gold)
- Synthetic negative generation using T5 paraphrasing (configurable intensity)
- Preservation of sample metadata (tier, source, agreement)
- Stratification by source and label

Usage:
    python scripts/07_balance_classes_improved.py \
        --input data/processed/latest/dataset.csv \
        --output data/processed/latest/balanced_dataset.csv \
        --seed 42 \
        --synthetic_intensity 0.5 \
        --no_synthetic_only_negatives  # Generate for all classes if set
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import numpy as np
import pandas as pd

from cara_finsent.io_utils import timestamp


def undersample_majority(df: pd.DataFrame, target_ratio: float = 2.0, 
                        seed: int = 42) -> pd.DataFrame:
    """
    Undersample majority classes to achieve target imbalance ratio.
    
    Prefers dropping silver rows over gold (tier-aware).
    """
    rng = np.random.RandomState(seed)
    df = df.copy()
    
    label_counts = df['label'].value_counts()
    min_count = label_counts.min()
    target_count = int(min_count * target_ratio)
    
    result_frames = []
    for label in df['label'].unique():
        label_df = df[df['label'] == label].copy()
        current_count = len(label_df)
        
        if current_count <= target_count:
            result_frames.append(label_df)
        else:
            # Need to undersample
            excess = current_count - target_count
            
            # Try to drop silver rows first
            silver_rows = label_df[label_df.get('tier', 'gold') == 'silver']
            if len(silver_rows) >= excess:
                # Drop from silver only
                keep_indices = rng.choice(len(silver_rows), size=len(silver_rows) - excess, replace=False)
                drop_indices = [silver_rows.index[i] for i in range(len(silver_rows)) if i not in keep_indices]
            else:
                # Drop all silver + some gold
                drop_indices = silver_rows.index.tolist()
                gold_rows = label_df[label_df.get('tier', 'gold') == 'gold']
                remaining_excess = excess - len(silver_rows)
                keep_indices = rng.choice(len(gold_rows), size=len(gold_rows) - remaining_excess, replace=False)
                drop_indices += [gold_rows.index[i] for i in range(len(gold_rows)) if i not in keep_indices]
            
            result_frames.append(label_df.drop(drop_indices))
    
    result_df = pd.concat(result_frames, ignore_index=False)
    
    print(f'[BALANCE] Target ratio: {target_ratio:.2f}x')
    print(f'  Before: {label_counts.to_dict()}')
    print(f'  After:  {result_df["label"].value_counts().to_dict()}')
    
    return result_df.reset_index(drop=True)


def generate_synthetic_negatives(df: pd.DataFrame, intensity: float = 0.5, 
                                seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic negative examples via T5 paraphrasing.
    
    intensity: fraction of new negatives to generate relative to current negative count.
              0.3 means generate enough to increase negatives by 30%.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except ImportError:
        print('[WARN] transformers not available; skipping synthetic generation')
        return df
    
    negative_rows = df[df['label'] == 'negative'].copy()
    if len(negative_rows) == 0:
        print('[WARN] No negative examples found; skipping synthetic generation')
        return df
    
    num_to_generate = max(1, int(len(negative_rows) * intensity))
    if num_to_generate == 0:
        print('[INFO] Synthetic intensity too low; skipping')
        return df
    
    print(f'[SYNTHETIC] Generating {num_to_generate} synthetic negatives (intensity={intensity:.1%})...')
    
    # Load T5 paraphrase model (lightweight)
    try:
        model_name = 'Vamsi/T5_Paraphrase_Paws'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception as e:
        print(f'[WARN] Failed to load T5 model: {e}; skipping synthetic generation')
        return df
    
    rng = np.random.RandomState(seed)
    source_indices = rng.choice(len(negative_rows), size=num_to_generate, replace=True)
    
    synthetic_rows = []
    for idx in source_indices:
        try:
            source_text = negative_rows.iloc[idx]
            text = source_text.get('text_clean', source_text.get('text', ''))
            
            # Paraphrase via T5
            input_text = f"paraphrase: {text} </s>"
            inputs = tokenizer(input_text, return_tensors='pt', max_length=256, truncation=True)
            outputs = model.generate(**inputs, max_length=256, num_beams=5, early_stopping=True)
            paraphrase = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Create synthetic row
            synthetic_row = source_text.copy()
            synthetic_row['text'] = paraphrase
            synthetic_row['is_synthetic'] = True
            synthetic_rows.append(synthetic_row)
        except Exception as e:
            print(f'  [WARN] Failed to paraphrase row {idx}: {e}')
            continue
    
    if synthetic_rows:
        synthetic_df = pd.DataFrame(synthetic_rows)
        df = pd.concat([df, synthetic_df], ignore_index=True)
        print(f'[INFO] Generated {len(synthetic_rows)} synthetic negatives')
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Improved class balancing with tier-aware undersampling and synthetic negatives.')
    parser.add_argument('--input', default='data/processed/latest/dataset.csv',
                       help='Input dataset CSV')
    parser.add_argument('--output', default='data/processed/latest/balanced_dataset.csv',
                       help='Output balanced dataset CSV')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--target_ratio', type=float, default=2.0,
                       help='Target max:min class count ratio')
    parser.add_argument('--synthetic_intensity', type=float, default=0.3,
                       help='Fraction of new negatives to generate')
    parser.add_argument('--no_synthetic_only_negatives', action='store_true',
                       help='If set, generate synthetic for all classes')
    args = parser.parse_args()
    
    # Load
    df = pd.read_csv(args.input)
    print(f'[LOAD] {len(df)} rows from {args.input}')
    print(f'  Labels: {df["label"].value_counts().to_dict()}')
    
    # Undersample
    df = undersample_majority(df, target_ratio=args.target_ratio, seed=args.seed)
    
    # Generate synthetic
    if not args.no_synthetic_only_negatives:
        df = generate_synthetic_negatives(df, intensity=args.synthetic_intensity, seed=args.seed)
    
    # Save
    df.to_csv(args.output, index=False)
    print(f'[SAVE] {len(df)} rows -> {args.output}')
    print(f'  Final labels: {df["label"].value_counts().to_dict()}')


if __name__ == '__main__':
    main()
