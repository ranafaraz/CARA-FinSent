"""Class-balancing utilities for the canonical CARA-FinSent dataset.

Two strategies (pluggable):

- **undersample** (default): drop majority rows until
  `max(class_count) / min(class_count) <= max_imbalance`.
  Silver rows are dropped before gold so the gold contribution stays maximal.

- **paraphrase** (optional, `--paraphrase`): generate synthetic minority-class
  rows via a CPU-friendly T5 paraphraser
  (`Vamsi/T5_Paraphrase_Paws`). Generated rows are tagged
  `is_synthetic=True` and `tier='synthetic'` so they can be ablated.

This script is normally called from `05_build_dataset.py` rather than
directly, but is exposed for ad-hoc rebalancing of any CSV.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import numpy as np
import pandas as pd


TIER_RANK = {'gold': 3, 'silver': 2, 'synthetic': 1, 'bronze': 0}


def undersample(df: pd.DataFrame, max_imbalance: float = 2.0,
                seed: int = 42) -> pd.DataFrame:
    counts = df['label'].value_counts()
    if counts.empty:
        return df
    target_max = int(counts.min() * max_imbalance)
    keep_frames = []
    for label, n in counts.items():
        cls = df[df['label'] == label]
        if n > target_max:
            cls = cls.assign(_r=cls['tier'].map(TIER_RANK).fillna(0))
            cls = cls.sort_values('_r', ascending=False).iloc[:target_max].drop(columns=['_r'])
        keep_frames.append(cls)
    return pd.concat(keep_frames).sample(frac=1, random_state=seed).reset_index(drop=True)


def paraphrase_minority(df: pd.DataFrame, target_label: str = 'negative',
                        cap_fraction: float = 0.3, seed: int = 42) -> pd.DataFrame:
    """Generate paraphrased copies of minority-class rows.

    Off by default in the canonical builder because T5 inference is slow
    on CPU (~25 min for 1k rows). Use only when the dataset cannot be
    balanced by undersampling alone.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
    except ImportError as exc:
        sys.exit(f'paraphrase_minority requires transformers + torch: {exc}')

    minority = df[df['label'] == target_label].copy()
    n_to_make = int(len(minority) * cap_fraction)
    if n_to_make <= 0:
        return df

    print(f'[paraphrase] generating {n_to_make} synthetic {target_label} rows '
          f'(cap_fraction={cap_fraction})')
    rng = np.random.RandomState(seed)
    seeds_idx = rng.choice(minority.index, size=n_to_make, replace=False)
    seeds = minority.loc[seeds_idx]

    name = 'Vamsi/T5_Paraphrase_Paws'
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    model.eval()

    syn_rows = []
    for _, row in seeds.iterrows():
        prompt = f'paraphrase: {row["text_clean"]} </s>'
        ids = tok(prompt, return_tensors='pt', truncation=True, max_length=256).input_ids
        with torch.no_grad():
            out = model.generate(ids, max_length=256, num_beams=4,
                                 num_return_sequences=1, no_repeat_ngram_size=2,
                                 early_stopping=True)
        para = tok.decode(out[0], skip_special_tokens=True)
        if not para or para == row['text_clean']:
            continue
        new = row.copy()
        new['text'] = para
        new['text_clean'] = para
        new['text_clean_lower'] = para.lower()
        new['source'] = f'{row["source"]}_paraphrased'
        new['tier'] = 'synthetic'
        new['is_synthetic'] = True
        syn_rows.append(new)

    if not syn_rows:
        return df
    syn_df = pd.DataFrame(syn_rows)
    return pd.concat([df, syn_df], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-imbalance', type=float, default=2.0)
    parser.add_argument('--paraphrase', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f'before: {df["label"].value_counts().to_dict()}')
    if args.paraphrase:
        df = paraphrase_minority(df, seed=args.seed)
    df = undersample(df, args.max_imbalance, args.seed)
    print(f'after:  {df["label"].value_counts().to_dict()}')
    df.to_csv(args.output, index=False)
    print(f'[DONE] wrote {len(df)} rows -> {args.output}')


if __name__ == '__main__':
    main()
