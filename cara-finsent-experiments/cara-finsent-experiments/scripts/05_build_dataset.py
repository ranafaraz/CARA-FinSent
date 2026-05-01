"""Build canonical CARA-FinSent dataset from raw + currently-cached sources.

Pipeline:
  1. Ingest from data/raw/<source>/*.csv (newest per source) AND from
     data/processed/latest.csv (legacy multi-source merge).
  2. Tag each row with `tier` (gold | silver | bronze).
  3. Apply preprocessing pipeline (URL/cashtag/emoji masking, language filter,
     length filter).
  4. Deduplicate within + across sources via MinHash LSH (tier-aware).
  5. Carve a stratified gold test set; remove those rows from train pool.
  6. Class-balance the train pool (undersample majority).
  7. Run train/test contamination check.
  8. Write canonical dataset, manifest, data card, archive snapshot.

CLI:
    python scripts/05_build_dataset.py [--dry-run]
                                       [--max-imbalance 2.0]
                                       [--target-test-size 1450]
                                       [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import textwrap
from dataclasses import asdict
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import numpy as np
import pandas as pd

from cara_finsent.preprocessing import preprocess_dataframe
from cara_finsent.dedup import dedup_dataframe, contamination_check
from cara_finsent.io_utils import timestamp, ensure_dir
from cara_finsent.data_utils import normalize_label

# ─── Tier definitions ───────────────────────────────────────────────────────
GOLD_SOURCES = {
    'financial_phrasebank',     # human annotated, agreement levels tracked
    'fiqa',                     # crowdsourced gold for benchmark
    'fiqa_sa',
    'auditor_sentiment',        # FinanceInc, human-annotated audit reports
    'semeval2017_task5',        # SemEval gold annotations
    'fomc_sentiment',           # gtfintechlab gold (hawkish/dovish)
}
SILVER_SOURCES = {
    'twitter_fin_news',         # API sentiment tags
    'twitter_fin_topic',
    'fingpt_sentiment',         # LLM-generated labels
    'financial_classification',
    'adaptllm_fpb',             # paraphrased PhraseBank
    'adaptllm_fiqasa',
    'fiqa_2018',                # extra fiqa source via pauri32
}
BRONZE_SOURCES = {
    'sec_10k_weak',
    'financial_news_headlines',
    'stocktwits',
}


def _tier(source: str) -> str:
    if source in GOLD_SOURCES:
        return 'gold'
    if source in SILVER_SOURCES:
        return 'silver'
    if source in BRONZE_SOURCES:
        return 'bronze'
    return 'silver'  # unknown defaults to silver (visible, but downweighted)


# ─── Ingestion ──────────────────────────────────────────────────────────────
def ingest_legacy_latest(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    # Normalise column names to canonical schema
    if 'source_dataset' in df.columns and 'source' not in df.columns:
        df = df.rename(columns={'source_dataset': 'source'})
    df['source'] = df['source'].fillna('unknown')
    if 'agreement' not in df.columns:
        df['agreement'] = pd.NA
    df['source_uri'] = df.get('source_uri', pd.NA)
    df['license'] = df.get('license', pd.NA)
    df['collected_at'] = df.get('collected_at', pd.NA)
    df['content_sha256'] = df.get('content_sha256', pd.NA)
    df['label_raw'] = df.get('label_raw', df.get('label'))
    if 'is_uncertain' not in df.columns:
        df['is_uncertain'] = False
    if 'split' not in df.columns:
        df['split'] = 'train'
    df['split'] = df['split'].fillna('train')
    return df[['text', 'label', 'label_raw', 'source', 'agreement', 'is_uncertain',
               'source_uri', 'license', 'collected_at', 'content_sha256',
               'split']]


def ingest_raw_dirs(base: Path) -> pd.DataFrame:
    """Pull newest CSV per data/raw/<source>/ folder."""
    if not base.exists():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for source_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        csvs = sorted(source_dir.rglob('*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
        if not csvs:
            continue
        try:
            f = pd.read_csv(csvs[0])
        except Exception as e:
            print(f'  [WARN] could not read {csvs[0]}: {e}')
            continue
        if 'text' not in f.columns or 'label' not in f.columns:
            continue
        if 'source' not in f.columns:
            f['source'] = source_dir.name
        for col in ('agreement', 'is_uncertain', 'source_uri', 'license', 'collected_at',
                    'content_sha256', 'label_raw', 'split'):
            if col not in f.columns:
                f[col] = pd.NA
        f['split'] = f['split'].fillna('train')
        frames.append(f[['text', 'label', 'label_raw', 'source', 'agreement', 'is_uncertain',
                         'source_uri', 'license', 'collected_at',
                         'content_sha256', 'split']])
        print(f'  [INGEST raw/{source_dir.name}] {len(f):>6d} rows')
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ─── Test-set construction ──────────────────────────────────────────────────
def carve_test_set(df: pd.DataFrame, target_size: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a stratified gold test set, return (train_pool, test_set)."""
    rng = np.random.RandomState(seed)
    test_frames: list[pd.DataFrame] = []
    used_idx: set[int] = set()

    # 1. Keep all rows already marked split='test' from gold sources
    pre_test = df[(df['split'] == 'test') & (df['tier'] == 'gold')]
    test_frames.append(pre_test)
    used_idx.update(pre_test.index.tolist())

    # 2. From each gold source, sample stratified rows
    per_source_quota = max(150, (target_size - len(pre_test)) // 5)
    for src in sorted(df['source'].dropna().unique()):
        src_rows = df[(df['source'] == src) & (df['tier'] == 'gold') & (~df.index.isin(used_idx))]
        if len(src_rows) < 60:
            continue
        # Prefer high-agreement rows when available
        src_rows = src_rows.copy()
        src_rows['_agr'] = pd.to_numeric(src_rows['agreement'], errors='coerce').fillna(1.0)
        src_rows = src_rows.sort_values('_agr', ascending=False)
        # Stratify by label
        per_label = max(20, per_source_quota // 3)
        picked: list[pd.DataFrame] = []
        for lab in ('negative', 'neutral', 'positive'):
            pool = src_rows[src_rows['label'] == lab]
            n = min(per_label, len(pool))
            if n == 0:
                continue
            picked.append(pool.sample(n=n, random_state=seed))
        if picked:
            chunk = pd.concat(picked).drop(columns=['_agr'])
            test_frames.append(chunk)
            used_idx.update(chunk.index.tolist())

    test_df = (pd.concat(test_frames).drop_duplicates(subset='text')
               .reset_index(drop=True))
    test_df['split'] = 'test'

    train_pool = df[~df.index.isin(used_idx)].copy().reset_index(drop=True)
    train_pool['split'] = 'train'
    return train_pool, test_df


# ─── Class balancing ────────────────────────────────────────────────────────
def balance_classes(df: pd.DataFrame, max_imbalance: float, seed: int) -> tuple[pd.DataFrame, dict]:
    counts = df['label'].value_counts()
    if counts.empty:
        return df, {}
    target_min = int(counts.min())
    target_max = int(target_min * max_imbalance)
    keep_frames: list[pd.DataFrame] = []
    report: dict = {'before': counts.to_dict(), 'target_max': target_max}
    for label, n in counts.items():
        cls = df[df['label'] == label]
        if n > target_max:
            # Prefer dropping silver before gold
            cls = cls.assign(_tier_rank=cls['tier'].map(
                {'gold': 3, 'silver': 2, 'bronze': 1}).fillna(0))
            cls = cls.sort_values('_tier_rank', ascending=False)
            cls = cls.iloc[:target_max].drop(columns=['_tier_rank'])
        keep_frames.append(cls)
    out = pd.concat(keep_frames).sample(frac=1, random_state=seed).reset_index(drop=True)
    report['after'] = out['label'].value_counts().to_dict()
    return out, report


# ─── Data card ──────────────────────────────────────────────────────────────
def write_data_card(out_dir: Path, manifest: dict) -> Path:
    card_path = out_dir / 'data_card.md'
    yaml_meta = textwrap.dedent("""\
    ---
    license: mixed
    language: en
    task_categories:
    - text-classification
    task_ids:
    - sentiment-classification
    multilinguality: monolingual
    size_categories:
    - 10K<n<100K
    pretty_name: CARA-FinSent Canonical Corpus
    ---
    """)
    sources_table = '\n'.join(
        f'| {s} | {info["tier"]} | {info["rows"]} | {info["license"]} |'
        for s, info in manifest['sources'].items()
    )
    body = textwrap.dedent(f"""
    # CARA-FinSent Canonical Sentiment Corpus

    Build ID: `{manifest['build_id']}`
    Built at: `{manifest['built_at']}`

    ## Composition

    - Total rows: **{manifest['total_rows']}**
    - Train: {manifest['split_counts'].get('train', 0)}
    - Test:  {manifest['split_counts'].get('test', 0)}
    - Tiers: gold = {manifest['tier_counts'].get('gold', 0)},
      silver = {manifest['tier_counts'].get('silver', 0)},
      bronze = {manifest['tier_counts'].get('bronze', 0)}
    - Class distribution (train): {manifest['train_label_counts']}
    - Class distribution (test):  {manifest['test_label_counts']}
    - Imbalance ratio (train, max/min): {manifest['imbalance_ratio_train']:.2f}

    ## Sources

    | source | tier | rows | license |
    |---|---|---|---|
    {sources_table}

    ## Preprocessing

    Applied uniformly to all sources via `src/cara_finsent/preprocessing.py`:

    - **Encoding repair** with `ftfy.fix_text` (corrects mojibake like `ÃÂ£` → `£`).
    - **Unicode NFKC normalisation**.
    - **Control / zero-width character removal**.
    - **HTML stripping** (`<...>`).
    - **URL → `[URL]`** (regex).
    - **`@user` → `[USER]`** (regex).
    - **`$TICKER` cashtags → `[TICKER]`** with the original ticker preserved in
      a separate `tickers` column.
    - **Emoji → text** via `emoji.demojize` (e.g. `😀` → `:grinning_face:`).
    - **Whitespace collapse**.

    Two text columns are produced:

    - `text_clean` — case-preserved, used by FinBERT.
    - `text_clean_lower` — additionally lowercased, used by TF-IDF / classical
      models.

    ## Filtering

    - Language: `langdetect` keeps `en` only (very short rows bypass the
      check, since langdetect is unreliable on <5-token strings).
    - Length: `3 <= n_tokens <= 512` (whitespace tokens).
    - Empty cleaned text: dropped.

    ## Deduplication

    `src/cara_finsent/dedup.py` runs:

    1. Exact dedup on lowercased `text_clean`, keeping the highest-tier copy.
    2. MinHash LSH (Jaccard ≥ 0.85, 128 permutations, 5-gram word shingles).
       When two near-duplicates differ in tier, the higher-tier row wins.

    Drop counts: exact = {manifest['dedup']['n_dropped_exact']},
    near = {manifest['dedup']['n_dropped_near']}.

    ## Train / test split

    A **stratified gold test set** of {manifest['split_counts'].get('test', 0)}
    rows is carved from gold sources only. Per-source, per-label
    stratification is enforced. High-agreement rows (when `agreement` is
    available) are preferred. The test set is then **frozen**: no future
    source addition can leak into it.

    A train/test contamination check (Jaccard ≥ 0.5) confirms no near-duplicates
    cross the split boundary. Detected contaminations: {manifest['contamination']['n_pairs']}.

    ## Class balancing

    The train pool is balanced so that
    `max(class_count) / min(class_count) ≤ {manifest['balance']['max_imbalance']}`.
    Majority classes are undersampled, dropping silver rows before gold.
    Synthetic minority generation (T5 paraphrasing) is supported but
    disabled by default.

    Train label distribution before → after balancing:
    {manifest['balance']['report']['before']} → {manifest['balance']['report']['after']}.

    ## Sample weights for tier-aware training

    Recommended per-row weights when fine-tuning:

    - gold      → `1.0`
    - silver    → `0.5`
    - synthetic → `0.3`

    ## Schema

    | column | type | description |
    |---|---|---|
    | `text` | str | raw original text |
    | `text_clean` | str | cleaned, case-preserved (FinBERT input) |
    | `text_clean_lower` | str | cleaned + lowercased (TF-IDF input) |
    | `label` | str | one of {{negative, neutral, positive}} |
    | `label_raw` | any | original label value from upstream source |
    | `agreement` | float? | annotator agreement (0.5 / 0.66 / 0.75 / 1.0) when known |
    | `source` | str | upstream dataset name |
    | `tier` | str | gold / silver / bronze |
    | `split` | str | train / test |
    | `language` | str | langdetect output |
    | `n_tokens` | int | whitespace token count of `text_clean` |
    | `qc_flags` | str | comma-sep flags: had_url, had_ticker, had_mention, encoding_repaired |
    | `tickers` | str | comma-sep upper-cased tickers extracted from raw text |
    | `is_synthetic` | bool | True if generated via paraphrasing |
    | `content_sha256` | str | first-16 hex of SHA-256 over raw text |
    | `license` | str | upstream license string when known |
    | `collected_at` | str | ISO8601 collection timestamp |
    | `build_id` | str | timestamp of this canonical build |

    ## Reproducibility

    - Seed: `42`
    - Build: `python scripts/05_build_dataset.py`
    - Validation: `pytest tests/test_dataset_quality.py`

    ## Considerations for use

    - **Domain mismatch**: Twitter posts and SEC paragraphs are stylistically
      different. The tier system lets you weight or ablate.
    - **Class imbalance is structural** in financial sentiment — neutral and
      positive dominate organically. Down-weighting positives may hurt
      production performance; the balanced version is for unbiased benchmarking.
    - **Synthetic data is off by default**.
    - **Ticker masking** removes information; if needed, restore via the
      `tickers` column.

    ## Citations

    Please cite the upstream sources:
    Financial PhraseBank (Malo et al. 2014); FiQA (Maia et al. 2018);
    SemEval-2017 Task 5 (Cortis et al. 2017); FOMC sentiment (Shah et al. 2023);
    FinGPT (Wang et al. 2023).
    """)
    card_path.write_text(yaml_meta + body, encoding='utf-8')
    return card_path


# ─── Driver ─────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--max-imbalance', type=float, default=2.0)
    parser.add_argument('--target-test-size', type=int, default=1450)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--legacy-latest', default='data/processed/latest.csv')
    args = parser.parse_args()

    ts = timestamp()
    print(f'[BUILD] build_id={ts}')

    # 1. Ingest
    print('[STEP 1] Ingest sources')
    legacy = ingest_legacy_latest(PROJECT_ROOT / args.legacy_latest)
    print(f'  legacy latest.csv: {len(legacy):>6d} rows')
    raw = ingest_raw_dirs(PROJECT_ROOT / 'data' / 'raw')
    df = pd.concat([legacy, raw], ignore_index=True) if not raw.empty else legacy
    if df.empty:
        sys.exit('[ERROR] no input data found')

    # Normalise labels
    df['label'] = df['label'].apply(normalize_label)
    df = df[df['label'].isin(['negative', 'neutral', 'positive'])].reset_index(drop=True)
    df['tier'] = df['source'].apply(_tier)
    print(f'  combined: {len(df):>6d} rows | tiers={df["tier"].value_counts().to_dict()}')

    # 2. Preprocess
    print('[STEP 2] Preprocess (clean / language / length filter)')
    df, prep_rep = preprocess_dataframe(df, text_col='text')
    print(f'  kept {prep_rep.n_kept}/{prep_rep.n_input}'
          f' (empty={prep_rep.n_empty_after_clean}, short={prep_rep.n_too_short},'
          f' long={prep_rep.n_too_long}, non_en={prep_rep.n_non_english})')

    # 3. Dedup
    print('[STEP 3] Dedup (exact + MinHash near-dup, tier-aware)')
    df, dedup_rep = dedup_dataframe(df, text_col='text_clean', tier_col='tier',
                                    threshold=0.85)
    print(f'  kept {dedup_rep.n_kept}/{dedup_rep.n_input}'
          f' (exact_dropped={dedup_rep.n_dropped_exact},'
          f' near_dropped={dedup_rep.n_dropped_near})')

    # 4. Test-set carving
    print('[STEP 4] Carve stratified gold test set')
    train_pool, test_df = carve_test_set(df, args.target_test_size, args.seed)
    print(f'  test={len(test_df)} (per-source: '
          f'{test_df["source"].value_counts().to_dict()})')
    print(f'  train_pool={len(train_pool)}')

    # 5. Class balancing
    print('[STEP 5] Balance classes in train pool')
    train_balanced, bal_rep = balance_classes(train_pool, args.max_imbalance, args.seed)
    print(f'  before={bal_rep["before"]} -> after={bal_rep["after"]}')

    # 6. Combine + contamination check
    train_balanced['split'] = 'train'
    final_df = pd.concat([train_balanced, test_df], ignore_index=True)
    final_df['is_synthetic'] = False
    final_df['build_id'] = ts

    print('[STEP 6] Train/test contamination check')
    contam = contamination_check(train_balanced, test_df, text_col='text_clean',
                                 threshold=0.5)
    if not contam.empty:
        # Drop offending train rows
        bad_train = contam['train_idx'].unique()
        train_balanced = train_balanced.drop(index=bad_train).reset_index(drop=True)
        final_df = pd.concat([train_balanced, test_df], ignore_index=True)
        final_df['is_synthetic'] = False
        final_df['build_id'] = ts
    print(f'  contamination_pairs={len(contam)}')

    # 7. Manifest
    sources_info = {}
    for src in sorted(final_df['source'].dropna().unique()):
        sub = final_df[final_df['source'] == src]
        sources_info[src] = {
            'tier': _tier(src),
            'rows': int(len(sub)),
            'license': str(sub['license'].dropna().iloc[0]) if sub['license'].notna().any() else 'unknown',
        }
    train_lab = final_df[final_df['split'] == 'train']['label'].value_counts().to_dict()
    test_lab = final_df[final_df['split'] == 'test']['label'].value_counts().to_dict()
    imbalance = (max(train_lab.values()) / max(min(train_lab.values()), 1)
                 if train_lab else 0.0)
    manifest = {
        'build_id': ts,
        'built_at': datetime.now(timezone.utc).isoformat(),
        'total_rows': int(len(final_df)),
        'split_counts': final_df['split'].value_counts().to_dict(),
        'tier_counts': final_df['tier'].value_counts().to_dict(),
        'train_label_counts': train_lab,
        'test_label_counts': test_lab,
        'imbalance_ratio_train': float(imbalance),
        'sources': sources_info,
        'preprocessing': asdict(prep_rep),
        'dedup': {
            'n_dropped_exact': dedup_rep.n_dropped_exact,
            'n_dropped_near': dedup_rep.n_dropped_near,
            'sample_pairs': dedup_rep.near_dup_pairs_sample,
        },
        'balance': {
            'max_imbalance': args.max_imbalance,
            'report': bal_rep,
        },
        'contamination': {
            'threshold': 0.5,
            'n_pairs': int(len(contam)),
        },
        'config': {
            'seed': args.seed,
            'target_test_size': args.target_test_size,
        },
    }

    if args.dry_run:
        print('\n[DRY RUN] Manifest:')
        print(json.dumps(manifest, indent=2, default=str))
        return

    # 8. Write
    print('[STEP 7] Write canonical dataset + manifest + data card')
    out_dir = ensure_dir(PROJECT_ROOT / 'data' / 'processed' / 'latest')
    archive_dir = ensure_dir(PROJECT_ROOT / 'data' / 'processed' / 'archive')

    # archive prior canonical if any
    prior = out_dir / 'dataset.csv'
    if prior.exists():
        archive_path = archive_dir / f'dataset_pre_{ts}.csv'
        shutil.copy(str(prior), str(archive_path))
        print(f'  archived prior -> {archive_path.relative_to(PROJECT_ROOT)}')

    # column order
    cols = ['text', 'text_clean', 'text_clean_lower', 'label', 'label_raw',
            'agreement', 'is_uncertain', 'source', 'tier', 'split', 'language', 'n_tokens',
            'qc_flags', 'tickers', 'is_synthetic', 'content_sha256',
            'license', 'collected_at', 'build_id']
    for c in cols:
        if c not in final_df.columns:
            final_df[c] = pd.NA
    final_df = final_df[cols]

    dataset_path = out_dir / 'dataset.csv'
    final_df.to_csv(dataset_path, index=False)
    (out_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, default=str), encoding='utf-8')
    write_data_card(out_dir, manifest)

    # Also archive the new build for permanent record
    snap = archive_dir / f'dataset_{ts}.csv'
    shutil.copy(str(dataset_path), str(snap))

    # Save contamination report if non-empty
    if not contam.empty:
        contam.to_csv(out_dir / 'contamination_report.csv', index=False)

    # ── Backwards compat: keep the legacy latest.csv pointer in sync ───────
    legacy_path = PROJECT_ROOT / 'data' / 'processed' / 'latest.csv'
    final_df.rename(columns={'text_clean_lower': 'text_lower'}).to_csv(
        legacy_path, index=False)

    print(f'\n[DONE] dataset -> {dataset_path.relative_to(PROJECT_ROOT)}')
    print(f'[DONE] manifest -> {(out_dir / "manifest.json").relative_to(PROJECT_ROOT)}')
    print(f'[DONE] data_card -> {(out_dir / "data_card.md").relative_to(PROJECT_ROOT)}')
    print(f'[DONE] archive snapshot -> {snap.relative_to(PROJECT_ROOT)}')
    print(f'[DONE] backwards-compat latest.csv updated')


if __name__ == '__main__':
    main()
