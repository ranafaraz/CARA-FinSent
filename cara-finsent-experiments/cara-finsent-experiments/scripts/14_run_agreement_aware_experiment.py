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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from cara_finsent.data_utils import apply_max_rows, load_standardized_csv, split_dataframe
from cara_finsent.io_utils import save_dataframe, timestamp, write_manifest
from cara_finsent.metrics import classwise_metrics, confusion_matrix_df, metrics_with_optional_proba


def agreement_weights(values, min_weight=0.4):
    v = pd.to_numeric(values, errors='coerce').fillna(1.0).astype(float)
    v = v.clip(lower=0.0, upper=1.0)
    return np.maximum(v.values, min_weight)


def main():
    parser = argparse.ArgumentParser(description='Run agreement-aware sample-weight experiments using PhraseBank agreement levels.')
    parser.add_argument('--data', required=True)
    parser.add_argument('--text_col', default=None)
    parser.add_argument('--label_col', default=None)
    parser.add_argument('--agreement_col', default='agreement')
    parser.add_argument('--max_rows', type=int, default=None)
    parser.add_argument('--results_dir', default='results')
    args = parser.parse_args()

    ts = timestamp()
    df = apply_max_rows(load_standardized_csv(args.data, args.text_col, args.label_col), args.max_rows)
    if args.agreement_col not in df.columns:
        raise SystemExit(f'Missing agreement column {args.agreement_col!r}. Run 00_prepare_phrasebank_fiqa.py first or provide a CSV with agreement values.')
    train_df, val_df, test_df = split_dataframe(df)
    weights = agreement_weights(train_df[args.agreement_col])

    experiments = [
        ('unweighted_logistic_regression', LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=-1), None),
        ('agreement_weighted_logistic_regression', LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=-1), weights),
        ('unweighted_linear_svm', LinearSVC(class_weight='balanced'), None),
        ('agreement_weighted_linear_svm', LinearSVC(class_weight='balanced'), weights),
    ]
    rows = []
    subset_rows = []
    preds = []
    files = {}
    for name, model, sample_weight in experiments:
        print(f'[RUN] {name}')
        pipe = Pipeline([('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words='english')), ('clf', model)])
        start = time.perf_counter()
        if sample_weight is None:
            pipe.fit(train_df['text'], train_df['label'])
        else:
            pipe.fit(train_df['text'], train_df['label'], clf__sample_weight=sample_weight)
        y_pred = pipe.predict(test_df['text'])
        elapsed = time.perf_counter() - start
        proba = pipe.predict_proba(test_df['text']) if hasattr(pipe, 'predict_proba') else None
        m = metrics_with_optional_proba(test_df['label'], y_pred, proba, model_name=name)
        m['seconds'] = elapsed
        rows.append(m)
        pred_df = test_df[['id', 'text', 'label', args.agreement_col]].copy()
        pred_df['model'] = name
        pred_df['prediction'] = y_pred
        preds.append(pred_df)
        for subset_name, mask in {
            'low_agreement_lt_075': pd.to_numeric(test_df[args.agreement_col], errors='coerce').fillna(1.0) < 0.75,
            'high_agreement_ge_075': pd.to_numeric(test_df[args.agreement_col], errors='coerce').fillna(1.0) >= 0.75,
        }.items():
            if mask.any():
                sm = metrics_with_optional_proba(test_df.loc[mask, 'label'], np.array(y_pred)[mask.values], None, model_name=name)
                sm['subset'] = subset_name
                sm['subset_rows'] = int(mask.sum())
                subset_rows.append(sm)
        cm_path = save_dataframe(confusion_matrix_df(test_df['label'], y_pred).reset_index().rename(columns={'index': 'actual'}), args.results_dir, f'{name}_confusion_matrix', ts)
        files[f'{name}_cm'] = str(cm_path)
        cw_path = save_dataframe(classwise_metrics(test_df['label'], y_pred), args.results_dir, f'{name}_classwise_metrics', ts)
        files[f'{name}_classwise'] = str(cw_path)

    summary_path = save_dataframe(pd.DataFrame(rows).sort_values('macro_f1', ascending=False), args.results_dir, 'agreement_aware_summary', ts)
    subset_path = save_dataframe(pd.DataFrame(subset_rows), args.results_dir, 'agreement_subset_metrics', ts)
    pred_path = save_dataframe(pd.concat(preds, ignore_index=True), args.results_dir, 'agreement_aware_predictions', ts)
    files.update({'summary': str(summary_path), 'subset_metrics': str(subset_path), 'predictions': str(pred_path)})
    manifest = write_manifest(args.results_dir, 'agreement_aware_experiment', files, {'data': args.data}, ts)
    print(f'[DONE] Summary -> {summary_path}')
    print(f'[DONE] Manifest -> {manifest}')


if __name__ == '__main__':
    main()
