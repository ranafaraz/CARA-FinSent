#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import argparse
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from cara_finsent.data_utils import apply_max_rows, load_standardized_csv, set_global_seeds, split_dataframe
from cara_finsent.io_utils import save_dataframe, timestamp, write_manifest
from cara_finsent.metrics import classwise_metrics, confusion_matrix_df, metrics_with_optional_proba
from cara_finsent.retrieval import TfidfRetriever, augment_with_context


def main():
    parser = argparse.ArgumentParser(description='Test whether retrieval context improves sentiment classification.')
    parser.add_argument('--data', default=None, help='Standardized CSV. Auto-detected from data/processed/latest.csv if omitted.')
    parser.add_argument('--external_corpus_csv', default=None, help='Optional raw corpus with a text/headline/title column for retrieval context.')
    parser.add_argument('--text_col', default=None)
    parser.add_argument('--label_col', default=None)
    parser.add_argument('--max_rows', type=int, default=None)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', default='results')
    args = parser.parse_args()

    set_global_seeds(args.seed)
    if not args.data:
        from cara_finsent.data_utils import auto_detect_data
        args.data = str(auto_detect_data())
        print(f'[AUTO] data = {args.data}')
    ts = timestamp()
    df = apply_max_rows(load_standardized_csv(args.data, args.text_col, args.label_col), args.max_rows, seed=args.seed)
    train_df, val_df, test_df = split_dataframe(df, seed=args.seed)

    corpus = train_df['text'].astype(str).tolist()
    if args.external_corpus_csv:
        ext = load_standardized_csv(args.external_corpus_csv, allow_unlabeled=True)
        corpus += ext['text'].astype(str).tolist()
    retriever = TfidfRetriever(max_features=50000).fit(corpus)
    train_aug, train_ctx = augment_with_context(train_df['text'], retriever, top_k=args.top_k)
    test_aug, test_ctx = augment_with_context(test_df['text'], retriever, top_k=args.top_k)

    experiments = [
        ('no_retrieval_logistic_regression', train_df['text'], test_df['text'], LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=-1, random_state=args.seed)),
        ('retrieval_logistic_regression', train_aug, test_aug, LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=-1, random_state=args.seed)),
        ('no_retrieval_linear_svm', train_df['text'], test_df['text'], LinearSVC(class_weight='balanced', random_state=args.seed)),
        ('retrieval_linear_svm', train_aug, test_aug, LinearSVC(class_weight='balanced', random_state=args.seed)),
    ]
    rows = []
    preds = []
    files = {}
    for name, train_texts, test_texts, model in experiments:
        print(f'[RUN] {name}')
        pipe = Pipeline([('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words='english')), ('clf', model)])
        start = time.perf_counter()
        pipe.fit(train_texts, train_df['label'])
        y_pred = pipe.predict(test_texts)
        elapsed = time.perf_counter() - start
        proba = pipe.predict_proba(test_texts) if hasattr(pipe, 'predict_proba') else None
        m = metrics_with_optional_proba(test_df['label'], y_pred, proba, model_name=name)
        m['seconds'] = elapsed
        m['top_k'] = args.top_k
        m['seed'] = args.seed
        rows.append(m)
        pred_df = test_df[['id', 'text', 'label']].copy()
        pred_df['model'] = name
        pred_df['prediction'] = y_pred
        preds.append(pred_df)
        cm_path = save_dataframe(confusion_matrix_df(test_df['label'], y_pred).reset_index().rename(columns={'index': 'actual'}), args.results_dir, f'{name}_confusion_matrix', ts)
        files[f'{name}_cm'] = str(cm_path)
        cw_path = save_dataframe(classwise_metrics(test_df['label'], y_pred), args.results_dir, f'{name}_classwise_metrics', ts)
        files[f'{name}_classwise'] = str(cw_path)

    summary_path = save_dataframe(pd.DataFrame(rows).sort_values('macro_f1', ascending=False), args.results_dir, 'retrieval_experiment_summary', ts)
    pred_path = save_dataframe(pd.concat(preds, ignore_index=True), args.results_dir, 'retrieval_experiment_predictions', ts)
    ctx_path = save_dataframe(test_ctx, args.results_dir, 'retrieval_test_contexts', ts)
    files.update({'summary': str(summary_path), 'predictions': str(pred_path), 'test_contexts': str(ctx_path)})
    manifest = write_manifest(args.results_dir, 'retrieval_experiment', files, {'data': args.data, 'external_corpus_csv': args.external_corpus_csv, 'top_k': args.top_k}, ts)
    print(f'[DONE] Summary -> {summary_path}')
    print(f'[DONE] Manifest -> {manifest}')


if __name__ == '__main__':
    main()
