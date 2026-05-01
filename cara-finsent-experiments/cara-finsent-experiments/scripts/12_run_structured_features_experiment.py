#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import argparse
import time

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from cara_finsent.data_utils import apply_max_rows, load_standardized_csv, set_global_seeds, split_dataframe
from cara_finsent.feature_extractor import FinancialFeatureExtractor
from cara_finsent.io_utils import save_dataframe, timestamp, write_manifest
from cara_finsent.metrics import classwise_metrics, confusion_matrix_df, metrics_with_optional_proba


def fit_transform_structured(train_text, test_text):
    extractor = FinancialFeatureExtractor()
    train_f = extractor.transform(train_text)
    test_f = extractor.transform(test_text)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_f)
    test_scaled = scaler.transform(test_f)
    return csr_matrix(train_scaled), csr_matrix(test_scaled), list(train_f.columns)


def main():
    parser = argparse.ArgumentParser(description='Compare TF-IDF-only vs TF-IDF + structured financial features.')
    parser.add_argument('--data', default=None, help='Standardized CSV. Auto-detected from data/processed/latest.csv if omitted.')
    parser.add_argument('--text_col', default=None)
    parser.add_argument('--label_col', default=None)
    parser.add_argument('--max_rows', type=int, default=None)
    parser.add_argument('--max_features', type=int, default=50000)
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

    vectorizer = TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2), stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(train_df['text'])
    X_test_tfidf = vectorizer.transform(test_df['text'])
    X_train_struct, X_test_struct, feature_names = fit_transform_structured(train_df['text'], test_df['text'])
    X_train_combo = hstack([X_train_tfidf, X_train_struct]).tocsr()
    X_test_combo = hstack([X_test_tfidf, X_test_struct]).tocsr()

    experiments = [
        ('tfidf_logistic_regression', LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=-1, random_state=args.seed), X_train_tfidf, X_test_tfidf),
        ('tfidf_structured_logistic_regression', LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=-1, random_state=args.seed), X_train_combo, X_test_combo),
        ('tfidf_linear_svm', LinearSVC(class_weight='balanced', random_state=args.seed), X_train_tfidf, X_test_tfidf),
        ('tfidf_structured_linear_svm', LinearSVC(class_weight='balanced', random_state=args.seed), X_train_combo, X_test_combo),
    ]
    rows = []
    preds = []
    files = {}
    for name, model, Xtr, Xte in experiments:
        print(f'[RUN] {name}')
        start = time.perf_counter()
        model.fit(Xtr, train_df['label'])
        y_pred = model.predict(Xte)
        elapsed = time.perf_counter() - start
        proba = model.predict_proba(Xte) if hasattr(model, 'predict_proba') else None
        m = metrics_with_optional_proba(test_df['label'], y_pred, proba, model_name=name)
        m['seconds'] = elapsed
        m['structured_features_used'] = 'structured' in name
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

    summary_path = save_dataframe(pd.DataFrame(rows).sort_values('macro_f1', ascending=False), args.results_dir, 'structured_features_summary', ts)
    pred_path = save_dataframe(pd.concat(preds, ignore_index=True), args.results_dir, 'structured_features_predictions', ts)
    feat_path = save_dataframe(pd.DataFrame({'structured_feature': feature_names}), args.results_dir, 'structured_feature_names', ts)
    files.update({'summary': str(summary_path), 'predictions': str(pred_path), 'feature_names': str(feat_path)})
    manifest = write_manifest(args.results_dir, 'structured_features_experiment', files, {'data': args.data}, ts)
    print(f'[DONE] Summary -> {summary_path}')
    print(f'[DONE] Manifest -> {manifest}')


if __name__ == '__main__':
    main()
