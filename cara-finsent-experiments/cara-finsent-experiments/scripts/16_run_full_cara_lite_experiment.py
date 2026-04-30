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
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from cara_finsent.data_utils import apply_max_rows, load_standardized_csv, split_dataframe
from cara_finsent.feature_extractor import FinancialFeatureExtractor
from cara_finsent.io_utils import save_dataframe, timestamp, write_manifest
from cara_finsent.metrics import abstention_curve, classwise_metrics, confusion_matrix_df, metrics_with_optional_proba, reliability_bins
from cara_finsent.plotting import save_confusion_matrix_plot, save_reliability_plot
from cara_finsent.retrieval import TfidfRetriever, augment_with_context


def make_calibrated_lr(cv: int):
    base = LogisticRegression(max_iter=3000, class_weight='balanced', n_jobs=-1)
    try:
        return CalibratedClassifierCV(estimator=base, method='sigmoid', cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base, method='sigmoid', cv=cv)


def main():
    parser = argparse.ArgumentParser(description='Run CARA-lite pilot: retrieval + structured features + agreement weighting + calibration.')
    parser.add_argument('--data', required=True)
    parser.add_argument('--external_corpus_csv', default=None)
    parser.add_argument('--text_col', default=None)
    parser.add_argument('--label_col', default=None)
    parser.add_argument('--agreement_col', default='agreement')
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--max_rows', type=int, default=None)
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--figures_dir', default='figures')
    args = parser.parse_args()

    ts = timestamp()
    df = apply_max_rows(load_standardized_csv(args.data, args.text_col, args.label_col), args.max_rows)
    train_df, val_df, test_df = split_dataframe(df)

    corpus = train_df['text'].astype(str).tolist()
    if args.external_corpus_csv:
        ext = load_standardized_csv(args.external_corpus_csv, allow_unlabeled=True)
        corpus += ext['text'].astype(str).tolist()
    retriever = TfidfRetriever(max_features=50000).fit(corpus)
    train_aug, train_ctx = augment_with_context(train_df['text'], retriever, top_k=args.top_k)
    test_aug, test_ctx = augment_with_context(test_df['text'], retriever, top_k=args.top_k)

    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words='english')
    X_train_text = tfidf.fit_transform(train_aug)
    X_test_text = tfidf.transform(test_aug)

    extractor = FinancialFeatureExtractor()
    train_struct = extractor.transform(train_df['text'])
    test_struct = extractor.transform(test_df['text'])
    scaler = StandardScaler()
    X_train_struct = csr_matrix(scaler.fit_transform(train_struct))
    X_test_struct = csr_matrix(scaler.transform(test_struct))
    X_train = hstack([X_train_text, X_train_struct]).tocsr()
    X_test = hstack([X_test_text, X_test_struct]).tocsr()

    sample_weight = None
    if args.agreement_col in train_df.columns:
        agreement = pd.to_numeric(train_df[args.agreement_col], errors='coerce').fillna(1.0).clip(0.4, 1.0)
        sample_weight = agreement.values

    min_class_count = int(train_df['label'].value_counts().min())
    if min_class_count < 2:
        raise SystemExit('CARA-lite calibration requires at least 2 training examples per class. Use more data or reduce filtering.')
    cv = max(2, min(5, min_class_count))
    model = make_calibrated_lr(cv)
    start = time.perf_counter()
    if sample_weight is not None:
        model.fit(X_train, train_df['label'], sample_weight=sample_weight)
    else:
        model.fit(X_train, train_df['label'])
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    elapsed = time.perf_counter() - start

    summary = metrics_with_optional_proba(test_df['label'], y_pred, y_proba, model_name='CARA_lite_retrieval_structured_agreement_calibrated')
    summary['seconds'] = elapsed
    summary['top_k'] = args.top_k
    summary['used_agreement_weights'] = sample_weight is not None
    summary_path = save_dataframe(pd.DataFrame([summary]), args.results_dir, 'cara_lite_summary', ts)

    pred_df = test_df[['id', 'text', 'label']].copy()
    if args.agreement_col in test_df.columns:
        pred_df[args.agreement_col] = test_df[args.agreement_col].values
    pred_df['prediction'] = y_pred
    for i, cls in enumerate(model.classes_):
        pred_df[f'proba_{cls}'] = y_proba[:, i]
    pred_path = save_dataframe(pred_df, args.results_dir, 'cara_lite_predictions', ts)

    cm = confusion_matrix_df(test_df['label'], y_pred)
    cm_path = save_dataframe(cm.reset_index().rename(columns={'index': 'actual'}), args.results_dir, 'cara_lite_confusion_matrix', ts)
    cw_path = save_dataframe(classwise_metrics(test_df['label'], y_pred), args.results_dir, 'cara_lite_classwise_metrics', ts)
    bins = reliability_bins(test_df['label'], y_pred, y_proba)
    bins_path = save_dataframe(bins, args.results_dir, 'cara_lite_reliability_bins', ts)
    abst = abstention_curve(test_df['label'], y_pred, y_proba)
    abst_path = save_dataframe(abst, args.results_dir, 'cara_lite_abstention_curve', ts)
    ctx_path = save_dataframe(test_ctx, args.results_dir, 'cara_lite_retrieved_contexts', ts)
    feature_path = save_dataframe(pd.DataFrame({'structured_feature': list(train_struct.columns)}), args.results_dir, 'cara_lite_structured_features', ts)
    cm_fig = f'{args.figures_dir}/cara_lite_confusion_matrix_{ts}.png'
    rel_fig = f'{args.figures_dir}/cara_lite_reliability_{ts}.png'
    save_confusion_matrix_plot(cm, cm_fig, title='CARA-lite Confusion Matrix')
    save_reliability_plot(bins, rel_fig, title='CARA-lite Reliability Diagram')
    manifest = write_manifest(args.results_dir, 'cara_lite_experiment', {'summary': str(summary_path), 'predictions': str(pred_path), 'confusion_matrix': str(cm_path), 'classwise': str(cw_path), 'reliability_bins': str(bins_path), 'abstention_curve': str(abst_path), 'retrieved_contexts': str(ctx_path), 'structured_features': str(feature_path), 'confusion_matrix_png': cm_fig, 'reliability_png': rel_fig}, {'data': args.data, 'external_corpus_csv': args.external_corpus_csv}, ts)
    print(pd.DataFrame([summary]))
    print(f'[DONE] Manifest -> {manifest}')


if __name__ == '__main__':
    main()
