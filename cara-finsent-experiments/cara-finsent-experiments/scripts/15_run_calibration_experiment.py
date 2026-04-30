#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import argparse
import time

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from cara_finsent.data_utils import apply_max_rows, load_standardized_csv, split_dataframe
from cara_finsent.io_utils import save_dataframe, timestamp, write_manifest
from cara_finsent.metrics import abstention_curve, classwise_metrics, confusion_matrix_df, metrics_with_optional_proba, reliability_bins
from cara_finsent.plotting import save_reliability_plot


def calibrated_estimator(base_name: str, method: str, cv: int):
    if base_name == 'linear_svm':
        base = LinearSVC(class_weight='balanced')
    elif base_name == 'logistic_regression':
        base = LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=-1)
    else:
        raise ValueError(base_name)
    try:
        return CalibratedClassifierCV(estimator=base, method=method, cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base, method=method, cv=cv)


def main():
    parser = argparse.ArgumentParser(description='Run calibration and abstention experiments.')
    parser.add_argument('--data', required=True)
    parser.add_argument('--text_col', default=None)
    parser.add_argument('--label_col', default=None)
    parser.add_argument('--base_model', choices=['logistic_regression', 'linear_svm'], default='linear_svm')
    parser.add_argument('--method', choices=['sigmoid', 'isotonic'], default='sigmoid')
    parser.add_argument('--max_rows', type=int, default=None)
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--figures_dir', default='figures')
    args = parser.parse_args()

    ts = timestamp()
    df = apply_max_rows(load_standardized_csv(args.data, args.text_col, args.label_col), args.max_rows)
    train_df, val_df, test_df = split_dataframe(df)

    min_class_count = int(train_df['label'].value_counts().min())
    if min_class_count < 2:
        raise SystemExit('Calibration requires at least 2 training examples per class. Use more data or reduce filtering.')
    cv = max(2, min(5, min_class_count))
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words='english')),
        ('clf', calibrated_estimator(args.base_model, args.method, cv=cv)),
    ])
    start = time.perf_counter()
    model.fit(train_df['text'], train_df['label'])
    y_pred = model.predict(test_df['text'])
    y_proba = model.predict_proba(test_df['text'])
    elapsed = time.perf_counter() - start

    summary = metrics_with_optional_proba(test_df['label'], y_pred, y_proba, model_name=f'calibrated_{args.base_model}_{args.method}')
    summary['seconds'] = elapsed
    summary['calibration_method'] = args.method
    summary_path = save_dataframe(pd.DataFrame([summary]), args.results_dir, 'calibration_summary', ts)

    pred_df = test_df[['id', 'text', 'label']].copy()
    pred_df['prediction'] = y_pred
    classes = list(model.named_steps['clf'].classes_)
    for i, cls in enumerate(classes):
        pred_df[f'proba_{cls}'] = y_proba[:, i]
    pred_path = save_dataframe(pred_df, args.results_dir, 'calibration_predictions', ts)

    bins = reliability_bins(test_df['label'], y_pred, y_proba)
    bins_path = save_dataframe(bins, args.results_dir, 'calibration_reliability_bins', ts)
    abst = abstention_curve(test_df['label'], y_pred, y_proba)
    abst_path = save_dataframe(abst, args.results_dir, 'calibration_abstention_curve', ts)
    cm_path = save_dataframe(confusion_matrix_df(test_df['label'], y_pred).reset_index().rename(columns={'index': 'actual'}), args.results_dir, 'calibration_confusion_matrix', ts)
    cw_path = save_dataframe(classwise_metrics(test_df['label'], y_pred), args.results_dir, 'calibration_classwise_metrics', ts)
    fig_path = f'{args.figures_dir}/calibration_reliability_{ts}.png'
    save_reliability_plot(bins, fig_path, title='Calibration Reliability Diagram')
    manifest = write_manifest(args.results_dir, 'calibration_experiment', {'summary': str(summary_path), 'predictions': str(pred_path), 'reliability_bins': str(bins_path), 'abstention_curve': str(abst_path), 'confusion_matrix': str(cm_path), 'classwise': str(cw_path), 'figure': fig_path}, {'data': args.data, 'base_model': args.base_model, 'method': args.method, 'cv': cv}, ts)
    print(pd.DataFrame([summary]))
    print(f'[DONE] Manifest -> {manifest}')


if __name__ == '__main__':
    main()
