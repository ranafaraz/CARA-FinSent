#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import argparse
import time

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from cara_finsent.data_utils import apply_max_rows, load_standardized_csv, split_dataframe
from cara_finsent.io_utils import save_dataframe, timestamp, write_manifest
from cara_finsent.metrics import classwise_metrics, confusion_matrix_df, metrics_with_optional_proba
from cara_finsent.plotting import save_confusion_matrix_plot, save_metric_bar_plot


def get_models(include_xgboost: bool = True):
    models = {
        'majority_baseline': DummyClassifier(strategy='most_frequent'),
        'tfidf_logistic_regression': LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=-1),
        'tfidf_linear_svm': LinearSVC(class_weight='balanced'),
        'tfidf_sgd_log_loss': SGDClassifier(loss='log_loss', class_weight='balanced', random_state=42),
        'tfidf_multinomial_nb': MultinomialNB(),
        'tfidf_random_forest': RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1),
    }
    if include_xgboost:
        try:
            from xgboost import XGBClassifier
            from sklearn.preprocessing import LabelEncoder
            models['tfidf_xgboost'] = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, eval_metric='mlogloss', random_state=42, n_jobs=-1)
        except Exception:
            print('[WARN] xgboost unavailable; skipping XGBoost.')
    return models


def main():
    parser = argparse.ArgumentParser(description='Run classical TF-IDF baselines and save timestamped CSV results.')
    parser.add_argument('--data', required=True, help='Standardized CSV with text and label columns.')
    parser.add_argument('--text_col', default=None)
    parser.add_argument('--label_col', default=None)
    parser.add_argument('--max_rows', type=int, default=None)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--max_features', type=int, default=50000)
    parser.add_argument('--ngram_max', type=int, default=2)
    parser.add_argument('--no_xgboost', action='store_true')
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--figures_dir', default='figures')
    args = parser.parse_args()

    ts = timestamp()
    df = load_standardized_csv(args.data, args.text_col, args.label_col)
    df = apply_max_rows(df, args.max_rows)
    train_df, val_df, test_df = split_dataframe(df, test_size=args.test_size, val_size=args.val_size)
    print(f'[INFO] Rows train/val/test: {len(train_df)}/{len(val_df)}/{len(test_df)}')

    rows = []
    prediction_frames = []
    output_files = {}
    for name, model in get_models(include_xgboost=not args.no_xgboost).items():
        print(f'[RUN] {name}')
        start = time.perf_counter()
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=args.max_features, ngram_range=(1, args.ngram_max), stop_words='english')),
            ('clf', model),
        ])
        if name == 'tfidf_xgboost':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train = le.fit_transform(train_df['label'])
            pipe.fit(train_df['text'], y_train)
            pred_ids = pipe.predict(test_df['text'])
            y_pred = le.inverse_transform(pred_ids)
            y_proba = pipe.predict_proba(test_df['text']) if hasattr(pipe, 'predict_proba') else None
        else:
            pipe.fit(train_df['text'], train_df['label'])
            y_pred = pipe.predict(test_df['text'])
            y_proba = pipe.predict_proba(test_df['text']) if hasattr(pipe, 'predict_proba') else None
        elapsed = time.perf_counter() - start
        metrics = metrics_with_optional_proba(test_df['label'].values, y_pred, y_proba, model_name=name)
        metrics['train_plus_infer_seconds'] = elapsed
        metrics['train_rows'] = len(train_df)
        metrics['test_rows'] = len(test_df)
        rows.append(metrics)

        pred_df = test_df[['id', 'text', 'label']].copy()
        pred_df['model'] = name
        pred_df['prediction'] = y_pred
        if y_proba is not None:
            for i, cls in enumerate(getattr(pipe.named_steps['clf'], 'classes_', ['negative', 'neutral', 'positive'])):
                cls_name = str(cls) if not isinstance(cls, (int, float)) else ['negative', 'neutral', 'positive'][int(cls)]
                pred_df[f'proba_{cls_name}'] = y_proba[:, i]
        prediction_frames.append(pred_df)

        cm = confusion_matrix_df(test_df['label'], y_pred)
        cm_path = save_dataframe(cm.reset_index().rename(columns={'index': 'actual'}), args.results_dir, f'{name}_confusion_matrix', ts)
        output_files[f'{name}_confusion_matrix'] = str(cm_path)
        fig_path = f'{args.figures_dir}/{name}_confusion_matrix_{ts}.png'
        save_confusion_matrix_plot(cm, fig_path, title=f'{name} Confusion Matrix')
        output_files[f'{name}_confusion_matrix_png'] = fig_path

        cw_path = save_dataframe(classwise_metrics(test_df['label'], y_pred), args.results_dir, f'{name}_classwise_metrics', ts)
        output_files[f'{name}_classwise_metrics'] = str(cw_path)

    results_df = pd.DataFrame(rows).sort_values('macro_f1', ascending=False)
    results_path = save_dataframe(results_df, args.results_dir, 'classical_baseline_summary', ts)
    predictions_path = save_dataframe(pd.concat(prediction_frames, ignore_index=True), args.results_dir, 'classical_baseline_predictions', ts)
    output_files['summary'] = str(results_path)
    output_files['predictions'] = str(predictions_path)
    metric_plot = f'{args.figures_dir}/classical_baseline_macro_f1_{ts}.png'
    save_metric_bar_plot(results_df, 'macro_f1', metric_plot)
    output_files['macro_f1_plot'] = metric_plot
    manifest = write_manifest(args.results_dir, 'classical_baselines', output_files, metadata={'data': args.data, 'rows': len(df)}, ts=ts)
    print(results_df)
    print(f'[DONE] Summary -> {results_path}')
    print(f'[DONE] Manifest -> {manifest}')


if __name__ == '__main__':
    main()
