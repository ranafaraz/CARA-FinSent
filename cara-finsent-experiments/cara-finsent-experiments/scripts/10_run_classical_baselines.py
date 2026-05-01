#!/usr/bin/env python3
from __future__ import annotations

import json
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

from cara_finsent.data_utils import (
    auto_detect_gold_split,
    infer_dataset_name,
    load_gold_split,
    set_global_seeds,
    split_label_distribution,
    text_hash_leakage_count,
)
from cara_finsent.io_utils import git_commit_sha, save_dataframe, timestamp, write_manifest
from cara_finsent.metrics import classwise_metrics, confusion_matrix_df, metrics_with_optional_proba
from cara_finsent.plotting import save_confusion_matrix_plot, save_metric_bar_plot


def get_models(include_xgboost: bool = True, seed: int = 42):
    models = {
        'majority_baseline': DummyClassifier(strategy='most_frequent', random_state=seed),
        'tfidf_logistic_regression': LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=-1, random_state=seed),
        'tfidf_linear_svm': LinearSVC(class_weight='balanced', random_state=seed),
        'tfidf_sgd_log_loss': SGDClassifier(loss='log_loss', class_weight='balanced', random_state=seed),
        'tfidf_multinomial_nb': MultinomialNB(),
        'tfidf_random_forest': RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=seed, n_jobs=-1),
    }
    if include_xgboost:
        try:
            from xgboost import XGBClassifier
            models['tfidf_xgboost'] = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, eval_metric='mlogloss', random_state=seed, n_jobs=-1)
        except Exception:
            print('[WARN] xgboost unavailable; skipping XGBoost.')
    return models


def main():
    parser = argparse.ArgumentParser(description='Run classical TF-IDF baselines and save timestamped CSV results.')
    parser.add_argument('--data', default=None, help='Controlled gold split CSV. Auto-detected from data/processed/gold if omitted.')
    parser.add_argument('--dataset_name', default=None, choices=['phrasebank', 'fiqa'])
    parser.add_argument('--text_col', default=None)
    parser.add_argument('--label_col', default=None)
    parser.add_argument('--max_rows', type=int, default=None)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--max_features', type=int, default=50000)
    parser.add_argument('--ngram_max', type=int, default=2)
    parser.add_argument('--no_xgboost', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--figures_dir', default='figures')
    args = parser.parse_args()

    set_global_seeds(args.seed)
    dataset_name = args.dataset_name or infer_dataset_name(args.data)
    if not args.data:
        if dataset_name == 'unknown':
            raise SystemExit('Provide --dataset_name when --data is omitted. Allowed values: phrasebank, fiqa.')
        args.data = str(auto_detect_gold_split(dataset_name))
        print(f'[AUTO] data = {args.data}')
    if dataset_name == 'unknown':
        dataset_name = infer_dataset_name(args.data)
    if dataset_name == 'unknown':
        raise SystemExit('Could not infer dataset_name from --data. Pass --dataset_name explicitly.')

    ts = timestamp()
    train_df, val_df, test_df = load_gold_split(args.data)
    if args.max_rows:
        print('[WARN] --max_rows is ignored for controlled gold split inputs.')
    print(f'[INFO] Controlled gold split: train/val/test = {len(train_df)}/{len(val_df)}/{len(test_df)}')

    split_source = 'controlled_gold_split'
    benchmark_mode = f'{dataset_name}_in_domain'
    label_dist = split_label_distribution(train_df, val_df, test_df)
    leakage_count = text_hash_leakage_count(train_df, val_df, test_df)
    git_sha = git_commit_sha(PROJECT_ROOT)

    rows = []
    prediction_frames = []
    output_files = {}
    for name, model in get_models(include_xgboost=not args.no_xgboost, seed=args.seed).items():
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
        metrics['dataset_name'] = dataset_name
        metrics['dataset_file'] = str(Path(args.data))
        metrics['split_source'] = split_source
        metrics['benchmark_mode'] = benchmark_mode
        metrics['train_plus_infer_seconds'] = elapsed
        metrics['train_rows'] = len(train_df)
        metrics['val_rows'] = len(val_df)
        metrics['test_rows'] = len(test_df)
        metrics['label_distribution_train'] = json.dumps(label_dist['train'], sort_keys=True)
        metrics['label_distribution_val'] = json.dumps(label_dist['val'], sort_keys=True)
        metrics['label_distribution_test'] = json.dumps(label_dist['test'], sort_keys=True)
        metrics['text_hash_leakage_count'] = leakage_count
        metrics['seed'] = args.seed
        metrics['git_commit_sha'] = git_sha
        rows.append(metrics)

        pred_df = test_df[['id', 'text', 'label']].copy()
        pred_df['model'] = name
        pred_df['dataset_name'] = dataset_name
        pred_df['split_source'] = split_source
        pred_df['prediction'] = y_pred
        if y_proba is not None:
            if name == 'tfidf_xgboost':
                proba_classes = list(le.classes_)
            else:
                proba_classes = list(getattr(pipe.named_steps['clf'], 'classes_', ['negative', 'neutral', 'positive']))
            for i, cls in enumerate(proba_classes):
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
    manifest = write_manifest(
        args.results_dir,
        'classical_baselines',
        output_files,
        metadata={
            'dataset_name': dataset_name,
            'dataset_file': str(Path(args.data)),
            'split_source': split_source,
            'benchmark_mode': benchmark_mode,
            'train_rows': len(train_df),
            'val_rows': len(val_df),
            'test_rows': len(test_df),
            'label_distribution': label_dist,
            'text_hash_leakage_count': leakage_count,
            'seed': args.seed,
            'git_commit_sha': git_sha,
        },
        ts=ts,
    )
    print(results_df)
    print(f'[DONE] Summary -> {results_path}')
    print(f'[DONE] Manifest -> {manifest}')


if __name__ == '__main__':
    main()
