#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import argparse

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from cara_finsent.data_utils import (
    infer_dataset_name,
    load_gold_split,
    set_global_seeds,
    split_label_distribution,
    text_hash_leakage_count,
)
from cara_finsent.io_utils import git_commit_sha, save_dataframe, timestamp, write_manifest
from cara_finsent.metrics import classwise_metrics, confusion_matrix_df, metrics_with_optional_proba
from cara_finsent.plotting import save_confusion_matrix_plot, save_metric_bar_plot


def get_models(seed: int = 42):
    return {
        'majority_baseline_transfer': DummyClassifier(strategy='most_frequent', random_state=seed),
        'tfidf_logistic_regression_transfer': LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=-1, random_state=seed),
        'tfidf_linear_svm_transfer': LinearSVC(class_weight='balanced', random_state=seed),
    }


def main():
    parser = argparse.ArgumentParser(description='Run explicit cross-domain transfer evaluation on controlled gold splits.')
    parser.add_argument('--train', required=True, help='Source controlled gold split CSV.')
    parser.add_argument('--test', required=True, help='Target controlled gold split CSV.')
    parser.add_argument('--max_features', type=int, default=50000)
    parser.add_argument('--ngram_max', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--figures_dir', default='figures')
    args = parser.parse_args()

    set_global_seeds(args.seed)
    train_dataset_name = infer_dataset_name(args.train)
    test_dataset_name = infer_dataset_name(args.test)
    if train_dataset_name == 'unknown' or test_dataset_name == 'unknown':
        raise SystemExit('Could not infer dataset names from --train/--test paths.')

    source_train_df, source_val_df, _source_test_df = load_gold_split(args.train)
    _target_train_df, _target_val_df, target_test_df = load_gold_split(args.test)

    ts = timestamp()
    benchmark_mode = f'{train_dataset_name}_to_{test_dataset_name}'
    split_source = 'controlled_gold_split'
    label_dist = split_label_distribution(source_train_df, source_val_df, target_test_df)
    leakage_count = text_hash_leakage_count(source_train_df, source_val_df, target_test_df)
    git_sha = git_commit_sha(PROJECT_ROOT)

    rows = []
    prediction_frames = []
    output_files = {}
    for name, model in get_models(seed=args.seed).items():
        print(f'[RUN] {name}')
        start = time.perf_counter()
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=args.max_features, ngram_range=(1, args.ngram_max), stop_words='english')),
            ('clf', model),
        ])
        pipe.fit(source_train_df['text'], source_train_df['label'])
        y_pred = pipe.predict(target_test_df['text'])
        y_proba = pipe.predict_proba(target_test_df['text']) if hasattr(pipe, 'predict_proba') else None
        elapsed = time.perf_counter() - start

        metrics = metrics_with_optional_proba(target_test_df['label'].values, y_pred, y_proba, model_name=name)
        metrics['dataset_name'] = benchmark_mode
        metrics['dataset_file'] = json.dumps({'train': str(Path(args.train)), 'test': str(Path(args.test))}, sort_keys=True)
        metrics['split_source'] = split_source
        metrics['benchmark_mode'] = benchmark_mode
        metrics['source_dataset_name'] = train_dataset_name
        metrics['target_dataset_name'] = test_dataset_name
        metrics['train_rows'] = len(source_train_df)
        metrics['val_rows'] = len(source_val_df)
        metrics['test_rows'] = len(target_test_df)
        metrics['label_distribution_train'] = json.dumps(label_dist['train'], sort_keys=True)
        metrics['label_distribution_val'] = json.dumps(label_dist['val'], sort_keys=True)
        metrics['label_distribution_test'] = json.dumps(label_dist['test'], sort_keys=True)
        metrics['text_hash_leakage_count'] = leakage_count
        metrics['seed'] = args.seed
        metrics['git_commit_sha'] = git_sha
        metrics['train_plus_infer_seconds'] = elapsed
        rows.append(metrics)

        pred_df = target_test_df[['id', 'text', 'label']].copy()
        pred_df['model'] = name
        pred_df['dataset_name'] = benchmark_mode
        pred_df['split_source'] = split_source
        pred_df['prediction'] = y_pred
        if y_proba is not None:
            for index, cls in enumerate(getattr(pipe.named_steps['clf'], 'classes_', [])):
                pred_df[f'proba_{cls}'] = y_proba[:, index]
        prediction_frames.append(pred_df)

        cm = confusion_matrix_df(target_test_df['label'], y_pred)
        cm_path = save_dataframe(cm.reset_index().rename(columns={'index': 'actual'}), args.results_dir, f'{name}_cross_domain_confusion_matrix', ts)
        output_files[f'{name}_confusion_matrix'] = str(cm_path)
        fig_path = f'{args.figures_dir}/{name}_cross_domain_confusion_matrix_{ts}.png'
        save_confusion_matrix_plot(cm, fig_path, title=f'{name} Cross-Domain Confusion Matrix')
        output_files[f'{name}_confusion_matrix_png'] = fig_path
        cw_path = save_dataframe(classwise_metrics(target_test_df['label'], y_pred), args.results_dir, f'{name}_cross_domain_classwise_metrics', ts)
        output_files[f'{name}_classwise_metrics'] = str(cw_path)

    results_df = pd.DataFrame(rows).sort_values('macro_f1', ascending=False)
    results_path = save_dataframe(results_df, args.results_dir, 'cross_domain_summary', ts)
    predictions_path = save_dataframe(pd.concat(prediction_frames, ignore_index=True), args.results_dir, 'cross_domain_predictions', ts)
    output_files['summary'] = str(results_path)
    output_files['predictions'] = str(predictions_path)
    metric_plot = f'{args.figures_dir}/cross_domain_macro_f1_{ts}.png'
    save_metric_bar_plot(results_df, 'macro_f1', metric_plot)
    output_files['macro_f1_plot'] = metric_plot
    manifest = write_manifest(
        args.results_dir,
        'cross_domain_eval',
        output_files,
        metadata={
            'dataset_name': benchmark_mode,
            'dataset_file': {'train': str(Path(args.train)), 'test': str(Path(args.test))},
            'split_source': split_source,
            'benchmark_mode': benchmark_mode,
            'source_dataset_name': train_dataset_name,
            'target_dataset_name': test_dataset_name,
            'train_rows': len(source_train_df),
            'val_rows': len(source_val_df),
            'test_rows': len(target_test_df),
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