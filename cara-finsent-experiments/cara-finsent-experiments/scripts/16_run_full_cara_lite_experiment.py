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
from sklearn.svm import LinearSVC

from cara_finsent.data_utils import apply_max_rows, load_standardized_csv, set_global_seeds, split_dataframe
from cara_finsent.feature_extractor import FinancialFeatureExtractor
from cara_finsent.io_utils import save_dataframe, timestamp, write_manifest
from cara_finsent.metrics import abstention_curve, classwise_metrics, confusion_matrix_df, metrics_with_optional_proba, reliability_bins
from cara_finsent.plotting import save_confusion_matrix_plot, save_reliability_plot
from cara_finsent.retrieval import TfidfRetriever, augment_with_context


def make_base(base_name: str, seed: int):
    if base_name == 'logreg':
        return LogisticRegression(max_iter=3000, class_weight='balanced', n_jobs=-1, random_state=seed)
    if base_name == 'svm':
        return LinearSVC(class_weight='balanced', random_state=seed)
    raise ValueError(f'Unknown base_model {base_name!r}')


def make_calibrated(base, cv: int):
    try:
        return CalibratedClassifierCV(estimator=base, method='sigmoid', cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base, method='sigmoid', cv=cv)


def main():
    parser = argparse.ArgumentParser(description='Run CARA-lite pilot: retrieval + structured features + agreement weighting + calibration.')
    parser.add_argument('--data', default=None, help='Standardized CSV. Auto-detected from data/processed/latest.csv if omitted.')
    parser.add_argument('--external_corpus_csv', default=None)
    parser.add_argument('--text_col', default=None)
    parser.add_argument('--label_col', default=None)
    parser.add_argument('--agreement_col', default='agreement')
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--max_rows', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--base_model', choices=['logreg', 'svm'], default='svm')
    parser.add_argument('--min_weight', type=float, default=0.4)
    parser.add_argument('--weight_schedule', choices=['linear', 'quadratic'], default='linear')
    # Ablation toggles -- each disables exactly one CARA-lite component.
    parser.add_argument('--no_retrieval', action='store_true')
    parser.add_argument('--no_structured', action='store_true')
    parser.add_argument('--no_agreement', action='store_true')
    parser.add_argument('--no_calibration', action='store_true')
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--figures_dir', default='figures')
    args = parser.parse_args()

    set_global_seeds(args.seed)
    if not args.data:
        from cara_finsent.data_utils import auto_detect_data
        args.data = str(auto_detect_data())
        print(f'[AUTO] data = {args.data}')
    ts = timestamp()
    df = apply_max_rows(load_standardized_csv(args.data, args.text_col, args.label_col), args.max_rows, seed=args.seed)
    train_df, val_df, test_df = split_dataframe(df, seed=args.seed)

    # Retrieval
    # Self-retrieval from training corpus alone degrades performance (similar-topic texts
    # have mixed sentiments and pollute the input signal). Only enable retrieval when an
    # external corpus is provided so that retrieved contexts are domain knowledge rather
    # than resampled training data.
    if args.no_retrieval or not args.external_corpus_csv:
        if not args.no_retrieval and not args.external_corpus_csv:
            print('[INFO] No external_corpus_csv provided; disabling self-retrieval to avoid label-noise contamination. '
                  'Pass --external_corpus_csv <path> to enable retrieval from an external corpus.')
            args.no_retrieval = True
        train_texts = train_df['text'].tolist()
        test_texts = test_df['text'].tolist()
        test_ctx = pd.DataFrame()
    else:
        corpus = train_df['text'].astype(str).tolist()
        if args.external_corpus_csv:
            ext = load_standardized_csv(args.external_corpus_csv, allow_unlabeled=True)
            corpus += ext['text'].astype(str).tolist()
        retriever = TfidfRetriever(max_features=50000).fit(corpus)
        train_texts, _ = augment_with_context(train_df['text'], retriever, top_k=args.top_k)
        test_texts, test_ctx = augment_with_context(test_df['text'], retriever, top_k=args.top_k)

    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words='english')
    X_train_text = tfidf.fit_transform(train_texts)
    X_test_text = tfidf.transform(test_texts)

    # Structured features
    if args.no_structured:
        X_train, X_test = X_train_text, X_test_text
        feature_names: list[str] = []
    else:
        extractor = FinancialFeatureExtractor()
        train_struct = extractor.transform(train_df['text'])
        test_struct = extractor.transform(test_df['text'])
        scaler = StandardScaler()
        X_train_struct = csr_matrix(scaler.fit_transform(train_struct))
        X_test_struct = csr_matrix(scaler.transform(test_struct))
        X_train = hstack([X_train_text, X_train_struct]).tocsr()
        X_test = hstack([X_test_text, X_test_struct]).tocsr()
        feature_names = list(train_struct.columns)

    # Agreement weighting
    sample_weight = None
    used_agreement = False
    if not args.no_agreement and args.agreement_col in train_df.columns:
        agreement = pd.to_numeric(train_df[args.agreement_col], errors='coerce').fillna(1.0).clip(0.0, 1.0).values
        if args.weight_schedule == 'quadratic':
            agreement = agreement ** 2
        sample_weight = np.maximum(agreement, args.min_weight)
        used_agreement = True

    # Model + optional calibration
    base = make_base(args.base_model, args.seed)
    if args.no_calibration:
        model = base
    else:
        min_class_count = int(train_df['label'].value_counts().min())
        if min_class_count < 2:
            raise SystemExit('CARA-lite calibration requires at least 2 training examples per class. Use more data or reduce filtering.')
        cv = max(2, min(5, min_class_count))
        model = make_calibrated(base, cv)

    start = time.perf_counter()
    if sample_weight is not None:
        try:
            model.fit(X_train, train_df['label'], sample_weight=sample_weight)
        except (TypeError, ValueError):
            # CalibratedClassifierCV in some sklearn versions doesn't accept sample_weight
            print('[WARN] base estimator did not accept sample_weight; falling back to unweighted fit.')
            sample_weight = None
            used_agreement = False
            model.fit(X_train, train_df['label'])
    else:
        model.fit(X_train, train_df['label'])
    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
    else:
        y_proba = None
    elapsed = time.perf_counter() - start

    ablation_flags = ','.join([
        f'retrieval={"off" if args.no_retrieval else "on"}',
        f'structured={"off" if args.no_structured else "on"}',
        f'agreement={"off" if args.no_agreement else ("on" if used_agreement else "missing")}',
        f'calibration={"off" if args.no_calibration else "on"}',
    ])
    summary = metrics_with_optional_proba(test_df['label'], y_pred, y_proba, model_name=f'CARA_lite_{args.base_model}')
    summary['seconds'] = elapsed
    summary['top_k'] = args.top_k
    summary['used_agreement_weights'] = used_agreement
    summary['base_model'] = args.base_model
    summary['min_weight'] = args.min_weight
    summary['weight_schedule'] = args.weight_schedule
    summary['seed'] = args.seed
    summary['ablation_flags'] = ablation_flags
    summary_path = save_dataframe(pd.DataFrame([summary]), args.results_dir, 'cara_lite_summary', ts)
    # Append to a rolling ablation table (one row per run; safe across runs).
    save_dataframe(pd.DataFrame([summary]), args.results_dir, 'cara_lite_ablation_summary', ts)

    pred_df = test_df[['id', 'text', 'label']].copy()
    if args.agreement_col in test_df.columns:
        pred_df[args.agreement_col] = test_df[args.agreement_col].values
    pred_df['prediction'] = y_pred
    if y_proba is not None:
        for i, cls in enumerate(model.classes_):
            pred_df[f'proba_{cls}'] = y_proba[:, i]
    pred_path = save_dataframe(pred_df, args.results_dir, 'cara_lite_predictions', ts)

    cm = confusion_matrix_df(test_df['label'], y_pred)
    cm_path = save_dataframe(cm.reset_index().rename(columns={'index': 'actual'}), args.results_dir, 'cara_lite_confusion_matrix', ts)
    cw_path = save_dataframe(classwise_metrics(test_df['label'], y_pred), args.results_dir, 'cara_lite_classwise_metrics', ts)
    files = {'summary': str(summary_path), 'predictions': str(pred_path), 'confusion_matrix': str(cm_path), 'classwise': str(cw_path)}
    if y_proba is not None:
        bins = reliability_bins(test_df['label'], y_pred, y_proba)
        bins_path = save_dataframe(bins, args.results_dir, 'cara_lite_reliability_bins', ts)
        abst = abstention_curve(test_df['label'], y_pred, y_proba)
        abst_path = save_dataframe(abst, args.results_dir, 'cara_lite_abstention_curve', ts)
        rel_fig = f'{args.figures_dir}/cara_lite_reliability_{ts}.png'
        save_reliability_plot(bins, rel_fig, title='CARA-lite Reliability Diagram')
        files.update({'reliability_bins': str(bins_path), 'abstention_curve': str(abst_path), 'reliability_png': rel_fig})
    if not test_ctx.empty:
        ctx_path = save_dataframe(test_ctx, args.results_dir, 'cara_lite_retrieved_contexts', ts)
        files['retrieved_contexts'] = str(ctx_path)
    if feature_names:
        feature_path = save_dataframe(pd.DataFrame({'structured_feature': feature_names}), args.results_dir, 'cara_lite_structured_features', ts)
        files['structured_features'] = str(feature_path)
    cm_fig = f'{args.figures_dir}/cara_lite_confusion_matrix_{ts}.png'
    save_confusion_matrix_plot(cm, cm_fig, title='CARA-lite Confusion Matrix')
    files['confusion_matrix_png'] = cm_fig
    manifest = write_manifest(args.results_dir, 'cara_lite_experiment', files, {'data': args.data, 'external_corpus_csv': args.external_corpus_csv, 'ablation_flags': ablation_flags}, ts)
    print(pd.DataFrame([summary]))
    print(f'[DONE] Manifest -> {manifest}')


if __name__ == '__main__':
    main()
