from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.preprocessing import label_binarize

from .data_utils import STANDARD_LABELS


def classification_metrics(y_true, y_pred, labels: Optional[List[str]] = None) -> Dict[str, float]:
    labels = labels or STANDARD_LABELS
    kappa = float(cohen_kappa_score(y_true, y_pred))
    if np.isnan(kappa):
        kappa = 0.0
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)),
        'macro_precision': float(precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        'weighted_precision': float(precision_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)),
        'weighted_recall': float(recall_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
        'cohen_kappa': kappa,
    }


def classwise_metrics(y_true, y_pred, labels: Optional[List[str]] = None) -> pd.DataFrame:
    labels = labels or STANDARD_LABELS
    rows = []
    for label in labels:
        yt = np.array(y_true) == label
        yp = np.array(y_pred) == label
        tp = int(np.sum(yt & yp))
        fp = int(np.sum(~yt & yp))
        fn = int(np.sum(yt & ~yp))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({'class': label, 'precision': precision, 'recall': recall, 'f1': f1, 'support': int(np.sum(yt))})
    return pd.DataFrame(rows)


def confusion_matrix_df(y_true, y_pred, labels: Optional[List[str]] = None) -> pd.DataFrame:
    labels = labels or STANDARD_LABELS
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f'actual_{x}' for x in labels], columns=[f'pred_{x}' for x in labels])


def multiclass_brier_score(y_true, y_proba, labels: Optional[List[str]] = None) -> float:
    labels = labels or STANDARD_LABELS
    y_bin = label_binarize(y_true, classes=labels)
    proba = np.asarray(y_proba)
    if proba.shape[1] != len(labels):
        raise ValueError(f'Expected probability shape (_, {len(labels)}), got {proba.shape}')
    return float(np.mean(np.sum((proba - y_bin) ** 2, axis=1)))


def expected_calibration_error(y_true, y_pred, y_proba, n_bins: int = 10) -> float:
    proba = np.asarray(y_proba)
    conf = proba.max(axis=1)
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            ece += (mask.mean()) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def reliability_bins(y_true, y_pred, y_proba, n_bins: int = 10) -> pd.DataFrame:
    proba = np.asarray(y_proba)
    conf = proba.max(axis=1)
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        rows.append({
            'bin_id': i,
            'bin_lower': lo,
            'bin_upper': hi,
            'count': int(mask.sum()),
            'coverage': float(mask.mean()),
            'mean_confidence': float(conf[mask].mean()) if mask.any() else 0.0,
            'accuracy': float(correct[mask].mean()) if mask.any() else 0.0,
            'abs_gap': float(abs(correct[mask].mean() - conf[mask].mean())) if mask.any() else 0.0,
        })
    return pd.DataFrame(rows)


def abstention_curve(y_true, y_pred, y_proba, thresholds: Optional[Iterable[float]] = None) -> pd.DataFrame:
    thresholds = list(thresholds) if thresholds is not None else [round(x, 2) for x in np.arange(0.0, 0.96, 0.05)]
    proba = np.asarray(y_proba)
    conf = proba.max(axis=1)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rows = []
    for t in thresholds:
        keep = conf >= t
        coverage = keep.mean()
        if keep.any():
            metrics = classification_metrics(y_true[keep], y_pred[keep])
            acc = metrics['accuracy']
            macro_f1 = metrics['macro_f1']
        else:
            acc = 0.0
            macro_f1 = 0.0
        rows.append({'threshold': float(t), 'coverage': float(coverage), 'abstention_rate': float(1.0 - coverage), 'accuracy_on_kept': float(acc), 'macro_f1_on_kept': float(macro_f1), 'kept_count': int(keep.sum()), 'abstained_count': int((~keep).sum())})
    return pd.DataFrame(rows)


def metrics_with_optional_proba(y_true, y_pred, y_proba=None, model_name: str = 'model') -> Dict[str, float | str]:
    result = {'model': model_name}
    result.update(classification_metrics(y_true, y_pred))
    if y_proba is not None:
        result['brier_score'] = multiclass_brier_score(y_true, y_proba)
        result['ece_10_bins'] = expected_calibration_error(y_true, y_pred, y_proba, n_bins=10)
        result['mean_confidence'] = float(np.asarray(y_proba).max(axis=1).mean())
    return result
