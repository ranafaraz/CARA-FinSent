from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_confusion_matrix_plot(cm_df: pd.DataFrame, path: str | Path, title: str = 'Confusion Matrix') -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm_df.values)
    ax.set_title(title)
    ax.set_xticks(range(len(cm_df.columns)))
    ax.set_xticklabels([c.replace('pred_', '') for c in cm_df.columns], rotation=45, ha='right')
    ax.set_yticks(range(len(cm_df.index)))
    ax.set_yticklabels([i.replace('actual_', '') for i in cm_df.index])
    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            ax.text(j, i, str(cm_df.iloc[i, j]), ha='center', va='center')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_metric_bar_plot(results_df: pd.DataFrame, metric: str, path: str | Path, title: str | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = results_df.sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df['model'].astype(str), df[metric].astype(float))
    ax.set_ylabel(metric)
    ax.set_title(title or f'Model comparison by {metric}')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_reliability_plot(bins_df: pd.DataFrame, path: str | Path, title: str = 'Reliability Diagram') -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.plot(bins_df['mean_confidence'], bins_df['accuracy'], marker='o')
    ax.set_xlabel('Mean confidence')
    ax.set_ylabel('Empirical accuracy')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
