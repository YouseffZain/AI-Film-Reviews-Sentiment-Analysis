"""
evaluate.py
-----------
Evaluation utilities for all model types.

Generates:
  - Accuracy, Precision, Recall, F1-score (classification_report)
  - Confusion matrices (heatmaps)
  - ROC-AUC curves
  - Cross-model comparison table and bar chart
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)


def print_report(y_true, y_pred, model_name='Model'):
    """Print accuracy and full classification report."""
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{model_name} — Test Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))
    return acc


def plot_confusion_matrices(y_test, preds_dict, cmap='Blues', title='Confusion Matrices'):
    """
    Plot confusion matrices for multiple models side by side.

    Args:
        y_test:     True binary labels
        preds_dict: {model_name: prediction_array}
        cmap:       Matplotlib colormap
        title:      Figure title
    """
    n = len(preds_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, preds) in zip(axes, preds_dict.items()):
        cm = confusion_matrix(y_test, preds)
        acc = accuracy_score(y_test, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                    xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
        ax.set_title(f'{name}\n({acc:.2%})', fontweight='bold')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_roc_curves(y_test, probs_dict, title='ROC Curves'):
    """
    Plot ROC-AUC curves for multiple models on the same axes.

    Args:
        y_test:     True binary labels
        probs_dict: {model_name: probability_array (positive class)}
        title:      Plot title
    """
    plt.figure(figsize=(8, 6))
    for name, probs in probs_dict.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random (AUC=0.5)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_training_history(histories_dict, metric='accuracy'):
    """
    Plot training vs validation curves for DL models.

    Args:
        histories_dict: {arch: Keras History object}
        metric:         'accuracy' or 'loss'
    """
    n = len(histories_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (arch, history) in zip(axes, histories_dict.items()):
        ax.plot(history.history[metric], label='Train', linewidth=2)
        ax.plot(history.history[f'val_{metric}'], label='Validation', linewidth=2)
        ax.set_title(f'{arch}', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle(f'Training History — {metric.capitalize()}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_model_comparison(all_results):
    """
    Create a horizontal bar chart comparing all models.

    Args:
        all_results: list of dicts with keys 'Model', 'Type', 'Test Accuracy'
                     Type must be one of: 'ML (TF-IDF)', 'DL (RNN+GloVe)', 'Transformer'
    """
    import pandas as pd
    from matplotlib.patches import Patch

    df = pd.DataFrame(all_results).sort_values('Test Accuracy', ascending=True)
    colors = {
        'ML (TF-IDF)':    '#3498db',
        'DL (RNN+GloVe)': '#e67e22',
        'Transformer':    '#9b59b6'
    }
    bar_colors = [colors.get(t, '#95a5a6') for t in df['Type']]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(df['Model'], df['Test Accuracy'], color=bar_colors, edgecolor='white', height=0.6)

    for bar, val in zip(bars, df['Test Accuracy']):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.2%}', va='center', fontweight='bold', fontsize=11)

    ax.set_xlim(0, 1.08)
    ax.set_xlabel('Test Accuracy', fontsize=12)
    ax.set_title('Full Model Comparison — IMDB Sentiment Analysis', fontsize=14, fontweight='bold')
    ax.axvline(x=0.9, color='red', linestyle='--', alpha=0.5)

    legend_elements = [Patch(facecolor=c, label=l) for l, c in colors.items()]
    legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', alpha=0.5, label='90% Target'))
    ax.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()
    plt.show()
