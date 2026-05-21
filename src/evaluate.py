import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score,
)

from config import CLASS_NAMES, CLASS_LABELS, NUM_CLASSES, PLOTS_DIR
from trainer import evaluate


def plot_confusion_matrix(labels, preds, save=True):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
    axes[0].set_title('Confusion Matrix (counts)')
    axes[0].set_xlabel('Predicted class')
    axes[0].set_ylabel('True class')

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1],
                vmin=0, vmax=1)
    axes[1].set_title('Confusion Matrix (row-normalized)')
    axes[1].set_xlabel('Predicted class')
    axes[1].set_ylabel('True class')

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_f1_comparison(base_labels, base_preds, resnet_labels, resnet_preds, save=True):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    base_f1   = f1_score(base_labels,   base_preds,   average=None, labels=list(range(NUM_CLASSES)))
    resnet_f1 = f1_score(resnet_labels, resnet_preds, average=None, labels=list(range(NUM_CLASSES)))

    x, width = np.arange(NUM_CLASSES), 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, base_f1,   width, label='Baseline CNN', color='#4C72B0', alpha=0.85)
    bars2 = ax.bar(x + width/2, resnet_f1, width, label='ResNet18 FT',  color='#DD8452', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_LABELS[c] for c in CLASS_NAMES], rotation=30, ha='right')
    ax.set_ylabel('F1-score')
    ax.set_title('Per-class F1-score: Baseline CNN vs ResNet18 Transfer Learning')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PLOTS_DIR, 'f1_comparison.png'), dpi=150, bbox_inches='tight')
    plt.show()


@torch.no_grad()
def get_probabilities(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        probs = F.softmax(model(imgs), dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.vstack(all_probs), np.array(all_labels)


def full_report(baseline_model, resnet_model, test_loader,
                criterion_baseline, criterion_resnet, device):
    from models import BaselineCNN
    from torchvision import models as tvm

    _, base_acc, base_preds, base_labels = evaluate(baseline_model, test_loader, criterion_baseline, device)
    base_f1 = f1_score(base_labels, base_preds, average='weighted')

    _, resnet_acc, resnet_preds, resnet_labels = evaluate(resnet_model, test_loader, criterion_resnet, device)
    resnet_f1 = f1_score(resnet_labels, resnet_preds, average='weighted')

    resnet_probs, true_arr = get_probabilities(resnet_model, test_loader, device)
    macro_auc = roc_auc_score(true_arr, resnet_probs, multi_class='ovr', average='macro')
    auc_per_class = roc_auc_score(true_arr, resnet_probs, multi_class='ovr', average=None)

    print('=' * 55)
    print(f"{'Model':<20} {'Accuracy':>10} {'Weighted F1':>13}")
    print('-' * 55)
    print(f"{'Baseline CNN':<20} {base_acc*100:>9.2f}% {base_f1:>13.4f}")
    print(f"{'ResNet18 FT':<20} {resnet_acc*100:>9.2f}% {resnet_f1:>13.4f}")
    print('=' * 55)

    label_names = [CLASS_LABELS[c] for c in CLASS_NAMES]
    print('\nResNet18 — Classification Report:')
    print(classification_report(resnet_labels, resnet_preds, target_names=label_names))

    print(f'\nAUC-ROC (macro): {macro_auc:.4f}')
    print('\nAUC-ROC per class:')
    for cls, auc in zip(CLASS_NAMES, auc_per_class):
        print(f'  {CLASS_LABELS[cls]:30s}: {auc:.4f}')

    n_base   = sum(p.numel() for p in BaselineCNN().parameters())
    n_resnet = sum(p.numel() for p in tvm.resnet18().parameters())
    summary = pd.DataFrame({
        'Model':         ['Baseline CNN', 'ResNet18 Transfer Learning'],
        'Test Accuracy': [f'{base_acc*100:.2f}%', f'{resnet_acc*100:.2f}%'],
        'Weighted F1':   [f'{base_f1:.4f}', f'{resnet_f1:.4f}'],
        'Macro AUC-ROC': ['—', f'{macro_auc:.4f}'],
        'Parameters':    [f'{n_base:,}', f'{n_resnet:,}'],
    })
    print('\n' + '=' * 70)
    print(summary.to_string(index=False))
    print('=' * 70)

    plot_confusion_matrix(resnet_labels, resnet_preds)
    plot_f1_comparison(base_labels, base_preds, resnet_labels, resnet_preds)

    return base_preds, base_labels, resnet_preds, resnet_labels
