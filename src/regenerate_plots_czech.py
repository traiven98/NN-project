"""
Regenerates all plots with Czech labels without retraining.
Run from the project root:  python src/regenerate_plots_czech.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score

from config import (
    CLASS_NAMES, NUM_CLASSES, PLOTS_DIR, CHECKPOINTS_DIR, SEED,
    IMAGENET_MEAN, IMAGENET_STD,
)
from dataset import load_dataframe, split_data, build_loaders
from models import BaselineCNN, build_resnet18
from evaluate import get_probabilities

# Czech labels for each class code
CZ_LABELS = {
    'akiec': 'Aktinická keratóza',
    'bcc':   'Bazocelulární karcinom',
    'bkl':   'Benigní keratóza',
    'df':    'Dermatofibrom',
    'mel':   'Melanom',
    'nv':    'Melanocytární névus',
    'vasc':  'Vaskulární léze',
}

os.makedirs(PLOTS_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


# ── helpers ───────────────────────────────────────────────────────────────────

def cz_names():
    return [CZ_LABELS[c] for c in CLASS_NAMES]


@torch.no_grad()
def get_preds(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


# ── 1. EDA: class distribution ───────────────────────────────────────────────

def plot_class_distribution_cz(df):
    counts = df['dx'].value_counts().reindex(CLASS_NAMES)
    colors = sns.color_palette('Set2', NUM_CLASSES)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(cz_names(), counts.values, color=colors)
    axes[0].set_title('Distribuce tříd (absolutní počty)')
    axes[0].set_xlabel('Třída')
    axes[0].set_ylabel('Počet snímků')
    axes[0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 30, str(v), ha='center', fontsize=9)

    axes[1].pie(
        counts.values,
        labels=cz_names(),
        autopct='%1.1f%%',
        colors=colors,
        startangle=140,
    )
    axes[1].set_title('Distribuce tříd (procentuální)')

    plt.suptitle('Nerovnováha tříd v datasetu HAM10000', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'class_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  class_distribution.png')


def plot_sample_images_cz(df):
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 4, figure=fig)

    for idx, cls in enumerate(CLASS_NAMES):
        samples = df[df['dx'] == cls].sample(1, random_state=SEED)
        img = Image.open(samples.iloc[0]['path']).resize((224, 224))
        ax = fig.add_subplot(gs[idx // 4, idx % 4])
        ax.imshow(img)
        ax.set_title(f'{CZ_LABELS[cls]}\n({cls})', fontsize=9)
        ax.axis('off')

    fig.suptitle('Jeden vzorový snímek na třídu', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'sample_images.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  sample_images.png')


def plot_patient_demographics_cz(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    df['age'].dropna().hist(ax=axes[0], bins=20, color='steelblue', edgecolor='white')
    axes[0].set_title('Věkové rozložení pacientů')
    axes[0].set_xlabel('Věk')
    axes[0].set_ylabel('Počet')

    sex_counts = df['sex'].value_counts()
    cz_sex = {'male': 'muž', 'female': 'žena', 'unknown': 'neznámé'}
    labels = [cz_sex.get(s, s) for s in sex_counts.index]
    axes[1].bar(labels, sex_counts.values, color=['#4C72B0', '#DD8452'])
    axes[1].set_title('Rozložení pohlaví')
    axes[1].set_ylabel('Počet')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'patient_demographics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  patient_demographics.png')


# ── 2. Confusion matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix_cz(labels, preds):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=cz_names(), yticklabels=cz_names(), ax=axes[0])
    axes[0].set_title('Matice záměn (počty)')
    axes[0].set_xlabel('Predikovaná třída')
    axes[0].set_ylabel('Skutečná třída')

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=cz_names(), yticklabels=cz_names(), ax=axes[1],
                vmin=0, vmax=1)
    axes[1].set_title('Matice záměn (řádkově normalizovaná)')
    axes[1].set_xlabel('Predikovaná třída')
    axes[1].set_ylabel('Skutečná třída')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  confusion_matrix.png')


# ── 3. F1 comparison ──────────────────────────────────────────────────────────

def plot_f1_comparison_cz(base_labels, base_preds, resnet_labels, resnet_preds):
    base_f1   = f1_score(base_labels,   base_preds,   average=None, labels=list(range(NUM_CLASSES)))
    resnet_f1 = f1_score(resnet_labels, resnet_preds, average=None, labels=list(range(NUM_CLASSES)))

    x, width = np.arange(NUM_CLASSES), 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, base_f1,   width, label='Základní CNN', color='#4C72B0', alpha=0.85)
    bars2 = ax.bar(x + width/2, resnet_f1, width, label='ResNet18 FT',  color='#DD8452', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(cz_names(), rotation=30, ha='right')
    ax.set_ylabel('F1-skóre')
    ax.set_title('F1-skóre podle tříd: Základní CNN vs ResNet18 s přenosovým učením')
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
    plt.savefig(os.path.join(PLOTS_DIR, 'f1_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  f1_comparison.png')


# ── 4. Accuracy / F1 bar chart ───────────────────────────────────────────────

def plot_accuracy_comparison_cz(base_acc, base_f1, resnet_acc, resnet_f1):
    metrics_names = ['Přesnost (test)', 'Vážené F1']
    baseline_vals = [base_acc, base_f1]
    resnet_vals   = [resnet_acc, resnet_f1]

    x, width = np.arange(len(metrics_names)), 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Základní CNN', color='#4C72B0', alpha=0.87)
    bars2 = ax.bar(x + width/2, resnet_vals,   width, label='ResNet18 FT',  color='#DD8452', alpha=0.87)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Skóre', fontsize=12)
    ax.set_title('Celkový výkon: Základní CNN vs ResNet18', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f'{bar.get_height():.3f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='#4C72B0')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f'{bar.get_height():.3f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='#DD8452')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'accuracy_f1_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  accuracy_f1_comparison.png')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print('Načítám data...')
    df = load_dataframe()
    train_df, val_df, test_df = split_data(df)
    _, _, test_loader, val_transform = build_loaders(train_df, val_df, test_df)

    print('Načítám modely z checkpointů...')
    baseline = BaselineCNN().to(device)
    baseline.load_state_dict(
        torch.load(os.path.join(CHECKPOINTS_DIR, 'best_baseline.pth'), map_location=device)
    )

    resnet = build_resnet18(freeze_backbone=False).to(device)
    resnet.load_state_dict(
        torch.load(os.path.join(CHECKPOINTS_DIR, 'best_resnet_finetuned.pth'), map_location=device)
    )

    print('Počítám predikce...')
    base_preds,   base_labels   = get_preds(baseline, test_loader)
    resnet_preds, resnet_labels = get_preds(resnet,   test_loader)

    base_acc   = (base_preds   == base_labels).mean()
    resnet_acc = (resnet_preds == resnet_labels).mean()
    base_f1    = f1_score(base_labels,   base_preds,   average='weighted')
    resnet_f1  = f1_score(resnet_labels, resnet_preds, average='weighted')

    print(f'\nZákladní CNN  — Přesnost: {base_acc*100:.2f}%  F1: {base_f1:.4f}')
    print(f'ResNet18 FT   — Přesnost: {resnet_acc*100:.2f}%  F1: {resnet_f1:.4f}')

    print('\nGeneruji grafy s českými popisky...')
    plot_class_distribution_cz(df)
    plot_sample_images_cz(df)
    plot_patient_demographics_cz(df)
    plot_confusion_matrix_cz(resnet_labels, resnet_preds)
    plot_f1_comparison_cz(base_labels, base_preds, resnet_labels, resnet_preds)
    plot_accuracy_comparison_cz(base_acc, base_f1, resnet_acc, resnet_f1)

    print(f'\nHotovo! Grafy uloženy do složky "{PLOTS_DIR}/"')
    print('Poznámka: grafy průběhu tréninku (history, comparison_curves)')
    print('  nelze přegenerovat bez opětovného trénování — data nejsou uložena.')


if __name__ == '__main__':
    main()
