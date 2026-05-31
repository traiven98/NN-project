import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image

from config import CLASS_NAMES, CLASS_LABELS, NUM_CLASSES, PLOTS_DIR, SEED


def plot_class_distribution(df, save=True):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    counts = df['dx'].value_counts().reindex(CLASS_NAMES)
    colors = sns.color_palette('Set2', NUM_CLASSES)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar([CLASS_LABELS[c] for c in counts.index], counts.values, color=colors)
    axes[0].set_title('Class distribution (absolute)')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Number of images')
    axes[0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 30, str(v), ha='center', fontsize=9)

    axes[1].pie(
        counts.values,
        labels=[CLASS_LABELS[c] for c in counts.index],
        autopct='%1.1f%%',
        colors=colors,
        startangle=140,
    )
    axes[1].set_title('Class distribution (percentage)')

    plt.suptitle('Class Imbalance in HAM10000', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PLOTS_DIR, 'class_distribution.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print('\nClass statistics:')
    for cls, cnt in counts.items():
        print(f'  {CLASS_LABELS[cls]:30s}: {cnt:5d} ({cnt / len(df) * 100:.1f}%)')


def plot_sample_images(df, save=True):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 4, figure=fig)

    for idx, cls in enumerate(CLASS_NAMES):
        samples = df[df['dx'] == cls].sample(1, random_state=SEED)
        img = Image.open(samples.iloc[0]['path']).resize((224, 224))
        ax = fig.add_subplot(gs[idx // 4, idx % 4])
        ax.imshow(img)
        ax.set_title(f'{CLASS_LABELS[cls]}\n({cls})', fontsize=9)
        ax.axis('off')

    fig.suptitle('One sample per class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PLOTS_DIR, 'sample_images.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_patient_demographics(df, save=True):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    df['age'].dropna().hist(ax=axes[0], bins=20, color='steelblue', edgecolor='white')
    axes[0].set_title('Patient age distribution')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Count')

    sex_counts = df['sex'].value_counts()
    axes[1].bar(sex_counts.index, sex_counts.values, color=['#4C72B0', '#DD8452'])
    axes[1].set_title('Sex distribution')
    axes[1].set_ylabel('Count')

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PLOTS_DIR, 'patient_demographics.png'), dpi=150, bbox_inches='tight')
    plt.show()


def run_eda(df):
    print('\n' + '=' * 60)
    print('EDA')
    print('=' * 60)
    print(f'Total images: {len(df)}')
    print(f'Columns: {list(df.columns)}')
    plot_class_distribution(df)
    plot_sample_images(df)
    plot_patient_demographics(df)
