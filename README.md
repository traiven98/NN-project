# Skin Lesion Classification — HAM10000

Multi-class image classification of dermoscopic skin lesions using PyTorch.  
Compares a custom CNN trained from scratch against ResNet18 with transfer learning.

---

## Problem Description

Skin cancer is one of the most common cancers worldwide. Early and accurate diagnosis
significantly improves patient outcomes. Dermoscopy — a non-invasive imaging technique —
allows detailed visualization of skin lesions, but manual classification by dermatologists
is time-consuming and subject to inter-observer variability.

**Goal:** Build a supervised multi-class classifier that assigns a dermoscopic image to one
of 7 diagnostic categories, ranging from benign moles to malignant melanoma.

**Type of learning:** Supervised multi-class classification. Every image has a confirmed
ground-truth label from histopathology, clinical follow-up, or expert consensus.

**Key challenge:** Severe class imbalance — *Melanocytic nevus* (benign mole) accounts for
~67% of all samples, while rare classes such as Dermatofibroma and Vascular lesion represent
under 2% each. A naive model that always predicts "nevus" reaches ~67% accuracy while being
clinically useless.

---

## Dataset

**HAM10000** ("Human Against Machine with 10000 training images") — a large public collection
of dermoscopic images published in 2018, used as the benchmark in the ISIC 2018 challenge.

### Images

- **Total:** 10,015 JPEG files (450×600 px)
- **Storage:** Two directories — `data/HAM10000_images_part_1` and `data/HAM10000_images_part_2`
- **Input size:** Resized to 224×224 px during loading

### Metadata CSV (`HAM10000_metadata.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `image_id` | string | Unique identifier — matches the image filename |
| `dx` | string | **Ground truth diagnosis** (one of 7 class codes below) |
| `dx_type` | string | Confirmation method: `histo`, `follow_up`, `consensus`, `confocal` |
| `age` | float | Patient age in years (may be `NaN`) |
| `sex` | string | Patient sex: `male`, `female`, or `unknown` |
| `localization` | string | Body location of the lesion (back, face, trunk, …) |

### Classes

| Code | Full name | Type |
|------|-----------|------|
| `nv` | Melanocytic nevus | Benign |
| `mel` | Melanoma | Malignant |
| `bkl` | Benign keratosis | Benign |
| `bcc` | Basal cell carcinoma | Malignant |
| `akiec` | Actinic keratosis | Pre-cancerous |
| `df` | Dermatofibroma | Benign |
| `vasc` | Vascular lesion | Benign |

### Data split

Stratified **70 / 15 / 15** (train / val / test). Stratification preserves class proportions
in every split — critical given the severe imbalance.

---

## Solution Pipeline

| Step | Module | Description |
|------|--------|-------------|
| 1. EDA | notebook | Class distribution, sample images, patient demographics |
| 2. Data preparation | `dataset.py` | Stratified split, WeightedRandomSampler for class balance |
| 3. Transforms | `dataset.py` | Augmentation for training; normalization for all splits |
| 4. Baseline model | `models.py`, `trainer.py` | 4-layer CNN from scratch — performance lower bound |
| 5. Transfer learning | `models.py`, `trainer.py` | ResNet18 fine-tuned in two phases |
| 6. Error analysis | `evaluate.py`, `gradcam.py` | Confusion matrix + Grad-CAM attention maps |
| 7. Comparison | `evaluate.py` | Accuracy, Weighted F1, AUC-ROC across both models |

---

## Data Transforms

Different transforms are applied at training vs. inference time:

| Transform | Train | Val/Test | Rationale |
|-----------|:-----:|:--------:|-----------|
| `Resize(224×224)` | ✅ | ✅ | Required input size for ResNet18 |
| `RandomHorizontalFlip` | ✅ | ❌ | Lesions have no natural orientation |
| `RandomVerticalFlip` | ✅ | ❌ | Same rationale as horizontal flip |
| `RandomRotation(±20°)` | ✅ | ❌ | Rotation-invariant: same lesion at any angle |
| `ColorJitter(b=0.2, c=0.2, s=0.2, h=0.1)` | ✅ | ❌ | Simulates lighting variation across devices/clinics |
| `ToTensor` | ✅ | ✅ | PIL → float tensor [0, 1] |
| `Normalize(ImageNet µ, σ)` | ✅ | ✅ | Matches distribution expected by pretrained weights |

**WeightedRandomSampler:** Each training sample gets a weight inversely proportional to its
class frequency. The sampler draws balanced mini-batches without duplicating images on disk.

---

## Model Architectures

### Baseline CNN (trained from scratch)

Purpose: establish a performance lower bound with no pretrained knowledge.

```
Input:  3 × 224 × 224

Feature extractor:
  Block 1:  Conv2d(3 → 32,  3×3) → BatchNorm → ReLU → MaxPool(2×2)   →  32 × 112 × 112
  Block 2:  Conv2d(32 → 64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)   →  64 ×  56 ×  56
  Block 3:  Conv2d(64 → 128,3×3) → BatchNorm → ReLU → MaxPool(2×2)   → 128 ×  28 ×  28
  Block 4:  Conv2d(128→ 256,3×3) → BatchNorm → ReLU → AdaptiveAvg(4) → 256 ×   4 ×   4

Classifier:
  Flatten → 4096
  Linear(4096 → 512) → ReLU → Dropout(0.4)
  Linear(512 → 7)

Parameters: ~4.7 M
```

| Design choice | Rationale |
|---------------|-----------|
| BatchNorm after every conv | Stabilises training, allows higher LR, mild regulariser |
| Doubling channels per block | Each block captures increasingly abstract features |
| AdaptiveAvgPool in last block | Avoids excessive spatial reduction before classification |
| Dropout(0.4) | Prevents co-adaptation; reduces overfitting |

Training: Adam (lr=1e-3, wd=1e-4) + ReduceLROnPlateau (patience=3, factor=0.5), 10 epochs.

---

### ResNet18 with Transfer Learning

Purpose: leverage ImageNet features to reach higher accuracy with fewer epochs.

```
Backbone — ResNet18 pretrained on ImageNet (frozen in Phase 1, unfrozen in Phase 2):
  Conv(3→64) → BN → ReLU → MaxPool
  Layer1: 2× BasicBlock  ( 64 ch, stride 1)
  Layer2: 2× BasicBlock  (128 ch, stride 2)
  Layer3: 2× BasicBlock  (256 ch, stride 2)
  Layer4: 2× BasicBlock  (512 ch, stride 2)   ← Grad-CAM target layer
  AdaptiveAvgPool(1×1)  →  512-dim vector

Custom head (replaces original FC 512→1000):
  Linear(512 → 256) → ReLU → Dropout(0.3)
  Linear(256 → 7)

Parameters: ~11.2 M total
```

**Two-phase training:**

| Phase | Backbone | Head | Epochs | LR backbone | LR head | Scheduler |
|-------|:--------:|:----:|:------:|:-----------:|:-------:|-----------|
| 1 — Frozen | ❄️ | 🔥 | 5 | — | 1e-3 | ReduceLROnPlateau |
| 2 — Fine-tune | 🔥 | 🔥 | 15 | 3e-5 | 1e-3 | CosineAnnealingLR |

Phase 1 trains only the head to avoid corrupting pretrained features before the head stabilises.
Phase 2 applies differential learning rates — backbone gets 100× lower LR than the head.

---

## Project Structure

```
NN-project/
├── data/                          # Dataset (git-ignored)
│   ├── HAM10000_metadata.csv
│   ├── HAM10000_images_part_1/
│   └── HAM10000_images_part_2/
├── src/
│   ├── config.py                  # All hyperparameters and paths
│   ├── dataset.py                 # Data loading, splits, transforms, DataLoaders
│   ├── models.py                  # BaselineCNN + build_resnet18()
│   ├── trainer.py                 # Training loop, evaluation, plot_history, plot_comparison_curves
│   ├── evaluate.py                # Confusion matrix, F1/accuracy comparison, AUC-ROC, full_report
│   ├── gradcam.py                 # Grad-CAM explainability visualisation
│   └── main.py                    # Entry point — runs the full pipeline
├── checkpoints/                   # Saved model weights (git-ignored)
├── plots/                         # Generated figures (git-ignored)
├── ham10000_classification.ipynb  # Notebook version of the same pipeline
└── README.md
```

---

## Generated Plots

After a full run, `plots/` contains:

| File | Content |
|------|---------|
| `Baseline_CNN_history.png` | Train/val loss & accuracy for Baseline CNN |
| `ResNet18_Transfer_Learning_history.png` | Train/val loss & accuracy for ResNet18 (both phases) |
| `comparison_training_curves.png` | **Both models on the same axes** — loss & accuracy with unfreeze marker |
| `confusion_matrix.png` | Absolute counts + row-normalised confusion matrix (ResNet18) |
| `f1_comparison.png` | Per-class F1-score: Baseline CNN vs ResNet18 |
| `accuracy_f1_comparison.png` | **Overall Accuracy + Weighted F1 bar chart** — both models |
| `gradcam_results.png` | Grad-CAM attention maps: correct vs misclassified samples |

---

## How to Run

```bash
# 1. Install dependencies
pip install torch torchvision scikit-learn matplotlib seaborn pandas grad-cam

# 2. Place the dataset
#    data/HAM10000_metadata.csv
#    data/HAM10000_images_part_1/*.jpg
#    data/HAM10000_images_part_2/*.jpg

# 3. Run the full pipeline
cd src
python main.py
```

Checkpoints are saved to `checkpoints/`, plots to `plots/`.  
GPU is used automatically if available (recommended: CUDA with ≥8 GB VRAM).
