---
name: project-ham10000
description: University ML project — HAM10000 skin lesion classification, 7-class CNN with transfer learning
metadata:
  type: project
---

HAM10000 skin lesion classification — university neural networks course project.

**Why:** Academic project demonstrating supervised learning pipeline on a real medical imaging dataset with class imbalance.

**How to apply:** When user asks about this project, context is a CNN classification task in PyTorch on Google Colab. The notebook is at `C:\Users\nyaha\UNI\NN\ham10000_classification.ipynb`. 

Key design choices made:
- WeightedRandomSampler to handle class imbalance (nv = 67% of data)
- Two-phase training: frozen backbone (5 epochs, LR=1e-3) → fine-tune all (15 epochs, LR=3e-5)
- ResNet18 primary model (literature baseline ~80% accuracy)
- Grad-CAM via pytorch-grad-cam library, targeting `layer4[-1]`
- Metrics: weighted F1, per-class AUC-ROC, confusion matrix (absolute + normalized)
- 70/15/15 stratified train/val/test split
