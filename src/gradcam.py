import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from config import IMG_SIZE, CLASS_NAMES, PLOTS_DIR, SEED


def visualize_gradcam(model, test_df, test_labels, test_preds, val_transform, device, n_show=8):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    model.eval()

    target_layer = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layer)

    def _get_overlay(img_path, pred_label):
        img_pil = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        img_np  = np.array(img_pil) / 255.0
        img_tensor = val_transform(img_pil).unsqueeze(0).to(device)
        grayscale  = cam(input_tensor=img_tensor, targets=[ClassifierOutputTarget(pred_label)])[0]
        overlay    = show_cam_on_image(img_np.astype(np.float32), grayscale, use_rgb=True)
        return img_np, overlay

    samples = []
    for cls_idx in range(len(CLASS_NAMES)):
        import pandas as pd
        cls_mask   = (np.array(test_labels) == cls_idx)
        candidates = test_df[cls_mask].copy()
        candidates['pred'] = np.array(test_preds)[cls_mask]
        correct   = candidates[candidates['pred'] == cls_idx]
        incorrect = candidates[candidates['pred'] != cls_idx]
        if len(correct) > 0:
            row = correct.sample(1, random_state=SEED).iloc[0]
            samples.append((row['path'], cls_idx, int(row['pred']), True))
        if len(incorrect) > 0:
            row = incorrect.sample(1, random_state=SEED).iloc[0]
            samples.append((row['path'], cls_idx, int(row['pred']), False))

    samples = samples[:n_show]
    fig, axes = plt.subplots(len(samples), 2, figsize=(8, len(samples) * 3))
    if len(samples) == 1:
        axes = [axes]

    for i, (path, true_lbl, pred_lbl, is_correct) in enumerate(samples):
        orig, overlay = _get_overlay(path, pred_lbl)
        status = 'Correct' if is_correct else 'Wrong'
        color  = 'green' if is_correct else 'red'

        axes[i][0].imshow(orig)
        axes[i][0].set_title(f'{status} | True: {CLASS_NAMES[true_lbl]}', color=color, fontsize=9)
        axes[i][0].axis('off')

        axes[i][1].imshow(overlay)
        axes[i][1].set_title(f'Pred: {CLASS_NAMES[pred_lbl]}  (Grad-CAM)', fontsize=9)
        axes[i][1].axis('off')

    plt.suptitle('Grad-CAM: what the network focuses on', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'gradcam_results.png'), dpi=150, bbox_inches='tight')
    plt.show()
