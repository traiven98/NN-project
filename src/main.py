import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    SEED, NUM_CLASSES, EPOCHS_FROZEN, EPOCHS_FINETUNE,
    LR_FROZEN, LR_FINETUNE, CHECKPOINTS_DIR,
)
from dataset import load_dataframe, split_data, build_loaders
from models import BaselineCNN, build_resnet18
from trainer import train_model, plot_history
from evaluate import full_report
from gradcam import visualize_gradcam


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def main():
    set_seed()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    # ── Data ──────────────────────────────────────────────────────────────────
    df = load_dataframe()
    train_df, val_df, test_df = split_data(df)
    print(f'Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}')

    train_loader, val_loader, test_loader, val_transform = build_loaders(train_df, val_df, test_df)

    # ── Stage 4: Baseline CNN ─────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('Training Baseline CNN')
    print('=' * 60)

    baseline_model     = BaselineCNN().to(device)
    criterion_baseline = nn.CrossEntropyLoss()
    optimizer_baseline = optim.Adam(baseline_model.parameters(), lr=LR_FROZEN, weight_decay=1e-4)
    scheduler_baseline = optim.lr_scheduler.ReduceLROnPlateau(optimizer_baseline, patience=3, factor=0.5)

    baseline_history = train_model(
        baseline_model, train_loader, val_loader,
        optimizer_baseline, scheduler_baseline, criterion_baseline,
        epochs=10, device=device, model_name='baseline',
    )
    plot_history(baseline_history, 'Baseline CNN')

    # ── Stage 5: Transfer Learning ────────────────────────────────────────────
    resnet_model     = build_resnet18(freeze_backbone=True).to(device)
    criterion_resnet = nn.CrossEntropyLoss()

    print('\n' + '=' * 60)
    print('Phase 1: frozen backbone')
    print('=' * 60)
    optimizer_frozen = optim.Adam(
        filter(lambda p: p.requires_grad, resnet_model.parameters()),
        lr=LR_FROZEN, weight_decay=1e-4,
    )
    scheduler_frozen = optim.lr_scheduler.ReduceLROnPlateau(optimizer_frozen, patience=2, factor=0.5)
    history_frozen = train_model(
        resnet_model, train_loader, val_loader,
        optimizer_frozen, scheduler_frozen, criterion_resnet,
        epochs=EPOCHS_FROZEN, device=device, model_name='resnet_frozen',
    )

    print('\n' + '=' * 60)
    print('Phase 2: full fine-tuning')
    print('=' * 60)
    for param in resnet_model.parameters():
        param.requires_grad = True

    optimizer_ft = optim.AdamW([
        {'params': list(resnet_model.parameters())[:-4], 'lr': LR_FINETUNE},
        {'params': list(resnet_model.parameters())[-4:], 'lr': LR_FROZEN},
    ], weight_decay=1e-4)
    scheduler_ft = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=EPOCHS_FINETUNE)
    history_ft = train_model(
        resnet_model, train_loader, val_loader,
        optimizer_ft, scheduler_ft, criterion_resnet,
        epochs=EPOCHS_FINETUNE, device=device, model_name='resnet_finetuned',
    )

    combined_history = {k: history_frozen[k] + history_ft[k] for k in history_frozen}
    plot_history(combined_history, 'ResNet18 Transfer Learning')

    # ── Stage 6 & 7: Evaluation + Grad-CAM ───────────────────────────────────
    baseline_model.load_state_dict(
        torch.load(os.path.join(CHECKPOINTS_DIR, 'best_baseline.pth'), map_location=device)
    )
    resnet_model.load_state_dict(
        torch.load(os.path.join(CHECKPOINTS_DIR, 'best_resnet_finetuned.pth'), map_location=device)
    )

    base_preds, base_labels, resnet_preds, resnet_labels = full_report(
        baseline_model, resnet_model, test_loader,
        criterion_baseline, criterion_resnet, device,
    )

    visualize_gradcam(resnet_model, test_df, resnet_labels, resnet_preds, val_transform, device)


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(__file__))
    main()
