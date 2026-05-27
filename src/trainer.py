import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from config import CHECKPOINTS_DIR, PLOTS_DIR


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


def train_model(model, train_loader, val_loader, optimizer, scheduler,
                criterion, epochs, device, model_name='model'):
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step(vl_loss)

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINTS_DIR, f'best_{model_name}.pth'))

        marker = ' * best' if vl_acc == best_val_acc else ''
        print(f'Epoch {epoch:3d}/{epochs} | '
              f'Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | '
              f'Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f}{marker}')

    print(f'\nBest val accuracy: {best_val_acc:.4f}')
    return history


def plot_history(history, title, save=True):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'],   label='Val',   linewidth=2)
    axes[0].set_title(f'{title} — Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot([a * 100 for a in history['train_acc']], label='Train', linewidth=2)
    axes[1].plot([a * 100 for a in history['val_acc']],   label='Val',   linewidth=2)
    axes[1].set_title(f'{title} — Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PLOTS_DIR, f'{title.replace(" ", "_")}_history.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_comparison_curves(baseline_history, combined_history, epochs_frozen, save=True):
    """Side-by-side training curves for Baseline CNN vs ResNet18 on the same axes.

    A vertical dashed line marks the epoch when the ResNet18 backbone was unfrozen.
    The two models have different total epoch counts (10 vs 20), so both are plotted
    with their own x-range on shared axes — the x-axis shows epoch index.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    ep_base   = range(1, len(baseline_history['train_loss']) + 1)
    ep_resnet = range(1, len(combined_history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Loss ──────────────────────────────────────────────────────────────────
    axes[0].plot(ep_base,   baseline_history['train_loss'],
                 label='Baseline — Train', color='#4C72B0', linewidth=2)
    axes[0].plot(ep_base,   baseline_history['val_loss'],
                 label='Baseline — Val',   color='#4C72B0', linewidth=2, linestyle='--')
    axes[0].plot(ep_resnet, combined_history['train_loss'],
                 label='ResNet18 — Train', color='#DD8452', linewidth=2)
    axes[0].plot(ep_resnet, combined_history['val_loss'],
                 label='ResNet18 — Val',   color='#DD8452', linewidth=2, linestyle='--')
    axes[0].axvline(x=epochs_frozen, color='gray', linestyle=':', linewidth=1.5,
                    label=f'ResNet18: backbone unfrozen (ep {epochs_frozen})')
    axes[0].set_title('Training Loss — Baseline vs ResNet18', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    axes[1].plot(ep_base,   [a * 100 for a in baseline_history['train_acc']],
                 label='Baseline — Train', color='#4C72B0', linewidth=2)
    axes[1].plot(ep_base,   [a * 100 for a in baseline_history['val_acc']],
                 label='Baseline — Val',   color='#4C72B0', linewidth=2, linestyle='--')
    axes[1].plot(ep_resnet, [a * 100 for a in combined_history['train_acc']],
                 label='ResNet18 — Train', color='#DD8452', linewidth=2)
    axes[1].plot(ep_resnet, [a * 100 for a in combined_history['val_acc']],
                 label='ResNet18 — Val',   color='#DD8452', linewidth=2, linestyle='--')
    axes[1].axvline(x=epochs_frozen, color='gray', linestyle=':', linewidth=1.5,
                    label=f'ResNet18: backbone unfrozen (ep {epochs_frozen})')
    axes[1].set_title('Validation Accuracy — Baseline vs ResNet18', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.suptitle('Model Comparison: Training Dynamics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PLOTS_DIR, 'comparison_training_curves.png'), dpi=150, bbox_inches='tight')
    plt.show()
