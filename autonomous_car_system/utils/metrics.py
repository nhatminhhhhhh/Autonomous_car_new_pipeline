import torch
import numpy as np
import matplotlib.pyplot as plt


def calculate_iou(pred, target, num_classes=3):
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return ious


def mean_iou(pred, target, num_classes=3):
    ious = calculate_iou(pred, target, num_classes)
    valid = [v for v in ious if not np.isnan(v)]
    return np.mean(valid) if valid else 0.0


def plot_losses(train_losses, val_losses, save_path='loss_plot.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss plot saved: {save_path}")


def plot_driving_loss(train_losses, val_losses, save_path='driving_loss.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Driving loss plot saved: {save_path}")
