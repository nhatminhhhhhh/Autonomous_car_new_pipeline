"""
Training script cho MaskDrivingNet.

Pipeline:
  1. Load RoadSegNet (frozen) để generate mask từ collected driving images
  2. Load MaskDrivingDataset — mỗi __getitem__ chạy RoadSegNet → mask → label
  3. Train MaskDrivingNet với MSELoss + HuberLoss

Usage:
  python training/train_mask_driving.py
  python training/train_mask_driving.py --data-dir path/to/collected_driving
  python training/train_mask_driving.py --road-ckpt path/to/road_model.pth
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from models.mask_driving_net import MaskDrivingNet
from data.dataset import MaskDrivingDataset
from inference.road_detector import RoadDetector
from utils.metrics import plot_driving_loss
from utils.model_utils import save_checkpoint
from configs.config import CONFIG


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Train MaskDrivingNet')
    parser.add_argument(
        '--data-dir', type=str,
        default=CONFIG['data_dir'],
        help='Thư mục collected_driving (chứa frame_*.jpg + frame_*.json)',
    )
    parser.add_argument(
        '--road-ckpt', type=str,
        default=os.path.join(CONFIG['save_dir_road'], 'best_model.pth'),
        help='Checkpoint của RoadSegNet (để generate mask)',
    )
    parser.add_argument(
        '--epochs', type=int,
        default=CONFIG['mask_driving_epochs'],
    )
    parser.add_argument(
        '--batch-size', type=int,
        default=CONFIG['mask_driving_batch_size'],
    )
    parser.add_argument(
        '--lr', type=float,
        default=CONFIG['mask_driving_lr'],
    )
    return parser.parse_args()


def train():
    args = parse_args()
    set_seed(CONFIG['seed'])
    device = CONFIG['device']
    print(f"[train_mask_driving] Device: {device}")

    # ─── Kiểm tra data ──────────────────────────────────────────────────────
    if not os.path.exists(args.data_dir):
        print(f"[ERROR] Data dir không tồn tại: {args.data_dir}")
        print("Hãy thu thập dữ liệu trước: python scripts/run.py --mode collect")
        return

    json_count = len([f for f in os.listdir(args.data_dir) if f.endswith('.json')])
    if json_count == 0:
        print(f"[ERROR] Không tìm thấy file JSON trong: {args.data_dir}")
        return
    print(f"[train_mask_driving] Tìm thấy {json_count} samples trong {args.data_dir}")

    # ─── Load RoadSegNet (frozen, chỉ dùng để sinh mask) ───────────────────
    print(f"[train_mask_driving] Loading RoadSegNet từ: {args.road_ckpt}")
    road_detector = RoadDetector(checkpoint_path=args.road_ckpt, use_trt=False)

    # ─── Dataset ─────────────────────────────────────────────────────────────
    mask_size = (
        CONFIG['mask_driving_input_height'],
        CONFIG['mask_driving_input_width'],
    )
    full_dataset = MaskDrivingDataset(
        data_dir=args.data_dir,
        road_detector=road_detector,
        mask_size=mask_size,
        num_classes=CONFIG['num_classes'],
        augment=True,
    )

    if len(full_dataset) == 0:
        print("[ERROR] Dataset rỗng!")
        return

    val_size = max(1, int(len(full_dataset) * CONFIG['val_split']))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG['seed'])
    )
    print(f"[train_mask_driving] Train: {train_size} | Val: {val_size}")

    # val dataset không augment — tắt augment bằng cách wrap lại
    val_dataset.dataset.augment = False  # tắt augmentation khi validate

    num_workers = 0  # mask sinh on-the-fly bằng RoadDetector, không fork-safe
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=(device.type == 'cuda')
    )

    # ─── Model ───────────────────────────────────────────────────────────────
    model = MaskDrivingNet(
        in_channels=CONFIG['mask_driving_in_channels'],
        input_height=mask_size[0],
        input_width=mask_size[1],
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[train_mask_driving] MaskDrivingNet params: {total_params:,}")

    # ─── Loss, Optimizer, Scheduler ──────────────────────────────────────────
    # Dùng Huber loss (smooth L1) — robust hơn MSE với outlier steering
    criterion = nn.HuberLoss(delta=0.1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    save_dir = CONFIG['save_dir_mask_driving']
    os.makedirs(save_dir, exist_ok=True)

    # ─── Training Loop ───────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = CONFIG['mask_driving_early_stop']

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for masks, steers in tqdm(
            train_loader,
            desc=f'Epoch {epoch}/{args.epochs} [Train]',
            leave=False,
        ):
            masks = masks.to(device)
            steers = steers.to(device).unsqueeze(1)  # [B, 1]

            optimizer.zero_grad()
            preds = model(masks)          # [B, 1]
            loss = criterion(preds, steers)
            loss.backward()
            # Gradient clipping để ổn định training
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * masks.size(0)

        train_loss /= train_size
        train_losses.append(train_loss)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for masks, steers in tqdm(
                val_loader,
                desc=f'Epoch {epoch}/{args.epochs} [Val]',
                leave=False,
            ):
                masks = masks.to(device)
                steers = steers.to(device).unsqueeze(1)
                preds = model(masks)
                loss = criterion(preds, steers)
                val_loss += loss.item() * masks.size(0)

        val_loss /= val_size
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        # Log mỗi 5 epoch
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | "
                f"Train Huber: {train_loss:.6f} | "
                f"Val Huber: {val_loss:.6f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

        # Lưu best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            ckpt_path = os.path.join(save_dir, 'best_mask_driving_model.pth')
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': {
                    'in_channels': CONFIG['mask_driving_in_channels'],
                    'input_height': mask_size[0],
                    'input_width': mask_size[1],
                    'num_classes': CONFIG['num_classes'],
                },
            }, ckpt_path)
            print(f"  ✓ Best model saved (val Huber: {val_loss:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"[Early stopping] Không cải thiện sau {early_stop_patience} epochs")
                break

    print(f"\n[train_mask_driving] Hoàn thành! Best val Huber: {best_val_loss:.6f}")
    plot_driving_loss(
        train_losses, val_losses,
        save_path=os.path.join(save_dir, 'mask_driving_loss.png')
    )
    print(f"[train_mask_driving] Loss plot saved: {save_dir}/mask_driving_loss.png")
    print(f"[train_mask_driving] Model saved:     {save_dir}/best_mask_driving_model.pth")


if __name__ == '__main__':
    train()
