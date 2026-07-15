import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random

from models.road_segnet import RoadSegNet
from data.dataset import RoadSegDataset, get_transforms
from utils.metrics import plot_losses
from utils.model_utils import save_checkpoint
from utils.visualization import visualize_prediction
from configs.config import CONFIG


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train():
    set_seed(CONFIG['seed'])
    device = CONFIG['device']
    print(f"Device: {device}")

    train_dir = os.path.join(CONFIG.get('road_data_dir', '../my_dataset'), 'train')
    val_dir = os.path.join(CONFIG.get('road_data_dir', '../my_dataset'), 'val')

    if not os.path.exists(train_dir):
        print(f"Train dir '{train_dir}' not found. Adjust path or run prepare_dataset.py")
        train_dir = os.path.join('..', 'my_dataset', 'train')
        val_dir = os.path.join('..', 'my_dataset', 'val')

    train_transform = get_transforms(augment=True, target_size=(CONFIG['img_height'], CONFIG['img_width']))
    val_transform = get_transforms(augment=False, target_size=(CONFIG['img_height'], CONFIG['img_width']))

    train_dataset = RoadSegDataset(train_dir, transform=train_transform)
    val_dataset = RoadSegDataset(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'],
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG['batch_size'],
        shuffle=False, num_workers=2, pin_memory=True
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    model = RoadSegNet(
        in_channels=CONFIG['in_channels'],
        out_channels=CONFIG['num_classes']
    ).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # Thêm trọng số (class weights) để phạt nặng lỗi nhận diện sai vạch kẻ đường
    # Vì vạch kẻ (lane, dividing_line) quá nhỏ so với background và road, model có xu hướng bỏ qua chúng
    if CONFIG['num_classes'] == 4:
        weights = torch.tensor([1.0, 2.0, 30.0, 40.0], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    elif CONFIG['num_classes'] == 3:
        weights = torch.tensor([1.0, 2.0, 30.0], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    save_dir = CONFIG['save_dir_road']
    os.makedirs(save_dir, exist_ok=True)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, CONFIG['num_epochs'] + 1):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch}/{CONFIG["num_epochs"]} [Train]'):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f'Epoch {epoch}/{CONFIG["num_epochs"]} [Val]'):
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if epoch % CONFIG['log_interval'] == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"  Best model saved (val: {val_loss:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= CONFIG['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(save_dir, f'checkpoint_epoch{epoch}.pth'))

    print(f"Training complete. Best val loss: {best_val_loss:.6f}")
    plot_losses(train_losses, val_losses, save_path=os.path.join(save_dir, 'loss_plot.png'))
    visualize_prediction(model, val_loader, device, num_samples=min(3, len(val_dataset)),
                         save_path=os.path.join(save_dir, 'prediction_examples.png'))


if __name__ == '__main__':
    train()
