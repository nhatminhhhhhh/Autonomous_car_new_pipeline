import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import random

from models.driving_net import DrivingNet
from data.dataset import DrivingDataset, get_driving_transform
from utils.metrics import plot_driving_loss
from utils.model_utils import save_checkpoint
from utils.visualization import visualize_driving
from configs.config import CONFIG


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train():
    set_seed(CONFIG['seed'])
    device = CONFIG['device']
    print(f"Device: {device}")

    data_dir = CONFIG['data_dir']
    if not os.path.exists(data_dir):
        print(f"Data dir '{data_dir}' not found.")
        print("Run 'python data/teleoperation.py' first to collect driving data.")
        return

    transform = get_driving_transform()
    full_dataset = DrivingDataset(
        data_dir, transform=transform,
        img_size=(CONFIG['driving_input_height'], CONFIG['driving_input_width']),
        in_channels=CONFIG['in_channels']
    )

    if len(full_dataset) == 0:
        print("No driving data found!")
        return

    val_size = max(1, int(len(full_dataset) * 0.15))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Driving data: {train_size} train, {val_size} val")

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['driving_batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['driving_batch_size'], shuffle=False, num_workers=2)

    model = DrivingNet(
        input_height=CONFIG['driving_input_height'],
        input_width=CONFIG['driving_input_width'],
        in_channels=CONFIG['in_channels'],
        output_dim=1
    ).to(device)
    print(f"DrivingNet params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['driving_lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    save_dir = CONFIG['save_dir_driving']
    os.makedirs(save_dir, exist_ok=True)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, CONFIG['driving_epochs'] + 1):
        model.train()
        train_loss = 0.0
        for images, steer, _ in tqdm(train_loader, desc=f'Epoch {epoch}/{CONFIG["driving_epochs"]} [Train]'):
            images = images.to(device)
            steer = steer.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, steer)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, steer, _ in tqdm(val_loader, desc=f'Epoch {epoch}/{CONFIG["driving_epochs"]} [Val]'):
                images = images.to(device)
                steer = steer.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, steer)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(save_dir, 'best_driving_model.pth'))
            print(f"  Best model saved (val MSE: {val_loss:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 15:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Training complete. Best val MSE: {best_val_loss:.6f}")
    plot_driving_loss(train_losses, val_losses, save_path=os.path.join(save_dir, 'driving_loss.png'))
    visualize_driving(model, val_loader, device, save_path=os.path.join(save_dir, 'driving_pred.png'))


if __name__ == '__main__':
    train()
