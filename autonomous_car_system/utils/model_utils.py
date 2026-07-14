import torch
import os


def save_checkpoint(state, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(filename, model, optimizer=None):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    return model, optimizer, epoch, loss
