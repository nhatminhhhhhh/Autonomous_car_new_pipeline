import torch
import numpy as np
import matplotlib.pyplot as plt


def visualize_prediction(model, dataloader, device, num_samples=1, save_path='prediction.png'):
    model.eval()
    with torch.no_grad():
        images, masks = next(iter(dataloader))
        images = images[:num_samples].to(device)
        masks = masks[:num_samples].cpu().numpy()
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images_np = images.cpu().numpy()
    images_np = (images_np * std[None, :, None, None] + mean[None, :, None, None]).transpose(0, 2, 3, 1)
    images_np = np.clip(images_np, 0, 1)

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    for i in range(num_samples):
        axes[i, 0].imshow(images_np[i])
        axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(masks[i], cmap='jet', vmin=0, vmax=2)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        axes[i, 2].imshow(preds[i], cmap='jet', vmin=0, vmax=2)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Prediction visualization saved: {save_path}")


def visualize_driving(model, dataloader, device, save_path='driving_pred.png'):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for images, steer, _ in dataloader:
            images = images.to(device)
            steer = steer.to(device)
            output = model(images)
            preds.extend(output.cpu().numpy())
            targets.extend(steer.cpu().numpy())

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(targets)), targets, s=10, label='Ground Truth', alpha=0.6)
    plt.scatter(range(len(preds)), preds, s=10, label='Prediction', alpha=0.6)
    plt.xlabel('Sample')
    plt.ylabel('Steering Angle')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    errors = np.array(targets) - np.array(preds).flatten()
    plt.hist(errors, bins=30, alpha=0.7)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Driving prediction plot saved: {save_path}")
