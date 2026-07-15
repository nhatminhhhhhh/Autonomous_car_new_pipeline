import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json

class RoadSegDataset(Dataset):
    def __init__(self, data_dir, transform=None, label_suffix='.png'):
        self.image_dir = os.path.join(data_dir, 'images')
        self.label_dir = os.path.join(data_dir, 'labels')
        self.transform = transform

        self.images = sorted(os.listdir(self.image_dir))
        self.labels = sorted(os.listdir(self.label_dir))

        assert len(self.images) == len(self.labels), \
            f"Image count ({len(self.images)}) != Label count ({len(self.labels)})"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']

        label = label.long()
        return image, label


def get_transforms(augment=True, target_size=(256, 256)):
    transforms_list = []
    if augment:
        transforms_list.extend([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), p=0.3),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.15),
            A.RandomGamma(p=0.15),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.15),
            A.RandomShadow(p=0.15),
            A.ISONoise(p=0.15),
        ])
    transforms_list.append(A.Resize(height=target_size[0], width=target_size[1]))
    transforms_list.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    transforms_list.append(ToTensorV2())
    return A.Compose(transforms_list)


class MaskDrivingDataset(Dataset):
    """
    Dataset cho DrivingNet (mask-based).

    Load cặp (mask.png, steering) từ thư mục collected_driving/.
    Mask là PNG 1-channel đã được lưu từ RoadSegNet trong lúc collect data.

    Args:
        data_dir:     đường dẫn thư mục chứa mask_*.png + frame_*.json
        mask_size:    (H, W) — resize mask nếu cần
        num_classes:  số lớp phân đoạn (để normalize)
        augment:      horizontal flip ngẫu nhiên
    """

    def __init__(
        self,
        data_dir: str,
        mask_size: tuple = (128, 128),
        num_classes: int = 3,
        augment: bool = True,
    ):
        self.data_dir = data_dir
        self.mask_h, self.mask_w = mask_size
        self.num_classes = num_classes
        self.augment = augment

        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        json_files = sorted(
            f for f in os.listdir(self.data_dir) if f.endswith('.json')
        )
        for jf in json_files:
            with open(os.path.join(self.data_dir, jf), 'r') as f:
                data = json.load(f)
            mask_file = data.get('mask_file', jf.replace('.json', '_mask.png'))
            mask_path = os.path.join(self.data_dir, mask_file)
            if os.path.exists(mask_path):
                samples.append({
                    'mask_path': mask_path,
                    'steering': float(data['steering']),
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        mask = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
        if mask.shape[0] != self.mask_h or mask.shape[1] != self.mask_w:
            mask = cv2.resize(mask, (self.mask_w, self.mask_h),
                              interpolation=cv2.INTER_NEAREST)

        steering = sample['steering']

        if self.augment and np.random.random() < 0.5:
            mask = np.fliplr(mask).copy()
            steering = -steering

        mask_tensor = torch.from_numpy(
            mask.astype(np.float32)
        ) / float(self.num_classes - 1)
        mask_tensor = mask_tensor.unsqueeze(0)

        steering_tensor = torch.tensor(steering, dtype=torch.float32)
        return mask_tensor, steering_tensor
