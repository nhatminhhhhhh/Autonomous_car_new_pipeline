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


class DrivingDataset(Dataset):
    def __init__(self, data_dir, transform=None, img_size=(66, 200), in_channels=3):
        self.data_dir = data_dir
        self.transform = transform
        self.img_h, self.img_w = img_size
        self.in_channels = in_channels

        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        json_files = sorted([
            f for f in os.listdir(self.data_dir) if f.endswith('.json')
        ])
        for jf in json_files:
            with open(os.path.join(self.data_dir, jf), 'r') as f:
                data = json.load(f)
            img_path = os.path.join(self.data_dir, data['image_file'])
            if os.path.exists(img_path):
                samples.append({
                    'image_path': img_path,
                    'steering': data['steering'],
                    'throttle': data.get('throttle', 0.0),
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (self.img_w, self.img_h))

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            image = image.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            image = (image - mean) / std
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        steering = torch.tensor(sample['steering'], dtype=torch.float32)
        throttle = torch.tensor(sample['throttle'], dtype=torch.float32)
        return image, steering, throttle


def get_transforms(augment=True, target_size=(256, 256)):
    transforms_list = []
    if augment:
        transforms_list.extend([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), p=0.3),
        ])
    transforms_list.append(A.Resize(height=target_size[0], width=target_size[1]))
    transforms_list.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    transforms_list.append(ToTensorV2())
    return A.Compose(transforms_list)


def get_driving_transform():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class MaskDrivingDataset(Dataset):
    """
    Dataset cho MaskDrivingNet.

    Load cặp (frame.jpg, frame.json) từ collected_driving/, sau đó:
    1. Chạy RoadSegNet để generate mask [H, W] class index
    2. Normalize mask → [1, H, W] float tensor ∈ [0, 1]
    3. Trả về (mask_tensor, steering_label)

    Args:
        data_dir:     đường dẫn tới thư mục chứa frame_*.jpg + frame_*.json
        road_detector: instance của RoadDetector đã load model (dùng để sinh mask)
        mask_size:    (H, W) — kích thước mask đầu ra, phải bằng RoadSegNet output
        num_classes:  số lớp phân đoạn (để normalize mask)
        augment:      nếu True, áp dụng horizontal flip ngẫu nhiên
    """

    def __init__(
        self,
        data_dir: str,
        road_detector,
        mask_size: tuple = (128, 128),
        num_classes: int = 3,
        augment: bool = True,
    ):
        self.data_dir = data_dir
        self.road_detector = road_detector
        self.mask_h, self.mask_w = mask_size
        self.num_classes = num_classes
        self.augment = augment

        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        json_files = sorted([
            f for f in os.listdir(self.data_dir) if f.endswith('.json')
        ])
        for jf in json_files:
            with open(os.path.join(self.data_dir, jf), 'r') as f:
                data = json.load(f)
            img_path = os.path.join(self.data_dir, data['image_file'])
            if os.path.exists(img_path):
                samples.append({
                    'image_path': img_path,
                    'steering': float(data['steering']),
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Đọc ảnh gốc (BGR)
        frame_bgr = cv2.imread(sample['image_path'])

        # Chạy RoadSegNet để sinh mask [H, W] uint8, class ∈ {0, 1, 2}
        mask = self.road_detector.predict(frame_bgr)  # [H, W]

        # Resize mask về mask_size nếu cần
        if mask.shape[0] != self.mask_h or mask.shape[1] != self.mask_w:
            mask = cv2.resize(mask, (self.mask_w, self.mask_h),
                              interpolation=cv2.INTER_NEAREST)

        steering = sample['steering']

        # Augmentation: horizontal flip với xác suất 0.5
        if self.augment and np.random.random() < 0.5:
            mask = np.fliplr(mask).copy()
            steering = -steering  # đảo chiều steering khi lật ảnh

        # Normalize mask → float [0, 1], shape [1, H, W]
        mask_tensor = torch.from_numpy(
            mask.astype(np.float32)
        ) / float(self.num_classes - 1)
        mask_tensor = mask_tensor.unsqueeze(0)  # [1, H, W]

        steering_tensor = torch.tensor(steering, dtype=torch.float32)
        return mask_tensor, steering_tensor
