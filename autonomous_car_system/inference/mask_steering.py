"""
MaskSteeringPredictor — inference wrapper cho MaskDrivingNet.

Nhận mask phân đoạn (đầu ra của RoadDetector) và trả về góc lái.

Usage:
    predictor = MaskSteeringPredictor(checkpoint_path)
    mask = road_detector.predict(frame_bgr)   # [H, W] uint8
    steering = predictor.predict(mask)         # float ∈ [-1.0, 1.0]
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from models.mask_driving_net import MaskDrivingNet
from configs.config import CONFIG


class MaskSteeringPredictor:
    """
    Inference wrapper cho MaskDrivingNet.

    Args:
        checkpoint_path: đường dẫn tới file .pth đã train
        use_trt:         (chưa hỗ trợ, reserved)
    """

    def __init__(self, checkpoint_path: str = None, use_trt: bool = False):
        self.device = CONFIG['device']
        self.mask_h = CONFIG['mask_driving_input_height']
        self.mask_w = CONFIG['mask_driving_input_width']
        self.num_classes = CONFIG['num_classes']
        self.in_channels = CONFIG['mask_driving_in_channels']

        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                CONFIG['save_dir_mask_driving'], 'best_mask_driving_model.pth'
            )

        self._load_model(checkpoint_path)

    def _load_model(self, path: str):
        if not os.path.exists(path):
            print(f"[MaskSteeringPredictor] Checkpoint không tìm thấy: {path}")
            print("[MaskSteeringPredictor] Chạy với model=None (steering = 0.0)")
            self.model = None
            return

        ckpt = torch.load(path, map_location=self.device)

        # Đọc config từ checkpoint nếu có (linh hoạt)
        saved_cfg = ckpt.get('config', {})
        in_channels = saved_cfg.get('in_channels', self.in_channels)
        h = saved_cfg.get('input_height', self.mask_h)
        w = saved_cfg.get('input_width', self.mask_w)
        num_classes = saved_cfg.get('num_classes', self.num_classes)

        model = MaskDrivingNet(
            in_channels=in_channels,
            input_height=h,
            input_width=w,
        ).to(self.device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        self.model = model
        self.mask_h = h
        self.mask_w = w
        self.num_classes = num_classes

        epoch = ckpt.get('epoch', '?')
        val_loss = ckpt.get('val_loss', float('nan'))
        print(
            f"[MaskSteeringPredictor] Model loaded — "
            f"epoch={epoch}, val_huber={val_loss:.6f}, "
            f"input=({h}×{w}), device={self.device}"
        )

    @torch.no_grad()
    def predict(self, mask: np.ndarray) -> float:
        """
        Dự đoán góc lái từ mask phân đoạn.

        Args:
            mask: numpy array [H, W] dtype uint8, class index ∈ {0, 1, 2}
                  Đây là output trực tiếp từ RoadDetector.predict()

        Returns:
            steering: float ∈ [-1.0, 1.0]
                      < 0 = rẽ trái, > 0 = rẽ phải, 0 = thẳng
        """
        if self.model is None:
            return 0.0

        import cv2

        # Resize mask về kích thước model cần nếu khác nhau
        if mask.shape[0] != self.mask_h or mask.shape[1] != self.mask_w:
            mask = cv2.resize(
                mask, (self.mask_w, self.mask_h),
                interpolation=cv2.INTER_NEAREST
            )

        # Normalize mask → float [0, 1]
        mask_f = mask.astype(np.float32) / float(self.num_classes - 1)

        # Chuyển sang tensor [1, 1, H, W]
        t = torch.from_numpy(mask_f).to(self.device)
        t = t.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        out = self.model(t)              # [1, 1]
        steering = out.squeeze().item()

        return float(np.clip(steering, -1.0, 1.0))
