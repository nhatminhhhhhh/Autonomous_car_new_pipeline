"""
MaskSteeringPredictor — inference wrapper cho MaskDrivingNet.

Nhận mask phân đoạn (đầu ra của RoadDetector) và trả về góc lái.

Usage:
    predictor = MaskSteeringPredictor(checkpoint_path)
    mask = road_detector.predict(frame_bgr)
    steering = predictor.predict(mask)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import cv2

from models.mask_driving_net import MaskDrivingNet
from configs.config import CONFIG


class MaskSteeringPredictor:
    def __init__(self, checkpoint_path: str = None):
        self.device = CONFIG['device']
        self.mask_h = CONFIG['mask_driving_input_height']
        self.mask_w = CONFIG['mask_driving_input_width']
        self.num_classes = CONFIG['num_classes']

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

        saved_cfg = ckpt.get('config', {})
        in_channels = saved_cfg.get('in_channels', CONFIG['mask_driving_in_channels'])
        h = saved_cfg.get('input_height', self.mask_h)
        w = saved_cfg.get('input_width', self.mask_w)

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

        epoch = ckpt.get('epoch', '?')
        val_loss = ckpt.get('val_loss', float('nan'))
        print(
            f"[MaskSteeringPredictor] Model loaded — "
            f"epoch={epoch}, val_huber={val_loss:.6f}, "
            f"input=({h}×{w}), device={self.device}"
        )

    @torch.no_grad()
    def predict(self, mask: np.ndarray) -> float:
        if self.model is None:
            return 0.0

        if mask.shape[0] != self.mask_h or mask.shape[1] != self.mask_w:
            mask = cv2.resize(
                mask, (self.mask_w, self.mask_h),
                interpolation=cv2.INTER_NEAREST
            )

        mask_f = mask.astype(np.float32) / float(self.num_classes - 1)

        t = torch.from_numpy(mask_f).to(self.device)
        t = t.unsqueeze(0).unsqueeze(0)

        out = self.model(t)
        steering = out.squeeze().item()

        return float(np.clip(steering, -1.0, 1.0))
