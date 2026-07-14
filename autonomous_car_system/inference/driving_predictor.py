import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import cv2

from models.driving_net import DrivingNet
from configs.config import CONFIG

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DrivingPredictor:
    def __init__(self, checkpoint_path=None, use_trt=False):
        self.device = CONFIG['device']
        self.input_h = CONFIG['driving_input_height']
        self.input_w = CONFIG['driving_input_width']
        self.use_fp16 = CONFIG['use_fp16'] and self.device.type == 'cuda'

        if checkpoint_path is None:
            checkpoint_path = os.path.join(CONFIG['save_dir_driving'], 'best_driving_model.pth')

        if use_trt:
            trt_path = checkpoint_path.replace('.pth', '_trt.pth')
            self._load_trt(trt_path)
        else:
            self._load_pytorch(checkpoint_path)

        self._mean_gpu = torch.tensor(_MEAN, device=self.device, dtype=self._dtype).view(1, 3, 1, 1)
        self._std_gpu = torch.tensor(_STD, device=self.device, dtype=self._dtype).view(1, 3, 1, 1)

    def _load_pytorch(self, path):
        if not os.path.exists(path):
            print(f"[DrivingPredictor] Checkpoint not found: {path}")
            print("[DrivingPredictor] Running without driving model (manual control fallback)")
            self.model = None
            return

        model = DrivingNet(
            input_height=self.input_h, input_width=self.input_w,
            in_channels=CONFIG['in_channels'], output_dim=1
        ).to(self.device)

        ckpt = torch.load(path, map_location=self.device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        if self.use_fp16 and self.device.type == 'cuda':
            model = model.half()
            self._dtype = torch.float16
        else:
            self._dtype = torch.float32

        self.model = model
        print(f"[DrivingPredictor] PyTorch model loaded ({'FP16' if self.use_fp16 else 'FP32'})")

    def _load_trt(self, path):
        if not os.path.exists(path):
            print(f"[DrivingPredictor] TRT not found, fallback to PyTorch")
            self._load_pytorch(path.replace('_trt.pth', '.pth'))
            return

        from torch2trt import TRTModule
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(path, map_location=self.device))
        self.model = model_trt
        self._dtype = torch.float16
        print(f"[DrivingPredictor] TensorRT model loaded")

    @torch.no_grad()
    def predict(self, frame_bgr):
        if self.model is None:
            return 0.0

        small = cv2.resize(frame_bgr, (self.input_w, self.input_h))

        rgb = small[:, :, ::-1].copy()
        t = torch.from_numpy(rgb).to(self.device, non_blocking=True)
        t = t.permute(2, 0, 1).unsqueeze(0)
        t = t.to(dtype=self._dtype).div_(255.0)
        t = (t - self._mean_gpu) / self._std_gpu

        out = self.model(t)
        steering = out.squeeze().item()

        return np.clip(steering, -1.0, 1.0)
