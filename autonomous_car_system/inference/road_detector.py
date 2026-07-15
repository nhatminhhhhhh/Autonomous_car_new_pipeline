import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import cv2

import time

from models.road_segnet import RoadSegNet
from configs.config import CONFIG

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

COLOR_LUT = np.array([
    [0, 0, 0],        # background
    [0, 200, 0],      # road (green)
    [200, 200, 0],    # lane_marking (yellow)
], dtype=np.uint8)


class RoadDetector:
    def __init__(self, checkpoint_path=None, use_trt=False):
        self.device = CONFIG['device']
        self.infer_h, self.infer_w = CONFIG['inference_size']
        self.use_fp16 = CONFIG['use_fp16'] and self.device.type == 'cuda'

        if checkpoint_path is None:
            checkpoint_path = os.path.join(CONFIG['save_dir_road'], 'best_model.pth')

        if use_trt:
            trt_path = checkpoint_path.replace('.pth', '_trt.pth')
            self._load_trt(trt_path)
        else:
            self._load_pytorch(checkpoint_path)

        self._mean_gpu = torch.tensor(_MEAN, device=self.device, dtype=self._dtype).view(1, 3, 1, 1)
        self._std_gpu = torch.tensor(_STD, device=self.device, dtype=self._dtype).view(1, 3, 1, 1)

    def _load_pytorch(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"RoadSegNet checkpoint not found: {path}")

        model = RoadSegNet(in_channels=CONFIG['in_channels'], out_channels=CONFIG['num_classes']).to(self.device)
        ckpt = torch.load(path, map_location=self.device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        if self.use_fp16 and self.device.type == 'cuda':
            model = model.half()
            self._dtype = torch.float16
        else:
            self._dtype = torch.float32

        self.model = model
        print(f"[RoadDetector] PyTorch model loaded ({'FP16' if self.use_fp16 else 'FP32'})")

    def _load_trt(self, path):
        if not os.path.exists(path):
            print(f"[RoadDetector] TRT not found at {path}, falling back to PyTorch")
            self._load_pytorch(path.replace('_trt.pth', '.pth'))
            return

        from torch2trt import TRTModule
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(path, map_location=self.device))
        self.model = model_trt
        self._dtype = torch.float16
        print(f"[RoadDetector] TensorRT model loaded")

    @torch.no_grad()
    def predict(self, frame_bgr, timing=False):
        h, w = frame_bgr.shape[:2]

        if timing: t0 = time.perf_counter()
        small = cv2.resize(frame_bgr, (self.infer_w, self.infer_h))
        if timing: t_resize = time.perf_counter()

        rgb = small[:, :, ::-1].copy()
        t = torch.from_numpy(rgb).to(self.device, non_blocking=True)
        t = t.permute(2, 0, 1).unsqueeze(0)
        t = t.to(dtype=self._dtype).div_(255.0)
        t = (t - self._mean_gpu) / self._std_gpu
        if timing: t_preproc = time.perf_counter()

        out = self.model(t)
        if timing: t_infer = time.perf_counter()

        pred = torch.argmax(out, dim=1).squeeze(0).byte().cpu().numpy()
        if timing: t_post = time.perf_counter()

        if pred.shape[0] != h or pred.shape[1] != w:
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

        if timing:
            print(f"  [Timing] resize={t_resize-t0:.1f}ms  preproc={t_preproc-t_resize:.1f}ms  "
                  f"infer={t_infer-t_preproc:.1f}ms  post={t_post-t_infer:.1f}ms  "
                  f"total={(time.perf_counter()-t0)*1000:.1f}ms")

        return pred

    def colorize(self, pred_mask):
        return COLOR_LUT[pred_mask]

    def overlay(self, frame, color_mask, alpha=0.5):
        return cv2.addWeighted(frame, 1.0 - alpha, color_mask, alpha, 0)
