import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from inference.road_detector import RoadDetector
from configs.config import CONFIG


def benchmark(use_trt, label, num_warmup=10, num_iter=50):
    print(f"\n{'='*50}")
    print(f"  Benchmark: {label}")
    print(f"{'='*50}")

    detector = RoadDetector(
        checkpoint_path=os.path.join(CONFIG['save_dir_road'], 'best_model.pth'),
        use_trt=use_trt,
    )

    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    for i in range(num_warmup):
        _ = detector.predict(dummy)

    times = []
    for i in range(num_iter):
        t0 = time.perf_counter()
        _ = detector.predict(dummy)
        times.append((time.perf_counter() - t0) * 1000)

    times = times[5:]
    print(f"  Mean: {np.mean(times):.1f}ms")
    print(f"  Min:  {np.min(times):.1f}ms")
    print(f"  Max:  {np.max(times):.1f}ms")
    print(f"  Std:  {np.std(times):.1f}ms")
    print(f"  FPS:  {1000/np.mean(times):.1f}")

    return np.mean(times)


if __name__ == '__main__':
    print(f"Device: {CONFIG['device']}")

    trt_time = benchmark(use_trt=True, label="TensorRT FP16")

    old_fp16 = CONFIG['use_fp16']
    CONFIG['use_fp16'] = True
    pt_fp16_time = benchmark(use_trt=False, label="PyTorch FP16")

    CONFIG['use_fp16'] = False
    pt_fp32_time = benchmark(use_trt=False, label="PyTorch FP32")

    CONFIG['use_fp16'] = old_fp16

    print(f"\n{'='*50}")
    print(f"  SUMMARY")
    print(f"{'='*50}")
    print(f"  TensorRT FP16:   {trt_time:.1f}ms  ({1000/trt_time:.1f} FPS)")
    print(f"  PyTorch FP16:    {pt_fp16_time:.1f}ms  ({1000/pt_fp16_time:.1f} FPS)")
    print(f"  PyTorch FP32:    {pt_fp32_time:.1f}ms  ({1000/pt_fp32_time:.1f} FPS)")
    best = min(trt_time, pt_fp16_time, pt_fp32_time)
    print(f"  Best: {best:.1f}ms → {1000/best:.1f} FPS")
