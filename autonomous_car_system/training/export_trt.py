import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from configs.config import CONFIG

ROAD_CKPT = os.path.join(CONFIG['save_dir_road'], 'best_model.pth')
ROAD_TRT = os.path.join(CONFIG['save_dir_road'], 'best_model_trt.pth')

DRIVING_CKPT = os.path.join(CONFIG['save_dir_driving'], 'best_driving_model.pth')
DRIVING_TRT = os.path.join(CONFIG['save_dir_driving'], 'best_driving_model_trt.pth')


def export_road_segnet():
    from models.road_segnet import RoadSegNet

    if not os.path.exists(ROAD_CKPT):
        print(f"Road checkpoint not found: {ROAD_CKPT}")
        return

    device = torch.device('cuda')
    model = RoadSegNet(
        in_channels=CONFIG['in_channels'],
        out_channels=CONFIG['num_classes']
    ).to(device).half().eval()

    ckpt = torch.load(ROAD_CKPT, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    h, w = CONFIG['inference_size']
    dummy = torch.ones((1, 3, h, w)).cuda().half()

    from torch2trt import torch2trt
    print("Converting RoadSegNet to TensorRT...")
    model_trt = torch2trt(model, [dummy], fp16_mode=True)
    torch.save(model_trt.state_dict(), ROAD_TRT)
    print(f"Saved: {ROAD_TRT}")


def export_driving_net():
    from models.driving_net import DrivingNet

    if not os.path.exists(DRIVING_CKPT):
        print(f"Driving checkpoint not found: {DRIVING_CKPT}")
        return

    device = torch.device('cuda')
    model = DrivingNet(
        input_height=CONFIG['driving_input_height'],
        input_width=CONFIG['driving_input_width'],
        in_channels=CONFIG['in_channels'],
        output_dim=1
    ).to(device).half().eval()

    ckpt = torch.load(DRIVING_CKPT, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    dummy = torch.ones((1, 3, CONFIG['driving_input_height'], CONFIG['driving_input_width'])).cuda().half()

    from torch2trt import torch2trt
    print("Converting DrivingNet to TensorRT...")
    model_trt = torch2trt(model, [dummy], fp16_mode=True)
    torch.save(model_trt.state_dict(), DRIVING_TRT)
    print(f"Saved: {DRIVING_TRT}")


if __name__ == '__main__':
    export_road_segnet()
    export_driving_net()
    print("TensorRT export complete.")
