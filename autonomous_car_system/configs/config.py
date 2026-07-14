import torch
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

    'seed': 42,

    'num_classes': 3,
    'in_channels': 3,
    'img_height': 128,
    'img_width': 128,

    'batch_size': 16,
    'num_epochs': 200,
    'learning_rate': 5e-3,
    'weight_decay': 5e-5,
    'early_stopping_patience': 15,
    'log_interval': 5,
    'val_split': 0.15,

    'save_dir_road': os.path.join(BASE_DIR, 'checkpoints', 'road_seg'),

    'inference_size': (128, 128),
    'use_fp16': True,

    'steering_bins': 15,
    'driving_lr': 1e-4,
    'driving_epochs': 100,
    'driving_batch_size': 32,
    'save_dir_driving': os.path.join(BASE_DIR, 'checkpoints', 'driving'),
    'driving_input_height': 66,
    'driving_input_width': 200,

    'camera_index': 0,
    'camera_width': 640,
    'camera_height': 480,
    'camera_fps': 60,

    'pwm_steering_center': 90,
    'pwm_steering_range': 45,
    'pwm_throttle_min': 60,
    'pwm_throttle_max': 180,
    'pwm_frequency': 50,

    'data_dir': os.path.join(BASE_DIR, 'data', 'collected_driving'),

    # MaskDrivingNet — điều khiển từ mask phân đoạn
    'mask_driving_input_height': 128,   # bằng inference_size của RoadSegNet
    'mask_driving_input_width': 128,
    'mask_driving_in_channels': 1,      # mask 1-channel (class index)
    'mask_driving_lr': 1e-3,
    'mask_driving_epochs': 80,
    'mask_driving_batch_size': 32,
    'mask_driving_early_stop': 15,
    'save_dir_mask_driving': os.path.join(BASE_DIR, 'checkpoints', 'mask_driving'),
}
