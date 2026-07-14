import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import cv2
import time

from inference.road_detector import RoadDetector
from inference.driving_predictor import DrivingPredictor
from inference.mask_steering import MaskSteeringPredictor
from hardware.camera import CameraThread
from configs.config import CONFIG


def parse_args():
    parser = argparse.ArgumentParser(description='Autonomous Car System')
    parser.add_argument(
        '--mode', type=str, default='test',
        choices=['test', 'collect', 'autonomous'],
        help='Chế độ hoạt động: test=kiểm tra segmentation, collect=thu dữ liệu, autonomous=tự lái'
    )
    parser.add_argument(
        '--steering-method', type=str, default='mask_model',
        choices=['mask_model', 'drivingnet'],
        help=(
            'Phương pháp tính góc lái khi autonomous:\n'
            '  mask_model  — MaskDrivingNet (CNN từ mask, mặc định)\n'
            '  drivingnet  — DrivingNet gốc (end-to-end từ ảnh RGB)'
        )
    )
    parser.add_argument('--use-trt', action='store_true',
                        help='Dùng TensorRT cho inference')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index')
    parser.add_argument(
        '--save-dir', type=str,
        default=CONFIG['data_dir'],
        help='Thư mục lưu dữ liệu thu thập (chế độ collect)'
    )
    parser.add_argument(
        '--road-ckpt', type=str,
        default=os.path.join(CONFIG['save_dir_road'], 'best_model.pth'),
        help='Checkpoint RoadSegNet'
    )
    parser.add_argument(
        '--mask-driving-ckpt', type=str,
        default=os.path.join(CONFIG['save_dir_mask_driving'], 'best_mask_driving_model.pth'),
        help='Checkpoint MaskDrivingNet (dùng khi --steering-method mask_model)'
    )
    parser.add_argument(
        '--driving-ckpt', type=str,
        default=os.path.join(CONFIG['save_dir_driving'], 'best_driving_model.pth'),
        help='Checkpoint DrivingNet gốc (dùng khi --steering-method drivingnet)'
    )
    return parser.parse_args()


# ─── Chế độ TEST ──────────────────────────────────────────────────────────────
def mode_test(road_detector, cam):
    print("[Mode] Road Detection Test")
    print("  Keys: q=quit, o/l=alpha, m=mode cycle")

    alpha = 0.5
    mode = 0
    modes = ['Overlay', 'Mask only', 'Camera only']
    fps_counter = 0
    fps_time = time.perf_counter()
    display_fps = 0.0

    frame = cam.read()

    while True:
        frame = cam.read()

        t0 = time.perf_counter()
        pred = road_detector.predict(frame)
        color_mask = road_detector.colorize(pred)
        infer_ms = (time.perf_counter() - t0) * 1000

        if mode == 0:
            display = road_detector.overlay(frame, color_mask, alpha)
        elif mode == 1:
            display = color_mask
        else:
            display = frame

        fps_counter += 1
        now = time.perf_counter()
        if now - fps_time >= 1.0:
            display_fps = fps_counter / (now - fps_time)
            fps_counter = 0
            fps_time = now

        cv2.putText(display,
                    f"FPS: {display_fps:.0f}  Infer: {infer_ms:.0f}ms  Alpha:{alpha:.2f}  [{modes[mode]}]",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.imshow('Road Detection Test', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('o'):
            alpha = min(1.0, alpha + 0.05)
        elif key == ord('l'):
            alpha = max(0.0, alpha - 0.05)
        elif key == ord('m'):
            mode = (mode + 1) % len(modes)

    cv2.destroyAllWindows()


# ─── Chế độ COLLECT ───────────────────────────────────────────────────────────
def mode_collect(save_dir, cam_index):
    from data.teleoperation import TeleoperationCollector
    collector = TeleoperationCollector(save_dir=save_dir, cam_index=cam_index)
    collector.run()


# ─── Chế độ AUTONOMOUS ────────────────────────────────────────────────────────
def mode_autonomous(road_detector, steering_predictor, cam, steering_method):
    print(f"[Mode] Autonomous Driving — steering_method={steering_method}")
    print("  Keys: q=quit, e=estop")

    method_label = {
        'mask_model': 'MaskCNN',
        'drivingnet': 'DrivingNet',
    }[steering_method]

    cv2.namedWindow('Autonomous Driving', cv2.WINDOW_NORMAL)

    while True:
        frame = cam.read()

        t0 = time.perf_counter()
        road_mask = road_detector.predict(frame)
        color_mask = road_detector.colorize(road_mask)
        overlay = road_detector.overlay(frame, color_mask, 0.4)

        # Tính góc lái theo phương pháp đã chọn
        if steering_method == 'mask_model':
            steering = steering_predictor.predict(road_mask)
        else:
            steering = steering_predictor.predict(frame)

        infer_ms = (time.perf_counter() - t0) * 1000

        cv2.putText(
            overlay,
            f"[{method_label}] Steering: {steering:+.2f} | Infer: {infer_ms:.0f}ms",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
        )

        # Hiển thị cảnh báo nếu model chưa load
        no_model = (
            (steering_method == 'mask_model' and steering_predictor.model is None) or
            (steering_method == 'drivingnet' and steering_predictor.model is None)
        )
        if no_model:
            cv2.putText(overlay,
                        "WARNING: No steering model loaded — steering=0.0",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('Autonomous Driving', overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            print("[ESTOP] Emergency stop triggered")

    cv2.destroyAllWindows()


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print(f"Loading RoadSegNet từ: {args.road_ckpt}")
    road_detector = RoadDetector(args.road_ckpt, use_trt=args.use_trt)

    cam = CameraThread(args.camera)

    if args.mode == 'test':
        mode_test(road_detector, cam)

    elif args.mode == 'collect':
        cam.stop()
        mode_collect(args.save_dir, args.camera)

    elif args.mode == 'autonomous':
        # Load steering model theo phương pháp được chọn
        if args.steering_method == 'mask_model':
            print(f"Loading MaskDrivingNet từ: {args.mask_driving_ckpt}")
            steering_predictor = MaskSteeringPredictor(args.mask_driving_ckpt)
        else:
            print(f"Loading DrivingNet từ: {args.driving_ckpt}")
            steering_predictor = DrivingPredictor(args.driving_ckpt, use_trt=args.use_trt)

        mode_autonomous(road_detector, steering_predictor, cam, args.steering_method)

    cam.stop()


if __name__ == '__main__':
    main()
