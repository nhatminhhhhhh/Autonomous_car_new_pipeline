import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import time

from inference.road_detector import RoadDetector
from inference.mask_steering import MaskSteeringPredictor
from hardware.camera import CameraThread
from configs.config import CONFIG


class AutonomousPipeline:
    def __init__(
        self,
        road_ckpt=None,
        mask_driving_ckpt=None,
        camera_src=0,
        use_trt=False,
    ):
        self.road_detector = RoadDetector(road_ckpt, use_trt=use_trt)
        self.steering_predictor = MaskSteeringPredictor(mask_driving_ckpt)
        self.cam = CameraThread(camera_src)
        self.steering = 0.0
        self.running = False

    def process_frame(self, frame_bgr):
        road_mask = self.road_detector.predict(frame_bgr)
        color_mask = self.road_detector.colorize(road_mask)
        overlay = self.road_detector.overlay(frame_bgr, color_mask, alpha=0.5) 

        steering = self.steering_predictor.predict(road_mask)

        return overlay, road_mask, steering

    def run(self):
        self.running = True
        frame = self.cam.read()

        fps_counter = 0
        fps_time = time.perf_counter()
        display_fps = 0.0

        cv2.namedWindow('Autonomous Car', cv2.WINDOW_NORMAL)

        while self.running:
            new_frame = self.cam.read_nowait()
            if new_frame is not None:
                frame = new_frame

            t0 = time.perf_counter()
            overlay, _, steering = self.process_frame(frame)
            infer_ms = (time.perf_counter() - t0) * 1000

            fps_counter += 1
            now = time.perf_counter()
            if now - fps_time >= 1.0:
                display_fps = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time = now

            cv2.putText(overlay,
                        f"FPS: {display_fps:.0f}  Infer: {infer_ms:.0f}ms",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(overlay, f"Steering: {steering:+.2f}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('Autonomous Car', overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        self.cam.stop()
        cv2.destroyAllWindows()
