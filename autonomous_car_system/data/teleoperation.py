import cv2
import json
import os
import time
from pynput import keyboard as kb

from hardware.camera import CameraThread

STEERING_STEP = 0.05
THROTTLE_STEP = 0.1


class TeleoperationCollector:
    def __init__(self, save_dir='collected_driving', cam_index=0):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.cam = CameraThread(cam_index)

        self.steering = 0.0
        self.throttle = 0.0
        self.recording = False
        self.frame_idx = 0
        self.current_keys = set()

        self.listener = kb.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()

    def _on_press(self, key):
        try:
            self.current_keys.add(key.char)
        except AttributeError:
            self.current_keys.add(key)

    def _on_release(self, key):
        try:
            self.current_keys.discard(key.char)
        except AttributeError:
            self.current_keys.discard(key)

    def _update_control(self):
        if 'a' in self.current_keys:
            self.steering = max(-1.0, self.steering - STEERING_STEP)
        if 'd' in self.current_keys:
            self.steering = min(1.0, self.steering + STEERING_STEP)
        if 'w' in self.current_keys:
            self.throttle = min(1.0, self.throttle + THROTTLE_STEP)
        if 's' in self.current_keys:
            self.throttle = max(-1.0, self.throttle - THROTTLE_STEP)
        if ' ' in self.current_keys:
            self.throttle = 0.0
            self.steering = 0.0

    def run(self):
        print("=== Teleoperation Data Collection ===")
        print("  WASD  : Steer/Throttle")
        print("  SPACE : Brake (reset to zero)")
        print("  r     : Toggle recording")
        print("  q     : Quit")
        print(f"  Saving to: {self.save_dir}")

        cv2.namedWindow('Teleoperation', cv2.WINDOW_NORMAL)

        while True:
            frame = self.cam.read()

            self._update_control()

            display = frame.copy()
            cv2.putText(display, f"Steering: {self.steering:+.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, f"Throttle: {self.throttle:+.2f}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            status = "REC" if self.recording else "IDLE"
            color = (0, 0, 255) if self.recording else (128, 128, 128)
            cv2.putText(display, f"Status: {status}  Frames: {self.frame_idx}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if self.recording:
                img_name = f"frame_{self.frame_idx:06d}.jpg"
                img_path = os.path.join(self.save_dir, img_name)
                cv2.imwrite(img_path, frame)

                data = {
                    'image_file': img_name,
                    'steering': float(self.steering),
                    'throttle': float(self.throttle),
                    'timestamp': time.time(),
                }
                json_path = os.path.join(self.save_dir, f"frame_{self.frame_idx:06d}.json")
                with open(json_path, 'w') as f:
                    json.dump(data, f, indent=2)

                self.frame_idx += 1

            cv2.imshow('Teleoperation', display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                self.recording = not self.recording
                print(f"{'Started' if self.recording else 'Stopped'} recording")

        self.cam.stop()
        cv2.destroyAllWindows()
        self.listener.stop()
        print(f"Collection ended. {self.frame_idx} frames saved in '{self.save_dir}'")


if __name__ == '__main__':
    import sys
    save_dir = sys.argv[1] if len(sys.argv) > 1 else 'collected_driving'
    collector = TeleoperationCollector(save_dir=save_dir)
    collector.run()
