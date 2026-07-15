import os
import sys
import time
import cv2
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hardware.camera import CameraThread
from hardware.esp32_controller import ESP32Controller
from configs.config import CONFIG

def parse_args():
    parser = argparse.ArgumentParser(description='Local Keyboard Teleoperation')
    parser.add_argument('--port', type=str, default=None, help='Cổng Serial của ESP32 (mặc định tự tìm)')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate của ESP32')
    parser.add_argument('--camera', type=int, default=CONFIG['camera_index'], help='Index của Camera')
    return parser.parse_args()

def main():
    args = parse_args()

    print("[KeyboardTeleop] Đang khởi động camera...")
    cam = CameraThread(args.camera)
    
    print("[KeyboardTeleop] Đang kết nối với ESP32 qua cáp Serial (UART)...")
    esp32 = ESP32Controller(port=args.port, baud=args.baud)

    print("\n" + "="*35)
    print("   ĐIỀU KHIỂN BẰNG BÀN PHÍM LOKAL")
    print("="*35)
    print("  W / S : Tiến / Lùi (Throttle)")
    print("  A / D : Trái / Phải (Steering)")
    print("  Space : Dừng khẩn cấp (Phanh)")
    print("  Q     : Thoát chương trình")
    print("="*35 + "\n")

    steering = 0.0
    throttle = 0.0
    step = 0.05  # Tốc độ tăng/giảm sau mỗi lần bấm phím

    cv2.namedWindow('Keyboard Teleop', cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = cam.read()
            
            # Hiển thị thông số lên màn hình camera
            overlay = frame.copy()
            cv2.putText(overlay, f"Steering: {steering:+.2f} | Throttle: {throttle:+.2f}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(overlay, f"ESP32: {'CONNECTED' if esp32.connected() else 'SIMULATION MODE'}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('Keyboard Teleop', overlay)

            # Lắng nghe phím bấm
            key = cv2.waitKey(1) & 0xFF
            
            changed = False
            if key == ord('w'):
                throttle = min(1.0, throttle + step)
                changed = True
            elif key == ord('s'):
                throttle = max(-1.0, throttle - step)
                changed = True
            elif key == ord('a'):
                steering = max(-1.0, steering - step)
                changed = True
            elif key == ord('d'):
                steering = min(1.0, steering + step)
                changed = True
            elif key == ord(' '):  # Nút Space
                steering = 0.0
                throttle = 0.0
                changed = True
            elif key == ord('q'):
                print("Đang thoát chương trình...")
                break

            # Truyền command xuống ESP32
            if changed:
                esp32.send_command(steering, throttle)

    finally:
        print("[KeyboardTeleop] Đóng kết nối...")
        esp32.stop()
        esp32.close()
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
