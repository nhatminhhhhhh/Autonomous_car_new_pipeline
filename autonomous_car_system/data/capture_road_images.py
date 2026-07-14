import cv2
import os
import sys
import time

# Thêm đường dẫn gốc để import các module hardware, configs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hardware.camera import CameraThread

def main():
    save_dir = "Road_img"
    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Đã tạo thư mục: {save_dir}")

    # Khởi tạo camera (sử dụng CameraThread của hệ thống)
    cam = CameraThread(src=0)
    print("=== Tool Chụp Ảnh Sân (RoadSegNet) ===")
    print("Nhấn 's' để chụp và lưu ảnh")
    print("Nhấn 'q' để thoát")
    print(f"Ảnh sẽ được lưu tại thư mục: {save_dir}")

    cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
    img_count = 0

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            # Hiển thị text hướng dẫn trên màn hình
            display = frame.copy()
            cv2.putText(display, f"Press 's' to save, 'q' to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Saved: {img_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Camera Feed', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Nhấn 's' để lưu ảnh
            if key == ord('s'):
                timestamp = int(time.time() * 1000)
                img_name = f"road_{timestamp}.jpg"
                img_path = os.path.join(save_dir, img_name)
                cv2.imwrite(img_path, frame)
                img_count += 1
                print(f"[{img_count}] Đã lưu ảnh: {img_path}")
                
                # Hiệu ứng chớp màn hình khi chụp
                flash = 255 - display
                cv2.imshow('Camera Feed', flash)
                cv2.waitKey(50)

            # Nhấn 'q' để thoát
            elif key == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print(f"Đã thoát. Tổng cộng lưu được {img_count} ảnh trong thư mục '{save_dir}'.")

if __name__ == '__main__':
    main()
