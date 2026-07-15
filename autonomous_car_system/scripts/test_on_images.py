import os
import glob
import cv2
import sys

# Đưa đường dẫn gốc vào sys.path để import các module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.road_detector import RoadDetector

def main():
    input_dir = '../data/Road_img'
    output_dir = '../data/test_results'
    os.makedirs(output_dir, exist_ok=True)

    print("Đang nạp mô hình đã huấn luyện...")
    detector = RoadDetector()  # Tự động nạp best_model.pth

    images = glob.glob(os.path.join(input_dir, '*.jpg'))
    if not images:
        print(f"Không tìm thấy ảnh nào trong {input_dir}")
        return

    print(f"Tìm thấy {len(images)} ảnh. Đang tiến hành test...")
    for i, img_path in enumerate(images):
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Dự đoán
        pred_mask = detector.predict(img)
        
        # Tô màu và đè lên ảnh gốc (overlay)
        color_mask = detector.colorize(pred_mask)
        overlay = detector.overlay(img, color_mask, alpha=0.6)
        
        # Ghép ảnh gốc bên trái, ảnh overlay bên phải để dễ so sánh
        combined = cv2.hconcat([img, overlay])
        
        # Lưu file
        base_name = os.path.basename(img_path)
        out_path = os.path.join(output_dir, base_name)
        cv2.imwrite(out_path, combined)
        
        if (i + 1) % 10 == 0:
            print(f" Đã xử lý {i + 1}/{len(images)} ảnh...")

    print(f"\nHoàn tất! Kết quả đã được lưu tại: {os.path.abspath(output_dir)}")

if __name__ == '__main__':
    main()
