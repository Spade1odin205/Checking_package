import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path

# ================= CẤU HÌNH (SỬA Ở ĐÂY) =================
MODEL_PATH = 'best_obb.pt'        # Đường dẫn tới model OBB (.pt) của bạn
IMAGE_DIR = 'images_raw'          # Thư mục chứa ảnh cần label
LABEL_OUTPUT_DIR = 'labels_auto'  # Thư mục lưu file text kết quả
DEBUG_DIR = 'debug_vis'           # Thư mục lưu ảnh vẽ box để kiểm tra (Optional)
CONF_THRESHOLD = 0.4              # Ngưỡng tin cậy (0.0 - 1.0)
SAVE_DEBUG_IMAGES = True          # True: Lưu ảnh vẽ box đè lên để kiểm tra
# ========================================================

def create_dirs():
    """Tạo các thư mục cần thiết nếu chưa tồn tại"""
    os.makedirs(LABEL_OUTPUT_DIR, exist_ok=True)
    if SAVE_DEBUG_IMAGES:
        os.makedirs(DEBUG_DIR, exist_ok=True)

def normalize_coordinates(points, img_w, img_h):
    """
    Chuyển đổi tọa độ pixel sang tọa độ chuẩn hóa (0-1)
    Input: points (numpy array shape 4x2) [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    Output: list phẳng [x1, y1, x2, y2, x3, y3, x4, y4] đã chia cho w, h
    """
    normalized_points = []
    for point in points:
        x, y = point
        # Đảm bảo toạ độ không vượt quá kích thước ảnh
        x = max(0, min(img_w, x))
        y = max(0, min(img_h, y))
        
        # Chuẩn hóa
        normalized_points.append(x / img_w)
        normalized_points.append(y / img_h)
    return normalized_points

def main():
    # 1. Khởi tạo môi trường
    create_dirs()
    print(f"🔄 Đang tải model OBB từ: {MODEL_PATH}...")
    
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ Lỗi không tải được model: {e}")
        return

    # Lấy danh sách ảnh
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [f for f in os.listdir(IMAGE_DIR) if Path(f).suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"⚠️ Không tìm thấy ảnh nào trong thư mục '{IMAGE_DIR}'")
        return

    print(f"🚀 Bắt đầu xử lý {len(image_files)} ảnh...")

    # 2. Vòng lặp xử lý từng ảnh
    for img_name in tqdm(image_files, desc="Auto Labeling"):
        img_path = os.path.join(IMAGE_DIR, img_name)
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(LABEL_OUTPUT_DIR, txt_name)
        
        # Đọc ảnh để lấy kích thước (dùng cho việc chuẩn hóa)
        # Lưu ý: Ultralytics tự đọc ảnh, nhưng ta cần width/height gốc chính xác
        img_cv2 = cv2.imread(img_path)
        if img_cv2 is None:
            continue
        h_img, w_img = img_cv2.shape[:2]

        # --- CHẠY INFERENCE ---
        # task='obb' là bắt buộc cho các model oriented bounding box
        results = model.predict(img_path, conf=CONF_THRESHOLD, verbose=False, task='obb')
        result = results[0]

        label_lines = []
        
        # --- XỬ LÝ KẾT QUẢ ---
        # result.obb chứa thông tin các box nghiêng
        if result.obb is not None:
            # Lấy các thông số: xyxyxyxy (4 điểm), cls (lớp), conf (độ tin cậy)
            obb_boxes = result.obb.xyxyxyxy.cpu().numpy()
            classes = result.obb.cls.cpu().numpy()
            
            for i, box in enumerate(obb_boxes):
                # box là array shape (4, 2) chứa 4 điểm góc của OBB
                cls_id = int(classes[i])
                
                # Chuẩn hóa tọa độ về 0-1
                normalized_flat = normalize_coordinates(box, w_img, h_img)
                
                # Tạo chuỗi định dạng YOLO OBB: class x1 y1 x2 y2 x3 y3 x4 y4
                coords_str = " ".join([f"{x:.6f}" for x in normalized_flat])
                line = f"{cls_id} {coords_str}"
                label_lines.append(line)

        # 3. Lưu file Label (.txt)
        if label_lines:
            with open(txt_path, 'w') as f:
                f.write('\n'.join(label_lines))
        else:
            # Tạo file rỗng nếu không detect được gì (để tool label không báo lỗi)
            open(txt_path, 'w').close()

        # 4. (Tùy chọn) Lưu ảnh Debug để kiểm tra mắt thường
        if SAVE_DEBUG_IMAGES and result.obb is not None:
            debug_path = os.path.join(DEBUG_DIR, img_name)
            
            # Vẽ box lên ảnh gốc
            # result.plot() của ultralytics tự vẽ rất đẹp
            plotted_img = result.plot() 
            cv2.imwrite(debug_path, plotted_img)

    print("\n✅ Hoàn tất!")
    print(f"📁 Labels đã lưu tại: {os.path.abspath(LABEL_OUTPUT_DIR)}")
    if SAVE_DEBUG_IMAGES:
        print(f"🖼️  Ảnh kiểm tra (Debug) tại: {os.path.abspath(DEBUG_DIR)}")
    print("💡 Lưu ý: Hãy kiểm tra file text và ảnh debug để đảm bảo góc xoay chính xác.")

if __name__ == "__main__":
    main()