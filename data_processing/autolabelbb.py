import os
import cv2
import json
from ultralytics import YOLO

# ================= CẤU HÌNH (SỬA TẠI ĐÂY) =================
# Đường dẫn folder ảnh gốc
IMAGE_DIR = r"D:\Code\Python\Project\checking_package\yolo_dataset\images\val"

# Đường dẫn folder lưu file nhãn (JSON)
OUTPUT_LABEL_DIR = r"D:\Code\Python\Project\checking_package\yolo_dataset\images\val"

# Đường dẫn model YOLOv8 đã train
MODEL_PATH = r"D:\Code\Python\Project\checking_package\best.pt"

# Ngưỡng tự tin (Confidence Threshold)
CONF_THRES = 0.1

# QUAN TRỌNG: Định nghĩa tên các object tương ứng với ID khi train
CLASS_MAPPING = {
    0: "Slot", 
    1: "Module-Phu",
    2: "Main_Board",
    3: "J-Link",
    4: "Cap_USB_Den",
    5: "Day_Jumper",
    6: "Cap_USB_Trang",
    7: "Tui_Linh_Kien",
    8: "Cap_Ribbon",
    9: "Box"
}
# ==========================================================

def auto_label_multi_object():
    # 1. Tạo thư mục nếu chưa có
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

    # 2. Nạp model
    print(f"⏳ Đang tải model: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ Lỗi tải model: {e}")
        return

    # 3. Lấy danh sách ảnh
    valid_extensions = (".jpg", ".png", ".jpeg", ".bmp")
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ Đường dẫn ảnh không tồn tại: {IMAGE_DIR}")
        return

    images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_extensions)]
    
    print(f"🔍 Tìm thấy {len(images)} ảnh. Bắt đầu xử lý...")

    count = 0
    for img_name in images:
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        # Đọc ảnh để lấy kích thước
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Không đọc được ảnh: {img_name}")
            continue
        height, width = img.shape[:2]

        # 4. Dự đoán (Predict)
        try:
            results = model(img_path, conf=CONF_THRES, verbose=False)[0]
        except Exception as e:
            print(f"❌ Lỗi khi predict ảnh {img_name}: {e}")
            continue

        # 5. Tạo danh sách shapes cho JSON
        shapes = []

        # --- FIX LỖI QUAN TRỌNG: Kiểm tra xem boxes có tồn tại không ---
        if results.boxes is not None:
            for box in results.boxes:
                # Kiểm tra an toàn từng box
                if box is None or box.cls is None or box.conf is None or box.xyxy is None:
                    continue

                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0])

                # Lấy tên class từ mapping, nếu không có thì lấy số ID
                label_name = CLASS_MAPPING.get(cls_id, str(cls_id))

                shape = {
                    "label": label_name,
                    "points": [
                        [x1, y1],
                        [x2, y2]
                    ],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {},
                    "confidence": round(conf, 2) 
                }
                shapes.append(shape)
        else:
            # Nếu không detect được gì, in ra thông báo nhỏ (tuỳ chọn)
            # print(f"ℹ️ {img_name}: Không tìm thấy đối tượng nào.")
            pass

        # 6. Cấu trúc file JSON chuẩn LabelMe
        data = {
            "version": "5.4.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": img_name,
            "imageData": None, # Để None để giảm dung lượng file
            "imageHeight": height,
            "imageWidth": width
        }

        # 7. Lưu file
        json_filename = os.path.splitext(img_name)[0] + ".json"
        json_path = os.path.join(OUTPUT_LABEL_DIR, json_filename)
        
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(data, jf, ensure_ascii=False, indent=4)

        count += 1
        if count % 10 == 0:
            print(f"✅ Đã xử lý {count}/{len(images)} ảnh...")

    print(f"🎯 Hoàn tất! Đã tạo nhãn cho {count} ảnh tại: {OUTPUT_LABEL_DIR}")

if __name__ == "__main__":
    auto_label_multi_object()