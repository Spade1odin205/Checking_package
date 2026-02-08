import os
import json
import cv2
import glob
from tqdm import tqdm

# ================= CẤU HÌNH (QUAN TRỌNG) =================

# 1. Đường dẫn thư mục chứa ảnh và file .txt (Đã auto-label)
INPUT_FOLDER = r"D:\Code\Python\Project\checking_package\data\data_processing\turn_3.2"

# 2. DANH SÁCH CLASS (Phải ĐÚNG THỨ TỰ ID trong file classes.txt hoặc data.yaml lúc train)
# Nếu bạn train với danh sách 8 class như các bước trước, hãy giữ nguyên.
CLASSES = [
    'slot_daytrang',  # ID 0
    'slot_tui',       # ID 1
    'slot_dayden',    # ID 2
    'slot_rgb',       # ID 3
    'slot_dayxam',    # ID 4
    'slot_board',     # ID 5
    'slot_jlink',     # ID 6
    'slot_module'     # ID 7
]
# ========================================================

def yolo_obb_to_labelme():
    print(f"--- START CONVERTING TXT -> JSON (LABELME) ---")
    
    # Lấy danh sách file ảnh
    img_files = glob.glob(os.path.join(INPUT_FOLDER, "*.[jJ][pP][gG]")) + \
                glob.glob(os.path.join(INPUT_FOLDER, "*.[pP][nN][gG]"))
    
    count = 0
    
    for img_path in tqdm(img_files):
        # 1. Xác định tên file txt tương ứng
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(INPUT_FOLDER, base_name + ".txt")
        json_path = os.path.join(INPUT_FOLDER, base_name + ".json")
        
        # Nếu không có file txt thì bỏ qua
        if not os.path.exists(txt_path):
            continue
            
        # 2. Đọc ảnh để lấy kích thước (W, H)
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        
        # 3. Đọc file TXT YOLO
        shapes = []
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9: continue # Format OBB cần ít nhất: id x1 y1 ... x4 y4
            
            class_id = int(parts[0])
            coords = list(map(float, parts[1:])) # Lấy các tọa độ còn lại
            
            # Kiểm tra ID có hợp lệ không
            if class_id < 0 or class_id >= len(CLASSES):
                print(f"⚠️ Warning: Class ID {class_id} không có trong danh sách config!")
                label_name = f"unknown_{class_id}"
            else:
                label_name = CLASSES[class_id]
            
            # 4. De-normalize (Chuyển từ 0-1 sang Pixel)
            # YOLO OBB format: x1 y1 x2 y2 x3 y3 x4 y4 (Normalized)
            points = []
            for i in range(0, len(coords), 2):
                px = coords[i] * w
                py = coords[i+1] * h
                points.append([px, py])
            
            # Tạo object shape cho LabelMe
            shape = {
                "label": label_name,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
            shapes.append(shape)
            
        # 5. Tạo cấu trúc JSON LabelMe
        labelme_data = {
            "version": "5.2.1", # Phiên bản giả lập
            "flags": {},
            "shapes": shapes,
            "imagePath": os.path.basename(img_path), # Chỉ lưu tên file, không lưu full path
            "imageData": None, # Để null cho nhẹ file (LabelMe tự load ảnh)
            "imageHeight": h,
            "imageWidth": w
        }
        
        # 6. Ghi file JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(labelme_data, f, indent=2)
            
        count += 1

    print(f"\n✅ DONE! Đã tạo {count} file JSON.")
    print("👉 Bây giờ bạn có thể mở folder này bằng LabelMe để chỉnh sửa.")

if __name__ == "__main__":
    yolo_obb_to_labelme()