import os
import json
import glob
from tqdm import tqdm

# ================= CẤU HÌNH (QUAN TRỌNG) =================

# 1. Đường dẫn thư mục chứa file .json (và ảnh)
INPUT_FOLDER = r"D:\Code\Python\Project\checking_package\data\data_processing\turn_3.2"

# 2. DANH SÁCH CLASS (PHẢI GIỐNG HỆT THỨ TỰ CỦA CODE CŨ)
# Logic: Code sẽ tìm tên label trong JSON, so sánh với list này để lấy ra ID (0, 1, 2...)
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

def labelme_to_yolo_obb():
    print(f"--- START CONVERTING JSON (LABELME) -> TXT (YOLO OBB) ---")
    
    # Lấy danh sách file json
    json_files = glob.glob(os.path.join(INPUT_FOLDER, "*.json"))
    
    if not json_files:
        print("❌ Không tìm thấy file .json nào trong thư mục!")
        return

    count = 0
    
    for json_path in tqdm(json_files):
        # 1. Xác định tên file txt output
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        txt_path = os.path.join(INPUT_FOLDER, base_name + ".txt")
        
        # 2. Đọc dữ liệu từ JSON
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️ Lỗi đọc file {json_path}: {e}")
            continue
            
        # Lấy kích thước ảnh từ JSON (LabelMe luôn lưu cái này)
        w = data.get('imageWidth')
        h = data.get('imageHeight')
        
        if w is None or h is None:
            print(f"⚠️ File {json_path} thiếu thông tin width/height. Bỏ qua.")
            continue
            
        yolo_lines = []
        
        # 3. Duyệt qua từng hình vẽ (shape)
        for shape in data.get('shapes', []):
            label = shape.get('label')
            points = shape.get('points') # Dạng [[x1, y1], [x2, y2], ...]
            
            # Kiểm tra xem label có trong danh sách CLASSES không
            if label not in CLASSES:
                print(f"⚠️ Warning: Label '{label}' trong file {base_name} không nằm trong danh sách CLASSES. Bỏ qua.")
                continue
                
            class_id = CLASSES.index(label)
            
            # 4. Chuẩn hóa tọa độ (Normalize 0-1)
            # YOLO format: class_id x1 y1 x2 y2 x3 y3 x4 y4 ...
            line_parts = [str(class_id)]
            
            for px, py in points:
                # Đảm bảo toạ độ không vượt quá khung hình (clip 0-1)
                nx = max(0.0, min(1.0, px / w))
                ny = max(0.0, min(1.0, py / h))
                
                # Giữ độ chính xác 6 số thập phân
                line_parts.append(f"{nx:.6f} {ny:.6f}")
            
            yolo_lines.append(" ".join(line_parts))
            
        # 5. Ghi ra file TXT
        # Lưu ý: Code này sẽ GHI ĐÈ file txt cũ.
        if yolo_lines:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(yolo_lines))
            count += 1
        else:
            # Nếu file json rỗng (không có shape nào), tạo file txt rỗng để YOLO biết là ảnh background
            with open(txt_path, 'w', encoding='utf-8') as f:
                pass 

    print(f"\n✅ DONE! Đã chuyển đổi {count} file JSON sang TXT.")
    print(f"👉 File TXT mới đã được lưu tại: {INPUT_FOLDER}")

if __name__ == "__main__":
    labelme_to_yolo_obb()