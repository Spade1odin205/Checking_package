import os
import json
import cv2

# === CẤU HÌNH ===
image_dir = r"D:\Code\Python\Project\Multi_cam\data_new\dataset_split_2\val\images"   # folder chứa ảnh
label_dir = r"D:\Code\Python\Project\Multi_cam\data_new\dataset_split_2\val\labels"   # folder chứa label .txt (YOLO format)
output_dir = r"D:\Code\Python\Project\Multi_cam\data_new\dataset_split_2\val\images"   # folder chứa file .json xuất ra

os.makedirs(output_dir, exist_ok=True)

# === (TÙY CHỌN) DANH SÁCH CLASS ===
# Nếu có file classes.txt thì đọc vào, không thì tạo tạm
classes_path = os.path.join(label_dir, "classes.txt")
if os.path.exists(classes_path):
    with open(classes_path, "r", encoding="utf-8") as f:
        classes = [c.strip() for c in f.readlines()]
else:
    classes = [f"class_{i}" for i in range(100)]  # fallback

def convert_one(image_path, label_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Không đọc được ảnh: {image_path}")
        return
    H, W = img.shape[:2]

    shapes = []
    if not os.path.exists(label_path):
        print(f"⚠️ Không có file label cho {os.path.basename(image_path)}")
        return

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id = int(parts[0])
            x_center, y_center, w, h = map(float, parts[1:])

            # Chuyển từ normalized → pixel
            x_center *= W
            y_center *= H
            w *= W
            h *= H

            # Tính tọa độ 2 góc
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2

            # Định nghĩa khung cho LabelMe
            shape = {
                "label": classes[cls_id] if cls_id < len(classes) else f"class_{cls_id}",
                "points": [[x1, y1], [x2, y2]],  # 2 góc (trái-trên, phải-dưới)
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
            shapes.append(shape)

    # Cấu trúc JSON đúng chuẩn LabelMe
    data = {
        "version": "5.3.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": H,
        "imageWidth": W
    }

    # Ghi file JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ {os.path.basename(output_path)}")

# === DUYỆT TẤT CẢ ẢNH ===
for filename in os.listdir(image_dir):
    name, ext = os.path.splitext(filename)
    if ext.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    image_path = os.path.join(image_dir, filename)
    label_path = os.path.join(label_dir, name + ".txt")
    output_path = os.path.join(output_dir, name + ".json")

    convert_one(image_path, label_path, output_path)

print("\n=== ✅ Hoàn tất chuyển đổi YOLO → LabelMe (bbox chuẩn) ===")
