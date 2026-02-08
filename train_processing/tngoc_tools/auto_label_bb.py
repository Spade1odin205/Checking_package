import os
import cv2
import json
from ultralytics import YOLO

# ====== CẤU HÌNH ======
IMAGE_DIR = r"D:\Code\Python\Project\Multi_cam\data_new\frame\lan6"        # Folder chứa ảnh
OUTPUT_LABEL_DIR = r"D:\Code\Python\Project\Multi_cam\data_new\frame\lan6" # Folder lưu nhãn auto
MODEL_PATH = r"D:\Code\Python\Project\Multi_cam\model_new\box_detector4\weights\best.pt"
CONF_THRES = 0.75  # Ngưỡng confidence

# ====== TẠO THƯ MỤC ======
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# ====== NẠP MODEL ======
model = YOLO(MODEL_PATH)

# ====== DANH SÁCH ẢNH ======
images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

print(f"🔍 Đang auto-label {len(images)} ảnh...")

for img_name in images:
    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)
    height, width = img.shape[:2]

    results = model(img_path, conf=CONF_THRES, verbose=False)[0]

    # ====== TẠO DỮ LIỆU JSON ======
    shapes = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(float, box.xyxy[0])

        shape = {
            "label": str(cls_id),
            "points": [
                [x1, y1],
                [x2, y2]
            ],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {},
            "confidence": conf
        }
        shapes.append(shape)

    data = {
        "version": "5.4.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": img_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }

    # ====== GHI FILE JSON ======
    json_path = os.path.join(OUTPUT_LABEL_DIR, os.path.splitext(img_name)[0] + ".json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(data, jf, ensure_ascii=False, indent=4)

    print(f"✅ {img_name} → {json_path}")

print("🎯 Hoàn tất auto-labeling JSON!")
