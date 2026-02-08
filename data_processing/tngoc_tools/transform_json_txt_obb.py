import os
import json
import cv2
import numpy as np

# === CẤU HÌNH ===
json_dir = r"D:\Code\Python\Project\Multi_cam\frames_all"
output_dir = r"D:\Code\Python\Project\Multi_cam\frames_all_labels_obb4pt"
os.makedirs(output_dir, exist_ok=True)

CLASS_MAP = {
    "Board": 0,
    "Den_to":1,
    "Den_nho":2,
    "tui": 3,
    "sac": 4,
    "rgb": 5,
    "day_trang": 6,
    "day_xam": 7,
}

def polygon_to_obb_points(points, W, H):
    """Chuyển polygon thành OBB gồm 4 điểm (normalized)."""

    pts = np.array(points, dtype=np.float32)

    # Tính OBB (min area rectangle)
    rect = cv2.minAreaRect(pts)  
    box = cv2.boxPoints(rect)  # 4 điểm
    box = np.array(box, dtype=np.float32)

    # Normalize 4 điểm theo YOLO
    box[:, 0] /= W
    box[:, 1] /= H

    # Clamp vào [0,1]
    box = np.clip(box, 0, 1)

    return box.flatten().tolist()   # trả về 8 giá trị


def convert_one(json_path, output_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    H = data.get("imageHeight", None)
    W = data.get("imageWidth", None)
    if H is None or W is None:
        print(f"⚠️ Không có kích thước ảnh trong: {json_path}")
        return

    lines = []
    for shape in data.get("shapes", []):
        label = shape.get("label", "object")
        class_id = CLASS_MAP.get(label, 0)
        points = shape.get("points", [])

        if len(points) < 3:
            print(f"⚠️ Polygon không hợp lệ: {json_path}")
            continue

        # Lấy 4 điểm OBB
        coords = polygon_to_obb_points(points, W, H)  # list 8 values

        # Format YOLO OBB: id + 8 normalized coords
        line = f"{class_id} " + " ".join(f"{v:.6f}" for v in coords)
        lines.append(line)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ Đã tạo OBB 4 điểm: {output_path}")


# === DUYỆT FILE ===
for filename in os.listdir(json_dir):
    if filename.endswith(".json"):
        json_path = os.path.join(json_dir, filename)
        name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, name + ".txt")
        convert_one(json_path, output_path)
