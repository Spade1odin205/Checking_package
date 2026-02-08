import os
import json

# === CẤU HÌNH ===
json_dir = r"D:\Code\Python\Project\Multi_cam\data_new\frame\lan6"   # Folder chứa file .json từ LabelMe
output_dir = r"D:\Code\Python\Project\Multi_cam\data_new\frame\lan6"  # Folder xuất file .txt YOLO
os.makedirs(output_dir, exist_ok=True)

# Nếu bạn có nhiều class, khai báo mapping ở đây:
CLASS_MAP = {
    "Box": 0,
}

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

        # Nếu là polygon hoặc 2 điểm (LabelMe rectangle)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        # Tính tọa độ trung tâm và kích thước (YOLO format)
        x_center = (xmin + xmax) / 2.0 / W
        y_center = (ymin + ymax) / 2.0 / H
        width = (xmax - xmin) / W
        height = (ymax - ymin) / H

        # Giới hạn [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        lines.append(line)

    # Lưu file .txt
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"✅ Đã tạo: {output_path}")


# === DUYỆT TOÀN BỘ FILE ===
for filename in os.listdir(json_dir):
    if not filename.endswith(".json"):
        continue

    name = os.path.splitext(filename)[0]
    json_path = os.path.join(json_dir, filename)
    output_path = os.path.join(output_dir, name + ".txt")

    convert_one(json_path, output_path)
