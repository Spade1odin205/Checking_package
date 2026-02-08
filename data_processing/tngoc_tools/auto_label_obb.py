import cv2, os, glob, random, json
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ===== CONFIG =====
IMG_DIR = r"D:\Code\Python\Project\Multi_cam\frames_all"
OUT_LABEL_DIR = r"D:\Code\Python\Project\Multi_cam\frames_all"
MODEL_PATH = r"D:\Code\Python\Project\Multi_cam\model1\weights\best.pt"

IMG_SIZE = 640
CONF = 0.25
IOU = 0.5

SPECIAL_CLASS = "Den_nho"   # ⚠️ class được giữ lại 3 box

# ===== PREPARE =====
os.makedirs(OUT_LABEL_DIR, exist_ok=True)
random.seed(0); np.random.seed(0)

# Load model
model = YOLO(MODEL_PATH)
names = model.model.names
print(f"[INFO] Model loaded: {MODEL_PATH}")
print(f"[INFO] Classes: {names}")

# List images
img_paths = sorted([
    p for ext in ("*.jpg", "*.png", "*.jpeg", "*.bmp")
    for p in glob.glob(str(Path(IMG_DIR) / ext))
])

# ===== PROCESS =====
for p in img_paths:
    im = cv2.imread(p)
    if im is None:
        print(f"[WARN] Cannot read {p}")
        continue

    h, w = im.shape[:2]
    stem = Path(p).stem
    json_path = Path(OUT_LABEL_DIR) / f"{stem}.json"

    results = model.predict(
        source=im,
        imgsz=IMG_SIZE,
        conf=CONF,
        iou=IOU,
        verbose=False,
        save=False
    )

    # --- Gom box theo class ---
    class_groups = {c: [] for c in names.values()}

    for r in results:

        # --- OBB detection ---
        if hasattr(r, "obb") and r.obb is not None and len(r.obb) > 0:
            for b in r.obb:
                cls = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                pts = b.xyxyxyxy[0].cpu().numpy().astype(float).tolist()

                entry = {
                    "label": names[cls],
                    "points": pts,
                    "shape_type": "polygon",
                    "group_id": None,
                    "flags": {},
                    "confidence": conf
                }
                class_groups[names[cls]].append(entry)

        # --- fallback to normal boxes ---
        elif hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                cls = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

                entry = {
                    "label": names[cls],
                    "points": pts,
                    "shape_type": "rectangle",
                    "group_id": None,
                    "flags": {},
                    "confidence": conf
                }
                class_groups[names[cls]].append(entry)

    # ===== Giữ lại box theo quy tắc =====
    final_shapes = []

    for cls_name, box_list in class_groups.items():
        if len(box_list) == 0:
            continue

        # Sort theo confidence giảm dần
        box_list = sorted(box_list, key=lambda x: x["confidence"], reverse=True)

        if cls_name == SPECIAL_CLASS:
            # Giữ tối đa 3 box
            final_shapes.extend(box_list[:3])
        else:
            # Giữ 1 box tốt nhất
            final_shapes.append(box_list[0])

    num_boxes = len(final_shapes)

    # ===== SAVE JSON =====
    if num_boxes > 0:
        data = {
            "version": "5.4.1",
            "flags": {},
            "shapes": final_shapes,
            "imagePath": Path(p).name,
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w
        }

        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(data, jf, ensure_ascii=False, indent=4)

        print(f"[OK] {stem}.jpg - {num_boxes} boxes → {json_path}")
    else:
        print(f"[INFO] No detections: {stem}.jpg")

print("\n=== ✅ Done Auto Labeling ===")
