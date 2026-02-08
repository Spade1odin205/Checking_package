import cv2
import os

# ====== CẤU HÌNH ======
VIDEO_PATH = r"D:\Code\Python\Project\checking_package\output_video5_datDung.mp4"  # đường dẫn video
OUTPUT_DIR = r"D:\Code\Python\Project\checking_package\data"     # thư mục lưu ảnh
FRAME_INTERVAL = 3  # Mỗi 1 frame lưu 1 ảnh

# ====== TẠO THƯ MỤC LƯU ẢNH ======
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== MỞ VIDEO ======
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Không thể mở video.")
    exit()

frame_count = 0
saved_count = 0

print("🎥 Bắt đầu cắt ảnh...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Nếu là frame thứ n thì lưu lại
    if frame_count % FRAME_INTERVAL == 0:
        filename = os.path.join(OUTPUT_DIR, f"frame_{saved_count:05d}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"✅ Đã lưu {saved_count} ảnh tại: {OUTPUT_DIR}")
