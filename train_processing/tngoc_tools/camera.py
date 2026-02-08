import cv2

# RTSP URL của camera
rtsp_url = "rtsp://admin:JRGMMV@192.168.1.151:554/ch1/main"

# Mở luồng RTSP
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("❌ Không thể kết nối tới camera RTSP")
    exit()

print("✅ Kết nối RTSP thành công")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Không nhận được frame từ camera")
        break

    # Hiển thị hình ảnh
    cv2.imshow("RTSP Camera", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
