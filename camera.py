import cv2
import threading
import time

class CameraStream:
    def __init__(self, url, id):
        self.url = url  # Lưu URL để reconnect
        self.cap = cv2.VideoCapture(url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.id = id
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.resize(frame, (640, 480))
                    with self.lock:
                        self.frame = frame
                else:
                    # Mất tín hiệu -> Mở lại bằng self.url
                    time.sleep(0.5)
                    self.cap.open(self.url) 
            else:
                # Camera chưa mở -> Mở lại
                time.sleep(0.5)
                self.cap.open(self.url)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.cap.release()