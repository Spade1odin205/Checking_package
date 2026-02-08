from ultralytics import YOLO

def train_slot_obb():
    # Load model OBB pre-trained
    model = YOLO("yolov8s-obb.pt") 

    results = model.train(
        data="D:\\Code\\Python\\Project\\checking_package\\data\\data\\slot\\data.yaml",
        epochs=100,
        imgsz=640,
        device=0,
        batch=4,
        workers=1,
        name="train_slot_obb_s",
        
        # --- AUGMENTATION CHO SLOT ---
        # Slot là vật tĩnh, hình dạng cố định trên xốp, nên hạn chế biến dạng
        degrees=10.0,    # Cho phép xoay nhẹ (để model học được slot khi thùng bị lệch)
        translate=0.1,   # Dịch chuyển ảnh
        scale=0.2,       # Scale nhẹ
        
        # TẮT các biến dạng hình học (Rất quan trọng với bài toán khớp khít)
        shear=0.0,       # TẮT: Không làm méo slot
        perspective=0.0, # TẮT: Không làm méo 3D
        
        # Ánh sáng (để phân biệt màu nâu của thùng và màu trắng của xốp)
        hsv_h=0.015,
        hsv_s=0.5,       # Tăng khả năng nhận diện độ đậm nhạt
        hsv_v=0.4,       # Tăng khả năng chịu đựng bóng đổ
        
        mosaic=1.0,      # Bật
        mixup=0.0,       # Tắt
        copy_paste=0.0,  # Tắt
    )

if __name__ == '__main__':
    train_slot_obb()