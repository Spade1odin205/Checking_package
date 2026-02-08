# QC SYSTEM – Kiểm tra lắp ráp bằng 4 camera + YOLOv8 (Detect + OBB Slot)

Dự án này chạy realtime từ 4 camera RTSP, dùng 2 model YOLOv8:

- **Model Detect (bbox)**: nhận diện linh kiện + class kích hoạt quy trình (mặc định: `Box`).
- **Model Slot (OBB)**: nhận diện các “slot” (hình chữ nhật xoay) trên khay/xốp.

Hệ thống đối chiếu **vật nằm trong slot nào** bằng cách kiểm tra **tâm object** có nằm trong **đa giác OBB của slot** (Shapely `Polygon.contains(Point)`), sau đó kết luận **PASS/FAIL** theo từng stage.

## 1) Luồng hoạt động

1. Mỗi camera đọc frame (resize 640x480) qua thread (`camera.py`).
2. Cứ mỗi `SKIP_FRAMES` frame, chạy inference cả 2 model trên batch ảnh.
3. Khi phát hiện `TRIGGER_CLASS` (mặc định `Box`):
	 - Chỉ “chốt” kết quả khi box nằm trong **vùng an toàn** (không sát biên ảnh) theo `EDGE_MARGIN`.
	 - Tính số lượng linh kiện đúng slot theo `CAM_RULES`.
4. `ProcessController` debounce nhiễu và “latch snapshot” kết quả khi box ổn định.
5. Quy trình kết thúc khi **Stage 4 đã PASS** và **không còn thấy box**.

## 2) Yêu cầu

- Python 3.9+ (khuyến nghị 3.10/3.11)
- Windows/Linux đều chạy được (repo đang cấu hình theo Windows)
- GPU NVIDIA (tùy chọn, giúp realtime mượt hơn)

Thư viện chính:

- `ultralytics` (YOLOv8)
- `opencv-python`
- `torch`
- `shapely`
- `numpy`

## 3) Cài đặt

### 3.0 Tải data, model & demo

- Data: https://drive.google.com/drive/folders/1Kg22JzU9NZkHe3BMPYofLNupHIUzuWKs?usp=sharing
- Model: https://drive.google.com/drive/folders/1dDyMiexnMFKcoB2MqiepaD91JjPuShMv?usp=sharing
- Demo video 1: https://drive.google.com/file/d/1MfdgkJo-Vfs0O2PNOmXKR6D25-YclozJ/view?usp=sharing
- Demo video 2: https://drive.google.com/file/d/1nSeorgnYm5wsTptmXDloDabxBHIEcEGS/view?usp=sharing

### 3.1 Tạo môi trường (khuyến nghị)

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
```

### 3.2 Cài dependencies

```bash
pip install ultralytics opencv-python shapely numpy tqdm
```

> Lưu ý về `torch`:
>
>- Nếu bạn muốn chạy GPU, hãy cài `torch` bản CUDA tương ứng với driver/CUDA máy.
>- Nếu chạy CPU vẫn được nhưng chậm.

Kiểm tra GPU nhanh:

```bash
python display.py
```

## 4) Cấu hình hệ thống

Chỉnh trong `config.py`:

- `MODEL_DETECT_PATH`: đường dẫn `.pt` cho model detect bbox.
- `MODEL_SLOT_PATH`: đường dẫn `.pt` cho model slot OBB.
- `RTSP_LINKS`: danh sách 4 link RTSP theo thứ tự CAM1..CAM4.
- `SKIP_FRAMES`: giảm tải inference (số càng lớn càng nhẹ nhưng trễ).
- `IMG_SIZE`: kích thước inference (mặc định 640).
- `TRIGGER_CLASS`: class dùng để xác định có “hộp/quy trình” (mặc định `Box`).
- `EDGE_MARGIN`: vùng biên (pixel) để tránh chốt kết quả khi box đang ra khỏi khung.
- `CAM_RULES`: luật từng stage:
	- `target`: số lượng yêu cầu của từng linh kiện.
	- `mapping`: slot nào tương ứng linh kiện nào.

Gợi ý: trong workspace hiện có sẵn các weight tại thư mục `model/`, bạn có thể trỏ bằng đường dẫn tương đối, ví dụ:

```python
MODEL_DETECT_PATH = r"model/model_object/run_component_v4_merged/weights/best.pt"
MODEL_SLOT_PATH   = r"model/model_slot/obb/train_slot_obb_s2/weights/best.pt"
```

Ví dụ (đang có sẵn):

- Stage 1 (CAM1) yêu cầu 3 `Module_Phu`
- Stage 2 (CAM2) yêu cầu 1 `Main_Board` + 1 `J-Link`
- Stage 3 (CAM3) yêu cầu 3 loại dây (có logic override theo thứ tự trái→phải)
- Stage 4 (CAM4) yêu cầu `Tui_Linh_Kien` + `Cap_USB_Trang`

## 5) Chạy chương trình realtime

```bash
python main.py
```

Phím tắt:

- `q`: thoát
- `r`: reset quy trình (xóa trạng thái đã chốt)

Cửa sổ hiển thị: **QC SYSTEM - PROCESS CONTROLLER**

- Mỗi camera hiển thị HUD (tên stage, trạng thái, 1 dòng lỗi ngắn).
- Phần footer dưới cùng hiển thị tiến trình / báo cáo tổng khi kết thúc.

Trạng thái thường gặp:

- `WAIT`: chưa có box / chưa đến stage
- `PASS`: stage đạt
- `FAIL`: stage lỗi (thiếu/sai/thừa)
- `SKIPPED`: bị bỏ qua (đã sang stage sau nhưng stage này chưa từng đạt)

## 6) Huấn luyện & xử lý dữ liệu (scripts có sẵn)

### 6.1 Train slot OBB

File: `train_processing/train.py`

- Train model slot OBB từ checkpoint `yolov8s-obb.pt`.
- `data=.../data.yaml` và nhiều tham số đang **hard-code đường dẫn**, cần sửa theo máy bạn.

Chạy:

```bash
python train_processing/train.py
```

### 6.2 Auto-label OBB → YOLO txt

File: `data_processing/autolabelobb.py`

- Input: `IMAGE_DIR`
- Output label YOLO OBB (txt): `LABEL_OUTPUT_DIR`
- (Optional) ảnh debug: `DEBUG_DIR`
- Model: `MODEL_PATH`

Chạy:

```bash
python data_processing/autolabelobb.py
```

### 6.3 Auto-label bbox → LabelMe JSON

File: `data_processing/autolabelbb.py`

- Tạo JSON theo format LabelMe (rectangle) từ model bbox.
- Mapping class id → tên nằm ở `CLASS_MAPPING`.

Chạy:

```bash
python data_processing/autolabelbb.py
```

### 6.4 Convert LabelMe JSON → YOLO OBB txt

File: `data_processing/json_txt.py`

- Đọc các file `.json` trong `INPUT_FOLDER`.
- Dùng `CLASSES` để map tên label → class id.
- Xuất `.txt` YOLO OBB (tọa độ chuẩn hóa 0..1).

Chạy:

```bash
python data_processing/json_txt.py
```

## 7) Cấu trúc thư mục (chính)

- `main.py`: vòng lặp realtime, chạy 2 model + hiển thị
- `config.py`: cấu hình model, RTSP, rule từng stage
- `camera.py`: reader RTSP dạng thread + auto reconnect
- `process_controller.py`: debounce + latch snapshot + báo cáo cuối
- `visualization.py`: HUD/summary + vẽ objects/slots
- `display.py`: kiểm tra Torch/CUDA
- `data_processing/`: tool auto label & convert nhãn
- `train_processing/`: script train

## 8) Troubleshooting

- **Không thấy hình / đen frame**: kiểm tra RTSP link, user/pass, IP; thử mở RTSP bằng VLC để test.
- **Chạy chậm / giật**: tăng `SKIP_FRAMES`, giảm `IMG_SIZE`, ưu tiên chạy GPU.
- **Lỗi `shapely`**: cài lại `pip install -U shapely`.
- **Torch báo CPU**: chạy `python display.py` để kiểm tra; cài lại Torch bản CUDA nếu cần.

## 9) Lưu ý bảo mật

`RTSP_LINKS` hiện chứa credential dạng plain-text. Không nên commit thông tin thật lên repo public; có thể chuyển qua biến môi trường hoặc file cấu hình riêng khi triển khai.

---

# QC SYSTEM – Assembly Checking with 4 Cameras + YOLOv8 (Detect + OBB Slots)

This project runs realtime from 4 RTSP cameras and uses two YOLOv8 models:

- **Detection model (bbox)**: detects components + a trigger class (default: `Box`).
- **Slot model (OBB)**: detects oriented slots (rotated rectangles) on the tray/foam.

The system matches **which object belongs to which slot** by checking whether the **object center point** lies inside the **slot OBB polygon** (Shapely `Polygon.contains(Point)`), then outputs **PASS/FAIL** per stage.

## 1) Processing flow

1. Each camera is read in a thread (frames are resized to 640x480) (`camera.py`).
2. Every `SKIP_FRAMES` frames, inference runs for both models on a batch of frames.
3. When `TRIGGER_CLASS` is detected (default `Box`):
	- Results are only “latched” when the box is inside a **safe zone** (not close to image borders) controlled by `EDGE_MARGIN`.
	- Component counts are validated using `CAM_RULES`.
4. `ProcessController` debounces noise and stores a snapshot while the box is stable.
5. The process finishes when **Stage 4 has passed** and **the box is no longer visible**.

## 2) Requirements

- Python 3.9+ (3.10/3.11 recommended)
- Windows/Linux
- NVIDIA GPU optional (recommended for smooth realtime)

Main packages:

- `ultralytics` (YOLOv8)
- `opencv-python`
- `torch`
- `shapely`
- `numpy`

## 3) Installation

### 3.0 Download data & models

- Data: https://drive.google.com/drive/folders/1Kg22JzU9NZkHe3BMPYofLNupHIUzuWKs?usp=sharing
- Models: https://drive.google.com/drive/folders/1dDyMiexnMFKcoB2MqiepaD91JjPuShMv?usp=sharing
- Demo video 1: https://drive.google.com/file/d/1MfdgkJo-Vfs0O2PNOmXKR6D25-YclozJ/view?usp=sharing
- Demo video 2: https://drive.google.com/file/d/1nSeorgnYm5wsTptmXDloDabxBHIEcEGS/view?usp=sharing

### 3.1 Create a virtual environment (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
```

### 3.2 Install dependencies

```bash
pip install ultralytics opencv-python shapely numpy tqdm
```

Torch notes:

- For GPU, install a CUDA-enabled `torch` matching your driver/CUDA.
- CPU works but will be slower.

Quick GPU check:

```bash
python display.py
```

## 4) Configuration

Edit `config.py`:

- `MODEL_DETECT_PATH`: path to the detection `.pt` file.
- `MODEL_SLOT_PATH`: path to the slot OBB `.pt` file.
- `RTSP_LINKS`: 4 RTSP links for CAM1..CAM4.
- `SKIP_FRAMES`: inference throttle (higher = lighter load, more latency).
- `IMG_SIZE`: inference size (default 640).
- `TRIGGER_CLASS`: process trigger class (default `Box`).
- `EDGE_MARGIN`: safe-zone margin in pixels.
- `CAM_RULES`: per-stage targets and slot-to-item mapping.

This workspace already contains trained weights under `model/`, e.g.:

```python
MODEL_DETECT_PATH = r"model/model_object/run_component_v4_merged/weights/best.pt"
MODEL_SLOT_PATH   = r"model/model_slot/obb/train_slot_obb_s2/weights/best.pt"
```

## 5) Run realtime

```bash
python main.py
```

Hotkeys:

- `q`: quit
- `r`: reset the process

Window title: **QC SYSTEM - PROCESS CONTROLLER**

Common states:

- `WAIT`: no box / not reached
- `PASS`: stage passed
- `FAIL`: stage failed (missing/wrong/extra)
- `SKIPPED`: bypassed (later stage reached while this stage never passed)

## 6) Training & data tools (included scripts)

### 6.1 Train slot OBB

Script: `train_processing/train.py`

- Trains an OBB slot model from `yolov8s-obb.pt`.
- Paths in the script are hard-coded; update them for your machine.

```bash
python train_processing/train.py
```

### 6.2 Auto-label OBB → YOLO txt

Script: `data_processing/autolabelobb.py`

```bash
python data_processing/autolabelobb.py
```

### 6.3 Auto-label bbox → LabelMe JSON

Script: `data_processing/autolabelbb.py`

```bash
python data_processing/autolabelbb.py
```

### 6.4 Convert LabelMe JSON → YOLO OBB txt

Script: `data_processing/json_txt.py`

```bash
python data_processing/json_txt.py
```

## 7) Project structure

- `main.py`: realtime loop (2 models + visualization)
- `config.py`: models, RTSP, per-stage rules
- `camera.py`: threaded RTSP reader + reconnect
- `process_controller.py`: debounce, snapshot, final report
- `visualization.py`: HUD/summary drawing
- `display.py`: Torch/CUDA check
- `data_processing/`: labeling & conversion tools
- `train_processing/`: training scripts

## 8) Troubleshooting

- **Black frames / no video**: verify RTSP links and credentials; test in VLC.
- **Low FPS**: increase `SKIP_FRAMES`, reduce `IMG_SIZE`, prefer GPU.
- **Shapely errors**: reinstall `pip install -U shapely`.
- **Torch running on CPU**: run `python display.py` and install a CUDA build of Torch if needed.

## 9) Security note

`RTSP_LINKS` currently contains plain-text credentials. Avoid committing real credentials to public repos; use environment variables or a separate private config for deployment.

