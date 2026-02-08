# ================= CẤU HÌNH HỆ THỐNG =================
MODEL_DETECT_PATH = r"model/model_object/run_component_v4_merged/weights/best.pt"
MODEL_SLOT_PATH   = r"model/model_slot/obb/train_slot_obb_s2/weights/best.pt"

RTSP_LINKS = [
    "rtsp://admin:CPSFLT@192.168.1.160:554/ch1/main", # CAM 1
    "rtsp://admin:DVCLRQ@192.168.1.116:554/ch1/main", # CAM 2
    "rtsp://admin:BWKUYM@192.168.1.144:554/ch1/main", # CAM 3
    "rtsp://admin:KXILGD@192.168.1.152:554/ch1/main", # CAM 4
]

SKIP_FRAMES = 2
IMG_SIZE = 640
TRIGGER_CLASS = 'Box' # Class dùng để xác định sự tồn tại của quy trình
EDGE_MARGIN = 20 # Khoảng cách từ mép màn hình (pixel) để coi là "Vùng biên"

CAM_RULES = {
    0: { 
        'title': 'CAM 1: MODULE',
        'target': {'Module_Phu': 3}, # Yêu cầu 3 module
        'mapping': {'slot_module': 'Module_Phu'}
    },
    1: { 
        'title': 'CAM 2: BOARD',
        'target': {'Main_Board': 1, 'J-Link': 1},
        'mapping': {'slot_board': 'Main_Board', 'slot_jlink': 'J-Link'}
    },
    2: { 
        'title': 'CAM 3: CABLE',
        'target': {'Day_Jumper': 1, 'Cap_Ribbon': 1, 'Cap_USB_Den': 1},
        'mapping': {'slot_rgb': 'Day_Jumper', 'slot_dayxam': 'Cap_Ribbon', 'slot_dayden': 'Cap_USB_Den'}
    },
    3: { 
        'title': 'CAM 4: ACCESSORY',
        'target': {'Tui_Linh_Kien': 1, 'Cap_USB_Trang': 1},
        'mapping': {'slot_tui': 'Tui_Linh_Kien', 'slot_daytrang': 'Cap_USB_Trang'}
    }
}