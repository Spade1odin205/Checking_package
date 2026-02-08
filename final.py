import cv2
import threading
import time
import numpy as np
import torch
from ultralytics import YOLO
from shapely.geometry import Polygon, Point

# ================= CẤU HÌNH HỆ THỐNG =================
MODEL_DETECT_PATH = r"D:\Code\Python\Project\checking_package\best_lk.pt"
MODEL_SLOT_PATH   = r"D:\Code\Python\Project\checking_package\runs\obb\train_slot_obb_s2\weights\best.pt"

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

# ================= MODULE QUẢN LÝ QUY TRÌNH (PROCESS CONTROLLER) =================
class ProcessController:
    def __init__(self, num_stages):
        self.num_stages = num_stages
        self.reset()

    def reset(self):
        # latched_results: Lưu trữ snapshot trạng thái cuối cùng KHI CÒN BOX
        # Cấu trúc: {'is_pass': bool, 'errors': list}
        self.latched_results = [None] * self.num_stages
        
        # stage_passed_history: Dùng để check logic skip realtime (để debounce)
        self.stage_passed_history = [False] * self.num_stages
        
        # was_ever_passed: Đánh dấu stage này ĐÃ TỪNG Pass (Sticky Flag)
        self.was_ever_passed = [False] * self.num_stages

        self.fail_consecutive_count = [0] * self.num_stages 
        self.is_finished = False     
        self.process_started = False 

    def update_stage(self, stage_idx, is_pass, errors, has_box, is_stable_position):
        """
        Cập nhật trạng thái realtime.
        Quan trọng: Chỉ update 'latched_results' khi has_box = True VÀ is_stable_position = True.
        """
        if self.is_finished: return 
        
        if has_box:
            self.process_started = True
            
            # 1. LOGIC DEBOUNCE CHO REALTIME STATUS
            if is_pass:
                self.stage_passed_history[stage_idx] = True
                self.was_ever_passed[stage_idx] = True # Ghi nhớ là đã từng Pass
                self.fail_consecutive_count[stage_idx] = 0
            else:
                self.fail_consecutive_count[stage_idx] += 1
                if self.fail_consecutive_count[stage_idx] >= 3:
                    self.stage_passed_history[stage_idx] = False
            
            # 2. SNAPSHOT (CHỤP LẠI TRẠNG THÁI HIỆN TẠI)
            # CHỈ CẬP NHẬT KHI HỘP NẰM TRONG VÙNG AN TOÀN (KHÔNG Ở MÉP)
            # Điều này giúp tránh việc ghi nhận lỗi khi hộp đang đi ra khỏi màn hình
            if is_stable_position:
                self.latched_results[stage_idx] = {
                    'is_pass': is_pass,
                    'errors': list(errors) # Copy danh sách lỗi
                }

    def check_end_of_process(self, any_box_visible):
        """
        Kết thúc khi Stage 4 đã từng xong VÀ hiện tại không còn thấy hộp
        """
        # Kiểm tra xem Stage 4 đã có snapshot là PASS chưa
        stage_4_passed = False
        if self.latched_results[3] and self.latched_results[3]['is_pass']:
            stage_4_passed = True

        if self.process_started and stage_4_passed and not any_box_visible:
            self.is_finished = True
            return True
        return False

    def get_display_state(self, stage_idx, current_status):
        """
        Trả về trạng thái hiển thị cho HUD realtime (trên video cam)
        """
        if self.is_finished:
            # Lấy từ snapshot cuối cùng
            res = self.latched_results[stage_idx]
            if not res: 
                return "SKIPPED", (0, 0, 255) # Không thấy box bao giờ
            
            if self.was_ever_passed[stage_idx]:
                return "PASS", (0, 255, 0)
            elif res['is_pass']:
                return "PASS", (0, 255, 0)
            else:
                return "FAIL", (0, 0, 255)

        # Realtime logic
        if current_status == "PASS":
            return "PASS", (0, 255, 0)
        elif current_status == "FAIL":
            return "FAIL", (0, 0, 255)
        else: # WAIT
            is_skipped = False
            for next_idx in range(stage_idx + 1, self.num_stages):
                if self.stage_passed_history[next_idx]:
                    is_skipped = True
                    break
            
            if is_skipped:
                return "SKIPPED", (0, 0, 255)
            return "WAIT", (0, 255, 255)

    def get_final_report(self):
        """
        Tổng hợp lỗi dựa trên SNAPSHOT (latched_results)
        Có xử lý logic "Sticky Pass"
        """
        stage_reports = []
        is_total_success = True
        
        for i in range(self.num_stages):
            snapshot = self.latched_results[i]
            
            status = "FAIL"
            errors = []
            
            if snapshot is None:
                status = "SKIPPED"
                errors = ["Bo qua (Khong thay hop)"]
                is_total_success = False
            
            elif self.was_ever_passed[i]:
                status = "OK"
                errors = []
                
            elif snapshot['is_pass']:
                status = "OK"
                errors = []
            else:
                status = "FAIL"
                errors = snapshot['errors']
                if not errors: errors = ["Loi khong xac dinh"]
                is_total_success = False
            
            stage_reports.append({
                "id": i,
                "status": status,
                "errors": errors
            })
                
        status_code = "SUCCESS" if is_total_success else "FAILURE"
        return status_code, stage_reports

# ================= CAMERA STREAM =================
class CameraStream:
    def __init__(self, url, id):
        self.url = url 
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
                    time.sleep(0.5)
                    self.cap.open(self.url) 
            else:
                time.sleep(0.5)
                self.cap.open(self.url)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.cap.release()

# ================= VẼ GIAO DIỆN & SUMMARY =================
def draw_hud(frame, rule, status, color, messages):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, rule['title'], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    if messages:
        cv2.putText(frame, messages[0], (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

def draw_global_summary(final_grid, controller):
    footer_height = 250
    footer = np.zeros((footer_height, 1280, 3), dtype=np.uint8)
    
    # 1. TRẠNG THÁI ĐANG CHẠY
    if not controller.is_finished:
        passed_count_sticky = sum(1 for passed in controller.was_ever_passed if passed)
        
        title = f"TIEN TRINH: {passed_count_sticky}/4 HOAN THANH"
        color = (0, 255, 255)
        cv2.putText(footer, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        
        x_start = 20
        for i in range(4):
            res = controller.latched_results[i]
            st = "WAIT"
            
            # Kiểm tra xem quy trình đã đi qua stage này chưa
            has_reached_next_stage = False
            for j in range(i + 1, 4):
                if controller.latched_results[j] is not None: 
                    has_reached_next_stage = True
                    break
            
            if not res: 
                st = "WAIT"
            elif res['is_pass']: 
                st = "OK"
            else: 
                if controller.was_ever_passed[i] and has_reached_next_stage:
                    st = "OK"
                else:
                    st = "FAIL"
            
            clr = (0,255,0) if st=="OK" else ((0,0,255) if st=="FAIL" else (100,100,100))
            cv2.putText(footer, f"ST{i+1}: {st}", (x_start + i*200, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, clr, 1)

        return np.vstack((final_grid, footer))

    # 2. TRẠNG THÁI KẾT THÚC
    status_code, stage_reports = controller.get_final_report()
    
    if status_code == "SUCCESS":
        title = "KET QUA: DAT CHUAN (OK)"
        color = (0, 255, 0)
    else:
        title = "KET QUA: LOI (FAIL)"
        color = (0, 0, 255)
        
    cv2.putText(footer, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    
    col_width = 300
    y_header = 100
    y_detail = 130
    
    for i, report in enumerate(stage_reports):
        x_pos = 20 + i * col_width
        
        cv2.putText(footer, f"STAGE {i+1}", (x_pos, y_header), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        st_text = report['status']
        if st_text == "OK": st_color = (0, 255, 0)
        elif st_text == "SKIPPED": st_color = (0, 0, 255)
        else: st_color = (0, 0, 255)
        
        cv2.putText(footer, st_text, (x_pos + 100, y_header), cv2.FONT_HERSHEY_SIMPLEX, 0.8, st_color, 2)
        
        if report['errors']:
            for idx, err in enumerate(report['errors']):
                if idx > 3: break
                cv2.putText(footer, f"- {err}", (x_pos, y_detail + idx*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return np.vstack((final_grid, footer))

def draw_objects_and_slots(frame, draw_data):
    if not draw_data: return
    for (x1, y1, x2, y2, label) in draw_data['objects']:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 1)
        cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
    for (poly, s_color, s_label, s_info) in draw_data['slots']:
        cv2.polylines(frame, [poly], True, s_color, 2)
        cv2.putText(frame, s_label, (poly[0][0], poly[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, s_color, 1)

# ================= MAIN =================
def main():
    if torch.cuda.is_available():
        DEVICE = 0; device_name = torch.cuda.get_device_name(0)
    else:
        DEVICE = 'cpu'; device_name = "CPU"
    
    print(f"🚀 Start System on: {device_name}")
    
    model_det = YOLO(MODEL_DETECT_PATH)
    model_slot = YOLO(MODEL_SLOT_PATH)
    
    cams = [CameraStream(url, i) for i, url in enumerate(RTSP_LINKS)]
    time.sleep(2) 

    controller = ProcessController(num_stages=4)
    
    cam_states = [{
        'status': 'WAIT', 
        'color': (0,255,255), 
        'messages': [],
        'draw_data': {'objects': [], 'slots': []}
    } for _ in range(4)]
    
    frame_count = 0
    
    while True:
        frame_count += 1
        
        raw_frames = [cam.read() for cam in cams]
        frames_to_process = []
        indices_to_process = []
        any_box_visible = False 

        for i, f in enumerate(raw_frames):
            if f is None:
                raw_frames[i] = np.zeros((480, 640, 3), dtype=np.uint8)
                continue
            frames_to_process.append(f)
            indices_to_process.append(i)

        if len(frames_to_process) > 0 and (frame_count % SKIP_FRAMES == 0) and not controller.is_finished:
            res_det = model_det(frames_to_process, verbose=False, conf=0.5, imgsz=IMG_SIZE, device=DEVICE)
            res_slot = model_slot(frames_to_process, verbose=False, conf=0.5, imgsz=IMG_SIZE, device=DEVICE)

            for idx_in_batch, (r_det, r_slot) in enumerate(zip(res_det, res_slot)):
                real_cam_idx = indices_to_process[idx_in_batch]
                rule = CAM_RULES[real_cam_idx]
                
                temp_objects = []
                temp_slots = []
                current_messages = []
                has_box = False
                box_valid = False # Cờ kiểm tra box có nằm trong vùng an toàn không
                detected_objs = [] 
                
                for box in r_det.boxes:
                    name = model_det.names[int(box.cls)]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    if name == TRIGGER_CLASS:
                        has_box = True
                        any_box_visible = True
                        
                        # --- CHECK VÙNG AN TOÀN (SAFE ZONE) ---
                        # Chỉ coi là valid nếu box nằm cách mép màn hình một khoảng EDGE_MARGIN
                        h_img, w_img = 480, 640 # Kích thước chuẩn
                        if (x1 > EDGE_MARGIN) and (y1 > EDGE_MARGIN) and \
                           (x2 < w_img - EDGE_MARGIN) and (y2 < h_img - EDGE_MARGIN):
                            box_valid = True
                            cv2.rectangle(raw_frames[real_cam_idx], (x1, y1), (x2, y2), (0, 255, 0), 2) # Xanh lá: OK
                        else:
                            box_valid = False
                            cv2.rectangle(raw_frames[real_cam_idx], (x1, y1), (x2, y2), (0, 255, 255), 2) # Vàng: Edge
                        
                        cv2.putText(raw_frames[real_cam_idx], "BOX", (x1, y1-5), 0, 0.5, (0,255,0) if box_valid else (0,255,255), 1)
                        continue 

                    x,y,w,h = box.xywh[0].cpu().numpy()
                    detected_objs.append({'name': name, 'center': (x,y)})
                    temp_objects.append((x1, y1, x2, y2, name))

                current_counts = {k: 0 for k in rule['target']}
                
                if has_box:
                    # --- LOGIC XỬ LÝ CAM 3 (CABLE) ---
                    override_map = {}
                    ignored_indices = set()
                    
                    if real_cam_idx == 2 and r_slot.obb is not None:
                        name_map = {v: k for k, v in model_slot.names.items()}
                        target_names = ['slot_dayden', 'slot_rgb', 'slot_dayxam']
                        if all(t in name_map for t in target_names):
                            t_ids = [name_map[t] for t in target_names]
                            
                            candidates = []
                            for i, obb in enumerate(r_slot.obb):
                                if int(obb.cls) in t_ids:
                                    candidates.append({'i': i, 'x': obb.xywhr[0][0].item(), 'conf': obb.conf.item()})
                            
                            if len(candidates) >= 3:
                                candidates.sort(key=lambda x: x['conf'], reverse=True)
                                best_3 = candidates[:3]
                                best_3.sort(key=lambda x: x['x'])
                                for rank, c in enumerate(best_3): override_map[c['i']] = target_names[rank]
                                ignored_indices = set(c['i'] for c in candidates) - set(c['i'] for c in best_3)
                            else:
                                candidates.sort(key=lambda x: x['conf'], reverse=True)
                                seen_classes = set()
                                for c in candidates:
                                    original_cls_id = int(r_slot.obb[c['i']].cls)
                                    if original_cls_id not in seen_classes:
                                        seen_classes.add(original_cls_id)
                                    else:
                                        ignored_indices.add(c['i'])

                    items_with_wrong_placement = set()
                    found_misplaced_items = set()

                    if r_slot.obb is not None:
                        for i, obb in enumerate(r_slot.obb):
                            if i in ignored_indices: continue
                            
                            if i in override_map:
                                slot_name = override_map[i]
                            else:
                                slot_name = model_slot.names[int(obb.cls)]

                            if slot_name not in rule['mapping']: continue
                            req_item = rule['mapping'][slot_name]
                            
                            poly = obb.xyxyxyxy.cpu().numpy()[0].astype(int)
                            slot_poly = Polygon(poly)
                            
                            found_correct = False
                            wrong_item_name = None
                            
                            for obj in detected_objs:
                                if slot_poly.contains(Point(obj['center'])):
                                    if obj['name'] == req_item:
                                        found_correct = True
                                    else:
                                        wrong_item_name = obj['name']

                            if found_correct:
                                s_color = (0, 255, 0); s_info = "OK"
                                current_counts[req_item] = current_counts.get(req_item, 0) + 1
                            elif wrong_item_name:
                                s_color = (0, 0, 255); s_info = "WRONG"
                                if real_cam_idx == 2:
                                    # CHỈNH SỬA: Cam 3 chỉ cần đếm thiếu, bỏ qua báo sai
                                    # Không thêm vào found_misplaced_items để đảm bảo logic đếm thiếu hoạt động
                                    pass 
                                else:
                                    current_messages.append(f"Sai: {wrong_item_name} o {slot_name}")
                                    items_with_wrong_placement.add(req_item)
                                    found_misplaced_items.add(wrong_item_name)
                            else:
                                s_color = (0, 255, 255); s_info = "MISSING"
                            
                            temp_slots.append((poly, s_color, slot_name, s_info))
                    
                    is_local_pass = True
                    for item, target in rule['target'].items():
                        count = current_counts.get(item, 0)
                        if count < target:
                            is_local_pass = False
                            if (item not in found_misplaced_items) and (item not in items_with_wrong_placement):
                                current_messages.append(f"Thieu: {item} ({count}/{target})")
                        elif count > target:
                            current_messages.append(f"Thua: {item}")

                    # Cập nhật Controller với cờ box_valid
                    controller.update_stage(real_cam_idx, is_local_pass, current_messages, has_box, box_valid)
                    
                    if is_local_pass:
                        ai_verdict = "PASS"
                    else:
                        ai_verdict = "FAIL"
                        current_messages = list(set(current_messages))
                else:
                    ai_verdict = "WAIT"
                    current_messages = []

                final_status, final_color = controller.get_display_state(real_cam_idx, ai_verdict)
                
                cam_states[real_cam_idx]['status'] = final_status
                cam_states[real_cam_idx]['color'] = final_color
                cam_states[real_cam_idx]['messages'] = current_messages
                cam_states[real_cam_idx]['draw_data'] = {'objects': temp_objects, 'slots': temp_slots}

        if (frame_count % SKIP_FRAMES == 0):
            controller.check_end_of_process(any_box_visible)

        display_list = []
        for i in range(4):
            frame_show = raw_frames[i].copy()
            state = cam_states[i]
            
            if state['draw_data']:
                draw_objects_and_slots(frame_show, state['draw_data'])
            
            draw_hud(frame_show, CAM_RULES[i], state['status'], state['color'], state['messages'])
            display_list.append(frame_show)

        try:
            top = np.hstack((display_list[0], display_list[1]))
            bot = np.hstack((display_list[2], display_list[3]))
            grid_final = cv2.resize(np.vstack((top, bot)), (1280, 720))
            
            full_screen = draw_global_summary(grid_final, controller)
            
            cv2.imshow("QC SYSTEM - PROCESS CONTROLLER", full_screen)
        except Exception as e:
            pass

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'): 
            print("Resetting Process...")
            controller.reset()
            for s in cam_states:
                s['status'] = 'WAIT'; s['color'] = (0, 255, 255); s['messages'] = []

    for cam in cams: cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()