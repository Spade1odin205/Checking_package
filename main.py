import cv2
import time
import numpy as np
import torch
from ultralytics import YOLO
from shapely.geometry import Polygon, Point

# Import từ các file thành phần
from config import *
from camera import CameraStream
from process_controller import ProcessController
from visualization import draw_hud, draw_global_summary, draw_objects_and_slots

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
                        h_img, w_img = 480, 640
                        if (x1 > EDGE_MARGIN) and (y1 > EDGE_MARGIN) and \
                           (x2 < w_img - EDGE_MARGIN) and (y2 < h_img - EDGE_MARGIN):
                            box_valid = True
                            cv2.rectangle(raw_frames[real_cam_idx], (x1, y1), (x2, y2), (0, 255, 0), 2)
                        else:
                            box_valid = False
                            cv2.rectangle(raw_frames[real_cam_idx], (x1, y1), (x2, y2), (0, 255, 255), 2)
                        
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