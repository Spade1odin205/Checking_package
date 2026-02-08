import cv2
import numpy as np

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