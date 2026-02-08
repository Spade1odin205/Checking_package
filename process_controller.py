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
        # Dùng để khôi phục trạng thái OK nếu bị lỗi nhiễu khi di chuyển hộp
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
                # Nếu đã từng Pass, coi như OK bất chấp snapshot cuối cùng (lúc rút hộp) bị lỗi
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