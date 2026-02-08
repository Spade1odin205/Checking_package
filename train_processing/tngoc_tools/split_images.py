import os
import shutil
import math

def chia_anh_cho_moi_nguoi(thu_muc_nguon, danh_sach_nguoi):
    # 1. Kiểm tra thư mục nguồn có tồn tại không
    if not os.path.exists(thu_muc_nguon):
        print(f"Lỗi: Thư mục '{thu_muc_nguon}' không tồn tại.")
        return

    # 2. Lấy danh sách các file ảnh (lọc theo đuôi file)
    # Bạn có thể thêm các đuôi file khác vào tuple này nếu cần
    duoi_file_anh = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')
    #duoi_file_anh = ('.json',)
    
    tat_ca_file = os.listdir(thu_muc_nguon)
    danh_sach_anh = [f for f in tat_ca_file if f.lower().endswith(duoi_file_anh)]
    
    tong_so_anh = len(danh_sach_anh)
    so_nguoi = len(danh_sach_nguoi)

    if tong_so_anh == 0:
        print("Không tìm thấy file ảnh nào trong thư mục.")
        return

    print(f"Tổng số ảnh tìm thấy: {tong_so_anh}")
    print(f"Tổng số người: {so_nguoi}")

    # 3. Tạo thư mục cho từng người nếu chưa có
    duong_dan_cac_thu_muc_con = {}
    for nguoi in danh_sach_nguoi:
        duong_dan_con = os.path.join(thu_muc_nguon, nguoi)
        if not os.path.exists(duong_dan_con):
            os.makedirs(duong_dan_con)
        duong_dan_cac_thu_muc_con[nguoi] = duong_dan_con

    # 4. Tiến hành chia ảnh
    # Sử dụng thuật toán chia theo lượt (Round-robin) để đảm bảo công bằng nhất
    for i, ten_file in enumerate(danh_sach_anh):
        # Xác định ảnh này thuộc về người thứ mấy trong danh sách
        index_nguoi = i % so_nguoi
        ten_nguoi = danh_sach_nguoi[index_nguoi]
        
        # Đường dẫn nguồn và đích
        src_path = os.path.join(thu_muc_nguon, ten_file)
        dst_path = os.path.join(duong_dan_cac_thu_muc_con[ten_nguoi], ten_file)
        
        try:
            # Dùng shutil.move để di chuyển file
            # Nếu bạn chỉ muốn copy (giữ lại file gốc), hãy đổi thành shutil.copy2
            shutil.move(src_path, dst_path)
            # print(f"Đã chuyển '{ten_file}' cho {ten_nguoi}") # Bỏ comment nếu muốn xem chi tiết
        except Exception as e:
            print(f"Lỗi khi chuyển file {ten_file}: {e}")

    print("--- Hoàn tất! ---")
    # In thống kê
    for nguoi in danh_sach_nguoi:
        so_luong = len(os.listdir(duong_dan_cac_thu_muc_con[nguoi]))
        print(f"- {nguoi}: {so_luong} ảnh")

# ==========================================
# CẤU HÌNH (BẠN CHỈNH SỬA PHẦN NÀY)
# ==========================================

# 1. Đường dẫn đến folder chứa ảnh của bạn
# Lưu ý: Trên Windows nên dùng dấu gạch chéo / hoặc hai dấu gạch ngược \\
folder_anh = r"D:\Code\Python\Project\checking_package\data\dataset_phase_4" 

# 2. Danh sách tên những người cần chia
danh_sach_ten = ["huy", "loi", "hung"]

# Chạy hàm
chia_anh_cho_moi_nguoi(folder_anh, danh_sach_ten)