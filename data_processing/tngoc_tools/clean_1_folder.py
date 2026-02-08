import os

def xoa_anh_khong_label_mot_thu_muc(thu_muc_du_lieu, duoi_mo_rong_anh='.jpg', duoi_mo_rong_label='.json'):
    """
    Xóa các file ảnh trong thư mục nếu không có file label tương ứng cùng tên trong cùng thư mục đó.
    """
    
    # 1. Kiểm tra thư mục tồn tại
    if not os.path.exists(thu_muc_du_lieu):
        print(f"Lỗi: Không tìm thấy thư mục tại: {thu_muc_du_lieu}")
        return

    # Lấy danh sách tất cả các file trong thư mục
    tat_ca_file = os.listdir(thu_muc_du_lieu)

    # 2. Tạo tập hợp (set) chứa tên gốc của các file label
    ten_label_goc = set()
    
    for file in tat_ca_file:
        if file.endswith(duoi_mo_rong_label):
            # Dùng os.path.splitext để tách tên và đuôi mở rộng an toàn hơn
            ten_goc = os.path.splitext(file)[0]
            ten_label_goc.add(ten_goc)
            
    print(f"Tìm thấy {len(ten_label_goc)} file label ({duoi_mo_rong_label}).")

    # 3. Lặp qua các file ảnh và kiểm tra
    so_luong_da_xoa = 0
    
    for file in tat_ca_file:
        if file.endswith(duoi_mo_rong_anh):
            ten_goc_anh = os.path.splitext(file)[0]
            
            # Kiểm tra xem tên file ảnh có nằm trong tập hợp label không
            if ten_goc_anh not in ten_label_goc:
                duong_dan_anh = os.path.join(thu_muc_du_lieu, file)
                
                try:
                    os.remove(duong_dan_anh)
                    print(f"Đã xóa: {file} (Không có label)")
                    so_luong_da_xoa += 1
                except OSError as e:
                    print(f"Lỗi khi xóa file {file}: {e}")

    print(f"\n--- Hoàn thành ---")
    print(f"Tổng số ảnh đã xóa: {so_luong_da_xoa}")

# --- Cách sử dụng ---

# Đường dẫn thư mục chứa cả ảnh và label
thu_muc_data = r"D:\Code\Python\Project\checking_package\data\data_processing\loi_check_final\loi" 

# Thực thi hàm
xoa_anh_khong_label_mot_thu_muc(
    thu_muc_data, 
    duoi_mo_rong_anh='.jpg', 
    duoi_mo_rong_label='.json' # Đổi thành .txt nếu bạn dùng format YOLO
)