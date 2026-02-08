import os

def xoa_anh_khong_label_hai_thu_muc(thu_muc_anh, thu_muc_label, duoi_mo_rong_anh='.jpg', duoi_mo_rong_label='.json'):
    """
    Xóa các file ảnh trong một thư mục (thu_muc_anh) nếu không có file label tương ứng 
    trong thư mục label (thu_muc_label).

    Args:
        thu_muc_anh (str): Đường dẫn đến thư mục chứa các file ảnh (.jpg, .png, ...).
        thu_muc_label (str): Đường dẫn đến thư mục chứa các file label (.txt, .json, ...).
        duoi_mo_rong_anh (str): Đuôi mở rộng của file ảnh (ví dụ: '.jpg').
        duoi_mo_rong_label (str): Đuôi mở rộng của file label (ví dụ: '.txt').
    """
    
    # 1. Lấy danh sách tên tất cả các file label
    try:
        tat_ca_file_label = os.listdir(thu_muc_label)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy thư mục label tại đường dẫn: {thu_muc_label}")
        return

    # 2. Tạo tập hợp (set) chứa tên gốc của các file label để tra cứu nhanh
    ten_label_goc = set()
    label_len = len(duoi_mo_rong_label)
    
    for file in tat_ca_file_label:
        if file.endswith(duoi_mo_rong_label):
            # Lấy phần tên file (không bao gồm phần mở rộng)
            ten_file_goc = file[:-label_len]
            ten_label_goc.add(ten_file_goc)
            
    # 3. Lấy danh sách tên tất cả các file ảnh
    try:
        tat_ca_file_anh = os.listdir(thu_muc_anh)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy thư mục ảnh tại đường dẫn: {thu_muc_anh}")
        return

    # 4. Lặp qua các file ảnh và kiểm tra
    so_luong_da_xoa = 0
    anh_len = len(duoi_mo_rong_anh)
    
    for file_anh in tat_ca_file_anh:
        if file_anh.endswith(duoi_mo_rong_anh):
            # Lấy phần tên file gốc của ảnh
            ten_file_goc = file_anh[:-anh_len]
            
            # Kiểm tra xem tên file này có trong tập hợp tên label không
            if ten_file_goc not in ten_label_goc:
                duong_dan_anh = os.path.join(thu_muc_anh, file_anh)
                
                try:
                    # Xóa file ảnh
                    os.remove(duong_dan_anh)
                    print(f"Đã xóa: {file_anh}")
                    so_luong_da_xoa += 1
                except OSError as e:
                    print(f"Lỗi khi xóa file {file_anh}: {e}")

    print(f"\n--- Hoàn thành ---")
    print(f"Tổng số ảnh đã xóa: {so_luong_da_xoa}")

# --- Cách sử dụng ---
# THAY ĐỔI ĐƯỜNG DẪN THƯ MỤC CỦA BẠN VÀO ĐÂY
# Đảm bảo sử dụng ký tự R (raw string) hoặc dấu gạch chéo ngược kép (\\) 
# cho đường dẫn trên Windows để tránh lỗi.
thu_muc_chua_anh = r"D:\Code\Python\Project\Multi_cam\frame_new\TuanNgoc\TuanNgoc" 
thu_muc_chua_label = r"D:\Code\Python\Project\Multi_cam\frame_new\TuanNgoc\TuanNgoc_label" 

# Thực thi hàm
xoa_anh_khong_label_hai_thu_muc(
    thu_muc_chua_anh, 
    thu_muc_chua_label, 
    duoi_mo_rong_anh='.jpg', 
    duoi_mo_rong_label='.json'
)