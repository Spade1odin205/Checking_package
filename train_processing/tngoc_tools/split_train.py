import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm  # Thư viện tạo thanh tiến trình (nếu chưa có, pip install tqdm)

def tao_cau_truc_thu_muc(output_dir):
    """Tạo cấu trúc thư mục đích: train/images, train/labels, val/images, val/labels"""
    dirs = [
        os.path.join(output_dir, 'train', 'images'),
        os.path.join(output_dir, 'train', 'labels'),
        os.path.join(output_dir, 'val', 'images'),
        os.path.join(output_dir, 'val', 'labels')
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs

def chia_tap_du_lieu(input_dir, output_dir, ty_le_train=0.8, ext_anh='.jpg', ext_label='.txt'):
    """
    Chia dữ liệu từ 1 folder chung thành train/val với cấu trúc riêng biệt.
    """
    
    # 1. Lấy danh sách các cặp file (ảnh, label)
    files = os.listdir(input_dir)
    pairs = [] # Danh sách chứa tuple (tên_ảnh, tên_label)
    
    # Tìm tất cả file ảnh
    anh_files = [f for f in files if f.endswith(ext_anh)]
    
    print(f"Đang quét thư mục: {input_dir}...")
    
    for anh in anh_files:
        ten_goc = os.path.splitext(anh)[0]
        label = ten_goc + ext_label
        
        # Chỉ thêm vào danh sách nếu tồn tại cả ảnh và label
        if label in files:
            pairs.append((anh, label))
        else:
            print(f"Cảnh báo: Bỏ qua {anh} vì không thấy label {label}")

    tong_so_file = len(pairs)
    if tong_so_file == 0:
        print("Lỗi: Không tìm thấy cặp ảnh và label nào hợp lệ.")
        return

    print(f"Tìm thấy {tong_so_file} cặp dữ liệu hợp lệ.")

    # 2. Trộn ngẫu nhiên dữ liệu
    random.shuffle(pairs)

    # 3. Tính toán số lượng chia
    so_luong_train = int(tong_so_file * ty_le_train)
    tap_train = pairs[:so_luong_train]
    tap_val = pairs[so_luong_train:]

    print(f"Chia tập dữ liệu: Train ({len(tap_train)}) - Val ({len(tap_val)})")

    # 4. Tạo thư mục đích
    tao_cau_truc_thu_muc(output_dir)

    # 5. Copy file vào thư mục đích
    def copy_files(danh_sach_cap, loai_tap):
        # loai_tap là 'train' hoặc 'val'
        print(f"\nĐang copy dữ liệu vào tập {loai_tap}...")
        
        path_images = os.path.join(output_dir, loai_tap, 'images')
        path_labels = os.path.join(output_dir, loai_tap, 'labels')

        # Sử dụng tqdm để hiện thanh phần trăm cho đẹp
        for anh, label in tqdm(danh_sach_cap, desc=f"Copy {loai_tap}"):
            src_anh = os.path.join(input_dir, anh)
            src_label = os.path.join(input_dir, label)
            
            dst_anh = os.path.join(path_images, anh)
            dst_label = os.path.join(path_labels, label)
            
            shutil.copy2(src_anh, dst_anh)   # copy2 giữ nguyên metadata
            shutil.copy2(src_label, dst_label)

    # Thực hiện copy
    copy_files(tap_train, 'train')
    copy_files(tap_val, 'val')

    print(f"\n--- HOÀN TẤT ---")
    print(f"Dữ liệu đã được lưu tại: {output_dir}")
    print(f"Cấu trúc thư mục:")
    print(f"  {output_dir}/train/images")
    print(f"  {output_dir}/train/labels")
    print(f"  {output_dir}/val/images")
    print(f"  {output_dir}/val/labels")

# --- CẤU HÌNH ---
if __name__ == "__main__":
    # Đường dẫn folder chứa tất cả ảnh và label (folder nguồn)
    INPUT_FOLDER = r"D:\Code\Python\Project\checking_package\data\data_processing\dataset_phase_4"
    
    # Đường dẫn folder bạn muốn lưu kết quả (folder đích)
    # Script sẽ tự tạo folder này nếu chưa có
    OUTPUT_FOLDER = r"D:\Code\Python\Project\checking_package\data\data\object"
    
    # Tỷ lệ chia (0.8 nghĩa là 80% train, 20% val)
    TY_LE_TRAIN = 0.8 
    
    # Đuôi mở rộng
    DUOI_ANH = '.jpg'
    DUOI_LABEL = '.txt' # Hoặc .txt
    
    chia_tap_du_lieu(INPUT_FOLDER, OUTPUT_FOLDER, TY_LE_TRAIN, DUOI_ANH, DUOI_LABEL)