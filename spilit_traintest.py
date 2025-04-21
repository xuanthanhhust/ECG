import os
import shutil
from sklearn.model_selection import train_test_split

def split_scalogram_data(input_dir, output_dir, test_size=0.5):
    """
    Chia dữ liệu ảnh scalogram thành train và test theo tỷ lệ.
    
    Args:
        input_dir (str): Thư mục chứa các ảnh scalogram đã lưu (theo nhãn).
        output_dir (str): Thư mục đầu ra để lưu dữ liệu train và test.
        test_size (float): Tỷ lệ dữ liệu dành cho test (ví dụ: 0.5 = 50%).
    """
    # Tạo thư mục đầu ra nếu chưa tồn tại
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Lặp qua từng nhãn (label)
    for label in os.listdir(input_dir):
        label_path = os.path.join(input_dir, label)
        if not os.path.isdir(label_path):
            continue

        # Lấy danh sách tất cả các ảnh trong nhãn
        images = [os.path.join(label_path, img) for img in os.listdir(label_path) if img.endswith(".png")]

        # Chia dữ liệu thành train và test
        train_images, test_images = train_test_split(images, test_size=test_size, random_state=42)

        # Tạo thư mục cho nhãn trong train và test
        train_label_dir = os.path.join(train_dir, label)
        test_label_dir = os.path.join(test_dir, label)
        os.makedirs(train_label_dir, exist_ok=True)
        os.makedirs(test_label_dir, exist_ok=True)

        # Di chuyển ảnh vào thư mục train
        for img in train_images:
            shutil.copy(img, train_label_dir)

        # Di chuyển ảnh vào thư mục test
        for img in test_images:
            shutil.copy(img, test_label_dir)

        print(f"Đã chia nhãn {label}: {len(train_images)} train, {len(test_images)} test")

    print("Hoàn thành chia dữ liệu!")

# Định nghĩa các tham số
input_dir = "./scalograms_rgb"  # Thư mục chứa ảnh scalogram đã lưu
output_dir = "./scalograms_split"      # Thư mục đầu ra để lưu train/test
test_size = 0.5                        # Tỷ lệ test (50%)

# Chia dữ liệu
split_scalogram_data(input_dir, output_dir, test_size)