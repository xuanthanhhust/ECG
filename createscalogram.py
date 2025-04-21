import pickle
import numpy as np
import pywt
import cv2
import os
import matplotlib.pyplot as plt

def save_scalograms_by_label(filename, wavelet, scales, sampling_rate, output_dir):
    """
    Hàm để tạo scalogram từ tín hiệu ECG và lưu vào các thư mục riêng theo nhãn (label) dưới dạng ảnh RGB.
    """
    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(filename, "rb") as f:
        train_data, _ = pickle.load(f)  # Chỉ tải dữ liệu train

    for sample_index, data in enumerate(train_data):
        signal = data["signal"]
        r_peaks = data["r_peaks"]
        categories = data["categories"]

        # Tạo scalogram bằng Continuous Wavelet Transform (CWT)
        coeffs, freqs = pywt.cwt(signal, scales, wavelet, 1.0 / sampling_rate)
        before, after = 90, 110  # heartbeat segmentation interval (90 samples before, 110 samples after)

        # Chọn từng nhịp tim và lưu scalogram theo nhãn
        for i in range(1, len(r_peaks) - 1):
            if categories[i] != 4:  # Bỏ qua nhãn không hợp lệ
                # Trích xuất đoạn tín hiệu xung quanh R-peak
                scalogram = coeffs[:, r_peaks[i] - before:r_peaks[i] + after]

                # Tạo đường dẫn file theo nhãn
                label = categories[i]
                label_dir = os.path.join(output_dir, f"label_{label}")
                if not os.path.exists(label_dir):
                    os.makedirs(label_dir)

                # Lưu ảnh scalogram dưới dạng RGB
                output_path = os.path.join(label_dir, f"sample_{sample_index}_beat_{i}.png")
                save_scalogram_as_rgb(scalogram, freqs, output_path)

                print(f"Saved RGB scalogram for sample {sample_index}, beat {i}, label {label} to {output_path}")

    print("All RGB scalograms have been saved.")

def save_scalogram_as_rgb(scalogram, freqs, output_path):
    """
    Hàm lưu scalogram dưới dạng ảnh RGB sử dụng matplotlib.
    """
    plt.figure(figsize=(4, 4))
    plt.contourf(np.arange(scalogram.shape[1]), freqs, np.abs(scalogram), 100, cmap="jet")
    plt.axis("off")  # Tắt trục
    plt.tight_layout(pad=0)

    # Lưu ảnh dưới dạng RGB
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

# Định nghĩa các tham số
filename = "./dataset/mitdb.pkl"  # Đường dẫn đến file dataset
wavelet = "mexh"
sampling_rate = 360
scales = pywt.central_frequency(wavelet) * sampling_rate / np.arange(1, 101, 1)
output_dir = "./scalograms_rgb"  # Thư mục lưu scalogram RGB

# Lưu scalograms theo nhãn
save_scalograms_by_label(filename, wavelet, scales, sampling_rate, output_dir)