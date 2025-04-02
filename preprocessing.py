#Preprocessing 
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import scipy.signal as sg
import wfdb
# Tạo đối tượng Path
PATH = Path("dataset")

sampling_rate = 360 

#non-beat labels
invalid_labels = ['|', '~', '!', '+', '[', ']', '"', 'x']

#for correct R-peak location
tol = 0.05

def worker(record):
    # Read ECG signal from ML II and labels
    signal = wfdb.rdrecord((PATH / record).as_posix(), channels=[0]).p_signal[:, 0] 
    # as_posix() = chuyển thành chuỗi đường dẫn, channel 0 = ML II channel in MIT-BIH dataset

    annotation = wfdb.rdann((PATH / record).as_posix(),'atr')
    r_peaks, labels = annotation.sample, np.array(annotation.symbol)

    # filtering uses a 200-ms witdh median filtrer and 600-ms width median filter 
    baseline = sg.medfilt(sg.medfilt(signal, int(0.2* sampling_rate) - 1), int(0.6 * sampling_rate) - 1)
    filtered_signal = signal - baseline

    #remove non-beat labels 
    indices = [i for i, label in enumerate(labels) if label not in invalid_labels]
    r_peaks, labels = r_peaks[indices], labels[indices]

    #allign r-peaks 
    newR = []
    for r_peak in r_peaks:
        r_left = np.maximum(r_peak - int(tol * sampling_rate), 0)
        r_right = np.minimum(r_peak + int(tol * sampling_rate), len(filtered_signal))
        newR.append(r_left + np.argmax(filtered_signal[r_left : r_right]))
    
    r_peaks = np.array(newR, dtype="int")

    # normalize inter-patient variation
    nornamlized_signal = filtered_signal / np.mean(filtered_signal[r_peaks])

    # catagories 
    AAMI = {
        "N": 0, "L": 0, "R": 0, "e": 0, "j": 0, # N
        "A": 1, "a": 1, "S": 1, "J": 1,         # SVEB
        "V": 2, "E": 2, #VEB
        "F": 3, #F
        "/": 4, "f": 4, "Q": 4 #Q
    }
    categories = [AAMI[label] for label in labels]

    return {
        "record": record,
        "signal": nornamlized_signal, "r_peaks": r_peaks, "categories": categories
    }

# Xử lý song song trên cpu để xử lý dữ liệu thô 
if __name__ == "__main__":
    print("ok!")
    # for multi-processing
    cpus = 22 if joblib.cpu_count() > 22 else joblib.cpu_count() -1 

    train_records = [
        '101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
        '122', '124', '201', '203', '205', '207', '208', '209', '215', '220',
        '223', '230'
    ]
    print("__________Start processing train data_________")
    with ProcessPoolExecutor(max_workers= 10) as executor:
        train_data = [result for result in executor.map(worker, train_records)]
    
    # Phần hiển thị thông tin R-peaks đơn giản
    # print("\nTHỐNG KÊ R-PEAKS:")
    # total_r_peaks = 0
    
    # for data in train_data:
    #     record_name = data['record']
    #     num_r_peaks = len(data['r_peaks'])
    #     total_r_peaks += num_r_peaks
    #     print(f"Record {record_name}: {num_r_peaks} R-peaks")
    
    # print(f"\nTỔNG SỐ R-PEAKS: {total_r_peaks}")

        # Thống kê chi tiết test data
    print("\nTHỐNG KÊ CHI TIẾT - TẬP TEST:")
    print("{:<10} {:<15} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        'Record', 'Total R-peaks', 'N (0)', 'SVEB (1)', 'VEB (2)', 'F (3)', 'Q (4)'))
    print("-"*80)
    train_total = np.zeros(5, dtype=int)
    for data in train_data:
        record_name = data['record']
        r_peaks = data['r_peaks']
        categories = data['categories']
        
        unique, counts = np.unique(categories, return_counts=True)
        count_dict = dict(zip(unique, counts))
        full_counts = [count_dict.get(i, 0) for i in range(5)]
        train_total += full_counts
        
        print("{:<10} {:<15} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
            record_name, len(r_peaks), *full_counts))
    
    print("\nTỔNG CỘNG TRAIN:")
    print("{:<10} {:<15} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        '', sum(train_total), *train_total))
    print("PHÂN BỐ (%): {:.1f}% {:.1f}% {:.1f}% {:.1f}% {:.1f}%".format(
        *(train_total/sum(train_total)*100)))
    

    test_records = [
        '100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
        '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
        '233', '234'
    ]

    print("__________Start processing test data_________")
    with ProcessPoolExecutor(max_workers= 10) as executor:
        test_data = [result for result in executor.map(worker, test_records)]

    # Thống kê chi tiết test data
    print("\nTHỐNG KÊ CHI TIẾT - TẬP TEST:")
    print("{:<10} {:<15} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        'Record', 'Total R-peaks', 'N (0)', 'SVEB (1)', 'VEB (2)', 'F (3)', 'Q (4)'))
    print("-"*80)
    
    test_total = np.zeros(5, dtype=int)
    for data in test_data:
        record_name = data['record']
        r_peaks = data['r_peaks']
        categories = data['categories']
        
        unique, counts = np.unique(categories, return_counts=True)
        count_dict = dict(zip(unique, counts))
        full_counts = [count_dict.get(i, 0) for i in range(5)]
        test_total += full_counts
        
        print("{:<10} {:<15} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
            record_name, len(r_peaks), *full_counts))
    
    print("\nTỔNG CỘNG TEST:")
    print("{:<10} {:<15} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        '', sum(test_total), *test_total))
    print("PHÂN BỐ (%): {:.1f}% {:.1f}% {:.1f}% {:.1f}% {:.1f}%".format(
        *(test_total/sum(test_total)*100)))
    


    with open((PATH / "mitdb.pkl").as_posix(), "wb") as f:
        pickle.dump((train_data, test_data), f, protocol=4 )

    print(" The process is end ")

