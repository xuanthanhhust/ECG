from concurrent.futures import ProcessPoolExecutor

import cv2
import joblib
import numpy as np
import pywt
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from functools import partial
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, Initializer, LRScheduler, TensorBoard
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from torch.backends import cudnn
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import ConfusionMatrixDisplay

cudnn.benchmark = False
cudnn.deterministic = True

torch.manual_seed(0)


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # Conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # Pool1
            nn.Conv2d(96, 256, kernel_size=5, padding=2),  # Conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # Pool2
            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # Conv3
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),  # Conv4
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # Conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # Pool5
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(256 * 6 * 6 +4, 4096), # Add 4 index for RR Interval
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, 4),  # Output layer
        )
    
    def forward(self, x1, x2):
        x1 = self.features(x)
        x1 = self.avgpool(x)
        x1 = torch.flatten(x, 1)  # Flatten the output of the last pooling layer
        x = torch.cat((x1, x2), dim=1)  # Concatenate the features and RR intervals
        x = self.classifier(x)
        return x


def worker(data, wavelet, scales, sampling_period):
    # heartbeat segmentation interval
    before, after = 90, 110

    coeffs, frequencies = pywt.cwt(data["signal"], scales, wavelet, sampling_period)
    r_peaks, categories = data["r_peaks"], data["categories"]

    # for remove inter-patient variation
    avg_rri = np.mean(np.diff(r_peaks))

    x1, x2, y, groups = [], [], [], []
    for i in range(len(r_peaks)):
        if i == 0 or i == len(r_peaks) - 1:
            continue

        if categories[i] == 4:  # remove AAMI Q class
            continue

        # Convert scalogram to RGB scale
        grayscale = cv2.resize(coeffs[:, r_peaks[i] - before: r_peaks[i] + after], (227, 227))
        x1.append(cv2.merge([grayscale, grayscale, grayscale]))  # Convert to RGB
        x2.append([
            r_peaks[i] - r_peaks[i - 1] - avg_rri,  # previous RR Interval
            r_peaks[i + 1] - r_peaks[i] - avg_rri,  # post RR Interval
            (r_peaks[i] - r_peaks[i - 1]) / (r_peaks[i + 1] - r_peaks[i]),  # ratio RR Interval
            np.mean(np.diff(r_peaks[np.maximum(i - 10, 0):i + 1])) - avg_rri  # local RR Interval
        ])
        y.append(categories[i])
        groups.append(data["record"])

    return x1, x2, y, groups


def load_data(wavelet, scales, sampling_rate, filename="./dataset/mitdb.pkl"):
    import pickle
    from sklearn.preprocessing import RobustScaler

    with open(filename, "rb") as f:
        train_data, test_data = pickle.load(f)

    cpus = 22 if joblib.cpu_count() > 22 else joblib.cpu_count() - 1  # for multi-process

    # for training
    x1_train, x2_train, y_train, groups_train = [], [], [], []
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        for x1, x2, y, groups in executor.map(
                partial(worker, wavelet=wavelet, scales=scales, sampling_period=1. / sampling_rate), train_data):
            x1_train.append(x1)
            x2_train.append(x2)
            y_train.append(y)
            groups_train.append(groups)

    x1_train = np.concatenate(x1_train, axis=0).astype(np.float32)  # No need for expand_dims, as RGB already has 3 channels
    x2_train = np.concatenate(x2_train, axis=0).astype(np.float32)
    y_train = np.concatenate(y_train, axis=0).astype(np.int64)
    groups_train = np.concatenate(groups_train, axis=0)

    # for test
    x1_test, x2_test, y_test, groups_test = [], [], [], []
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        for x1, x2, y, groups in executor.map(
                partial(worker, wavelet=wavelet, scales=scales, sampling_period=1. / sampling_rate), test_data):
            x1_test.append(x1)
            x2_test.append(x2)
            y_test.append(y)
            groups_test.append(groups)

    x1_test = np.concatenate(x1_test, axis=0).astype(np.float32)  # No need for expand_dims, as RGB already has 3 channels
    x2_test = np.concatenate(x2_test, axis=0).astype(np.float32)
    y_test = np.concatenate(y_test, axis=0).astype(np.int64)
    groups_test = np.concatenate(groups_test, axis=0)

    #normalization
    scaler = RobustScaler()
    x2_train = scaler.fit_transform(x2_train)
    x2_test = scaler.transform(x2_test)

    return (x1_train, x2_train, y_train, groups_train), (x1_test, x2_test, y_test, groups_test)


def plot_confusion_matrix_percentage(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, normalize='true')  # Normalize để hiển thị phần trăm
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_true))
    disp.plot(cmap="Blues", values_format=".2%")  # Hiển thị dưới dạng phần trăm
    return disp


def main():
    sampling_rate = 360

    wavelet = "mexh"  # mexh, morl, gaus8, gaus4
    scales = pywt.central_frequency(wavelet) * sampling_rate / np.arange(1, 101, 1)

    (x1_train, x2_train, y_train, groups_train), (x1_test, x2_test, y_test, groups_test) = load_data(
        wavelet=wavelet, scales=scales, sampling_rate=sampling_rate)
    print("Data loaded successfully!")

    log_dir = "./logs/{}".format(wavelet)
    shutil.rmtree(log_dir, ignore_errors=True)

    callbacks = [
        Initializer("[conv|fc]*.weight", fn=torch.nn.init.kaiming_normal_),  # Khởi tạo trọng số bằng phương pháp Kaiming Normal
        Initializer("[conv|fc]*.bias", fn=partial(torch.nn.init.constant_, val=0.0)),  # Khởi tạo bias bằng giá trị 0
        LRScheduler(policy=StepLR, step_size=5, gamma=0.1),  # Giảm learning rate sau mỗi 5 epoch với hệ số gamma=0.1
        EpochScoring(scoring=make_scorer(f1_score, average="macro"), lower_is_better=False, name="valid_f1"),  # Tính F1-score macro sau mỗi epoch
        TensorBoard(SummaryWriter(log_dir))  # Ghi log vào TensorBoard để trực quan hóa quá trình huấn luyện
    ]
    net = NeuralNetClassifier(  # skorch is extensive package of pytorch for compatible with scikit-learn
        AlexNet,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        lr=0.001,
        max_epochs=30,
        batch_size=1024,
        train_split=predefined_split(Dataset({"x1":x1_test, "x2": x2_test}, y_test)),
        verbose=1,
        device="cuda",
        callbacks=callbacks,
        iterator_train__shuffle=True,
        optimizer__weight_decay=0,
    )
    net.fit({"x1":x1_train, "x2": x2_train}, y_train)
    y_true, y_pred = y_test, net.predict({"x1":x1_train, "x2": x2_train})

    print(confusion_matrix(y_true, y_pred))
    plot_confusion_matrix_percentage(y_true, y_pred)
    print(classification_report(y_true, y_pred, digits=4))

    net.save_params(f_params="./models/model_{}.pkl".format(wavelet))


if __name__ == "__main__":
    main()