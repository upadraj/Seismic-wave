import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt

from scipy.signal import stft
from sklearn.model_selection import train_test_split


def test_train_split(df, df_label, split):
    print(df_label.columns)
    df["labels"] = df_label['S']
    train, test = train_test_split(df, test_size=split, shuffle=True)

    return (train, test)


def split_3_channel(df, split=3):
    num_rows, num_cols = df.shape
    num_parts = num_cols // split
    data = np.zeros((num_rows, split, num_parts))
    for x in range(num_rows):
        a = df.iloc[x, 0:num_parts]
        b = df.iloc[x, num_parts : 2 * num_parts]
        c = df.iloc[x, 2 * num_parts : 3 * num_parts]
        data[x][0] = a
        data[x][1] = b
        data[x][2] = c

    label = df["labels"]
    label_data = label.to_numpy()

    return (data, label_data)


def short_time_fourier_transform(
    data, sampling_rate=50, nperseg=100, desired_output_shape=(50, 50)
):
    output = np.zeros(
        (data.shape[0], data.shape[1], desired_output_shape[0], desired_output_shape[1])
    )

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            _, _, Zxx = stft(data[i, j], fs=sampling_rate, nperseg=nperseg)
            spectrogram = np.abs(Zxx)
            spectrogram_resized = spectrogram[
                : desired_output_shape[0], : desired_output_shape[1]
            ]
            output[i, j] = spectrogram_resized
    return output


def wavelet_transform(data, wavelet='morl', scales=np.arange(1,51)):
    output = np.zeros((data.shape[0], data.shape[1], len(scales), data.shape[2]))

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            coeffs, _ = pywt.cwt(data[i, j], scales, wavelet)
            output[i, j] = np.abs(coeffs)
    return output


def integer_label(labels):
    unique_labels = np.unique(labels)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    print("Label Mapping:", label_mapping)
    integer_labels = np.array([label_mapping[label] for label in labels])
    targets = torch.from_numpy(integer_labels).long()
    return targets


def count_unique_colum_and_vlaues(label_data):
    unique_values, count = np.unique(label_data, return_counts=True)
    return (unique_values, count)

def znorm(features):
    mean = torch.mean(features,dim=0)
    std = torch.std(features,dim=0)
    z_score = (features - mean) / std
    return z_score