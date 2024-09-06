import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import stft
from sklearn.model_selection import train_test_split


def test_train_split(feature_path, label_path, split):
    df = pd.read_csv(feature_path)
    df_label = pd.read_csv(label_path)
    df["lables"] = df_label["S"]
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

    label = df["lables"]
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


def count_unique_colum_and_vlaues(label_data):
    unique_values, count = np.unique(label_data, return_counts=True)
    return (unique_values, count)
