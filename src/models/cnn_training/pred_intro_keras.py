import argparse

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

from src.models.cnn_training.train_intro_cnn_keras import IntroGenerator


def get_ua():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", default=32, type=int)
    ap.add_argument("-t", "--trained-model", required=True)
    ap.add_argument("-i", "--input-data", required=True)
    ap.add_argument("-o", "--output-dir", default="./intro_cnn_pred")
    ap.add_argument("--encoding", choices=["01", "0255", "neg11"], default="01")
    ap.add_argument("--net", choices=["1d", "2d"])

    return ap.parse_args()


def main():
    ua = get_ua()
    input_data = ua.input_data
    cnn = load_model(ua.trained_model)

    classes = list(input_data.keys())
    idxs = list(input_data[f"{classes[0]}"].keys())
    if "1d" in ua.net:
        data_shape = input_data[f"{classes[0]}/0/x"].shape[1:][::-1]
        # flatten channels, e.g. (256, 32, 2) -> (256, 64)
        data_shape = (data_shape[0], data_shape[1] * data_shape[2])
    else:
        data_shape = input_data[f"{classes[0]}/0/x"].shape[1:][::-1]

    print("Data shape:", data_shape)
    test_gen = IntroGenerator(
        data_shape, input_data, ua.batch_size, shuffle=False, encoding=ua.encoding
    )
