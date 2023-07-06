import argparse
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from train_demo_cnn_keras import *


def get_ua():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", default=32, type=int)
    ap.add_argument("-t", "--trained-model", required=True)
    ap.add_argument("-i", "--test-data", required=True)
    ap.add_argument(
        "-td", "--training-data", required=True, help="For calculating mean/std"
    )
    ap.add_argument("-o", "--output-dir", default="./demo_cnn_pred")
    ap.add_argument("--encoding", choices=["01", "0255", "neg11"], default="01")
    ap.add_argument("--net", choices=["1d", "2d"])

    return ap.parse_args()


def main():
    ua = get_ua()
    train_data = h5py.File(ua.training_data, "r")
    test_data = h5py.File(ua.test_data, "r")
    train_X, train_p, train_y = get_data(train_data)
    test_X, test_p, test_y = get_data(test_data)

    cnn = load_model(ua.trained_model)

    if not os.path.exists(ua.output_dir):
        os.makedirs(ua.output_dir, exist_ok=True)

    train_mean = np.mean(train_y, axis=0)
    train_std = np.std(train_y, axis=0)

    print("Mean", train_mean)
    print("std", train_std)

    train_p = np.array(getDistancesBetweenSnps(train_p))
    test_p = np.array(getDistancesBetweenSnps(test_p))

    if ua.encoding == "01":
        p_scaler = MinMaxScaler((0.0, 1.0)).fit(train_p)
        pass
    elif ua.encoding == "0255":
        p_scaler = MinMaxScaler((0.0, 255.0)).fit(train_p)
        train_X = np.where(train_X > 0, 255, 0)
        test_X = np.where(test_X > 0, 255, 0)
    elif ua.encoding == "neg11":
        p_scaler = MinMaxScaler((-1.0, 1.0)).fit(train_p)
        train_X = np.where(train_X > 0, 1, -1)
        test_X = np.where(test_X > 0, 1, -1)

    train_p = p_scaler.transform(train_p)
    test_p = p_scaler.transform(test_p)

    if ua.net == "2d":
        data_shape = ((*train_X.shape[1:], 1), train_p.shape[1])
    elif ua.net == "1d":
        data_shape = (train_X.shape[1:], train_p.shape[1])

    print("Data shape:", data_shape)

    preds = cnn.predict([test_X, test_p])
    raw_preds = r_zscore(preds, train_mean, train_std)
    raw_preds[:, [0, 2, 4]] = raw_preds[:, [0, 2, 4]] + np.log(10000)
    raw_trues = np.log(test_y)

    names = ["N0", "t1", "N1", "t2", "N2"]
    write_preds(
        raw_trues,
        raw_preds,
        names,
        f"{ua.output_dir}/test_intro_{ua.net}_{ua.encoding}",
    )


if __name__ == "__main__":
    main()
