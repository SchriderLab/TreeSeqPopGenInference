import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model
from keras.layers import (
    AveragePooling1D,
    Conv1D,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D,
    MaxPooling2D,
    concatenate,
)
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tqdm import tqdm


def get_DemoNet(
    convDim,
    inputShape,
    convSize,
    poolSize=2,
    useDropout=True,
    numParams=5,
):
    if convDim == "2d":
        convFunc = Conv2D
        poolFunc = MaxPooling2D
    else:
        convFunc = Conv1D
        poolFunc = AveragePooling1D

    b1 = Input(inputShape[0])
    conv11 = convFunc(128, kernel_size=convSize, activation="relu")(b1)
    pool11 = poolFunc(pool_size=poolSize)(conv11)
    if useDropout:
        pool11 = Dropout(0.25)(pool11)
    conv12 = convFunc(128, kernel_size=2, activation="relu")(pool11)
    pool12 = poolFunc(pool_size=poolSize)(conv12)
    if useDropout:
        pool12 = Dropout(0.25)(pool12)
    conv13 = convFunc(128, kernel_size=2, activation="relu")(pool12)
    pool13 = poolFunc(pool_size=poolSize)(conv13)
    if useDropout:
        pool13 = Dropout(0.25)(pool13)
    conv14 = convFunc(128, kernel_size=2, activation="relu")(pool13)
    pool14 = poolFunc(pool_size=poolSize)(conv14)
    if useDropout:
        pool14 = Dropout(0.25)(pool14)
    flat11 = Flatten()(pool14)

    b2 = Input(shape=(inputShape[1],))
    dense21 = Dense(32, activation="relu")(b2)
    if useDropout:
        dense21 = Dropout(0.25)(dense21)

    merged = concatenate([flat11, dense21])
    denseMerged = Dense(256, activation="relu", kernel_initializer="normal")(merged)
    if useDropout:
        denseMerged = Dropout(0.25)(denseMerged)
    denseOutput = Dense(numParams, activation="linear")(denseMerged)
    model = Model(inputs=[b1, b2], outputs=denseOutput)
    print(model.summary())

    model.compile(loss="mean_squared_error", optimizer="adam")

    return model


def write_preds(true, preds, names, run_name):
    resdict = {}
    for i, n in enumerate(names):
        resdict[f"true_{n}"] = true[:, i]
        resdict[f"pred_{n}"] = preds[:, i]

    pd.DataFrame(resdict).to_csv(f"demo/{run_name}/{run_name}_preds.csv", index=False)

    plot_preds(resdict, names, run_name)


def plot_preds(preds, names, run_name):
    for i, name in enumerate(names):
        plt.scatter(preds[f"true_{name}"], preds[f"pred_{name}"])
        m, b = np.polyfit(preds[f"true_{name}"], preds[f"pred_{name}"], 1)
        plt.plot(
            preds[f"pred_{name}"],
            m * preds[f"pred_{name}"] + b,
            color="black",
            label=f"""Spearmans rho: {spearmanr(preds[f"true_{name}"], preds[f"pred_{name}"])[0]:.2f}, 
            RMSE: {mean_squared_error(preds[f"true_{name}"], preds[f"pred_{name}"], squared=False):.2f}""",
        )
        # if "t" in name:
        #    plt.ylim(-8, 2)
        #    plt.xlim(-8, 2)
        # elif "N" in name:
        #    plt.ylim(4, 10)
        #    plt.xlim(4, 10)
        plt.legend()
        plt.plot()
        plt.title(name)
        plt.xlabel(f"True {name}")
        plt.ylabel(f"Pred {name}")
        plt.tight_layout()

        plt.savefig(f"demo/{run_name}/{run_name}_{name}_preds.png")
        plt.clf()


def get_norm_params(h5file, log):
    print("Calculating normalization parameters")
    y = []
    for i in h5file.keys():
        y.append(h5file[i]["y"])

    data = np.concatenate(y)
    data = data

    if log == "log":
        data = np.log(data)

    return np.mean(data, axis=0), np.std(data, axis=0)


def zscore(data, y_mean, y_std):
    return (data - y_mean) / y_std


def r_zscore(data, y_mean, y_std):
    return y_mean + (data * y_std)


def getDistancesBetweenSnps(positionVectors):
    distVectors = []
    for i in range(len(positionVectors)):
        currVec = []
        prevPos = 0.0
        for j in range(len(positionVectors[i])):
            currVec.append(positionVectors[i][j] - prevPos)
            prevPos = positionVectors[i][j]
        currVec.append(1.0 - prevPos)
        distVectors.append(currVec)
    return distVectors


def get_data(h5file):
    print("Getting data")
    X, p, y = [], [], []
    for i in tqdm(h5file.keys()):
        X.append(h5file[i]["x"])
        p.append(h5file[i]["p"])
        y.append(h5file[i]["y"])

    return np.concatenate(X), np.concatenate(p), np.concatenate(y)


def get_ua():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", default=32, type=int)
    ap.add_argument("--epochs", default=10, type=int)
    ap.add_argument(
        "--conv-blocks", default=4, type=int, help="Number of convolutional blocks"
    )
    ap.add_argument("--net", default="2d", choices=["2d", "1d"])
    ap.add_argument(
        "--encoding",
        choices=["01", "0255", "neg11"],
        default="01",
        help="Encoding of the input data",
    )
    ap.add_argument("--in-train", help="Path to the training data hdf5 file")
    ap.add_argument("--in-val", help="Path to the validation data hdf5 file")

    return ap.parse_args()


def main():
    ua = get_ua()
    train_file = h5py.File(ua.in_train, "r")
    val_file = h5py.File(ua.in_val, "r")

    model = ua.net

    run_name = ua.in_train.split("/")[-1].split(".")[0]
    run_name = f"demo_{ua.encoding}_{run_name}_{model}_{ua.conv_blocks}blocks"
    os.makedirs(f"demo/{run_name}", exist_ok=True)

    print(f"Running {run_name}")

    train_X, train_p, train_y = get_data(train_file)
    val_X, val_p, val_y = get_data(val_file)

    train_y = np.log(train_y)
    val_y = np.log(val_y)

    t_mean = np.mean(train_y, axis=0)
    t_std = np.std(train_y, axis=0)

    print("Mean", t_mean)
    print("std", t_std)

    train_yz = zscore(train_y, t_mean, t_std)
    val_yz = zscore(val_y, t_mean, t_std)

    train_p = np.array(getDistancesBetweenSnps(train_p))
    val_p = np.array(getDistancesBetweenSnps(val_p))

    if ua.encoding == "01":
        p_scaler = MinMaxScaler((0.0, 1.0)).fit(train_p)
        pass
    elif ua.encoding == "0255":
        p_scaler = MinMaxScaler((0.0, 255.0)).fit(train_p)
        train_X = np.where(train_X > 0, 255, 0)
        val_X = np.where(val_X > 0, 255, 0)
    elif ua.encoding == "neg11":
        p_scaler = MinMaxScaler((-1.0, 1.0)).fit(train_p)
        train_X = np.where(train_X > 0, 1, -1)
        val_X = np.where(val_X > 0, 1, -1)

    train_p = p_scaler.transform(train_p)
    val_p = p_scaler.transform(val_p)

    if ua.net == "2d":
        data_shape = ((*train_X.shape[1:], 1), train_p.shape[1])
    elif ua.net == "1d":
        data_shape = (train_X.shape[1:], train_p.shape[1])

    cnn = get_DemoNet(model, data_shape, 3)

    print("Data shape:", data_shape)

    earlystop = keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0, patience=3, verbose=0, mode="auto"
    )
    checkpoint = keras.callbacks.ModelCheckpoint(
        f"demo/{run_name}/{run_name}_model.hdf5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
    )
    callbacks = [earlystop, checkpoint]

    cnn.fit(
        x=[train_X, train_p],
        y=train_yz,
        validation_data=[[val_X, val_p], val_yz],
        batch_size=ua.batch_size,
        validation_batch_size=ua.batch_size,
        epochs=ua.epochs,
        callbacks=callbacks,
        verbose=2,  # type: ignore
    )

    names = ["N0", "t1", "N1", "t2", "N2"]

    preds = cnn.predict((val_X, val_p))

    # Scaled for comparison by Ne
    raw_preds = r_zscore(preds, t_mean, t_std)
    raw_trues = r_zscore(val_yz, t_mean, t_std)

    raw_preds[:, [0, 2, 4]] = raw_preds[:, [0, 2, 4]] + np.log(10000)
    raw_trues[:, [0, 2, 4]] = raw_trues[:, [0, 2, 4]] + np.log(10000)

    write_preds(raw_trues, raw_preds, names, run_name)


if __name__ == "__main__":
    main()
