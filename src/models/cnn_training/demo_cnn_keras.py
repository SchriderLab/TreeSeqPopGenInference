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
from tensorflow import keras
from tqdm import tqdm


class DemoDataGenerator(tf.keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(self, h5file, model, norm_params, log, batch_size=32, shuffle=True):
        "Initialization"
        self.h5file = h5file
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.idxs = list(self.h5file.keys())
        self.log = log
        self.model = model
        self.y_mean = norm_params[0]
        self.y_std = norm_params[1]

        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.idxs) / self.batch_size))

    def __getitem__(self, index):
        indices = self.idxs[index * self.batch_size : (index + 1) * self.batch_size]
        X = []
        y = []
        for i in indices:
            X.append(np.array(self.h5file[f"{i}/x"]))
            y.append(np.array(self.h5file[f"{i}/y"]))

        X_arr = np.concatenate(X)
        y_arr = np.concatenate(y)

        if "2d" in self.model:
            X_arr = np.expand_dims(X_arr, 3)

        if self.log == "log":
            y_arr = np.log(y_arr)

        # y_arr = norm(y_arr, self.y_mean, self.y_std)

        X_arr = np.where(X_arr == 0.0, 0.0, 255.0)

        return X_arr, y_arr

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.idxs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


def get_DemoNet(
    convDim,
    inputShape,
    convSize,
    poolSize=2,
    useDropout=True,
    numParams=5,
):
    if convDim == "2dcnn":
        convFunc = Conv2D
        poolFunc = MaxPooling2D
    else:
        convFunc = Conv1D
        poolFunc = AveragePooling1D

    b1 = Input(inputShape)
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

    merged = concatenate([flat11])
    denseMerged = Dense(256, activation="relu", kernel_initializer="normal")(merged)
    if useDropout:
        denseMerged = Dropout(0.25)(denseMerged)
    denseOutput = Dense(numParams, activation="linear")(denseMerged)
    model = Model(inputs=[b1], outputs=denseOutput)
    print(model.summary())

    model.compile(loss="mean_squared_error", optimizer="adam")

    return model


def get_Resnet(imgRows, imgCols):
    base = tf.keras.applications.ResNet50(
        weights=None, input_shape=(imgRows, imgCols, 1), include_top=False
    )
    x = Flatten()(base.output)
    x = Dense(5, activation="linear")(x)
    model = Model(inputs=base.inputs, outputs=x)
    print(model.summary())

    model.compile(loss="mean_squared_error", optimizer="adam")

    return model


def write_preds(true, preds, names, run_name):
    resdict = {}
    for i, n in enumerate(names):
        resdict[f"true_{n}"] = true[:, i]
        resdict[f"pred_{n}"] = preds[:, i]

    pd.DataFrame(resdict).to_csv(f"{run_name}/{run_name}_preds.csv", index=False)

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
                p-value: {spearmanr(preds[f"true_{name}"], preds[f"pred_{name}"])[1]:.6f},
                MSE: {mean_squared_error(preds[f"true_{name}"], preds[f"pred_{name}"]):.2f}""",
        )
        plt.legend()
        plt.plot()
        plt.title(name)
        plt.ylabel("True")
        plt.xlabel("Pred")
        plt.tight_layout()

        plt.savefig(f"{run_name}/{run_name}_{name}_preds.png")
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


def get_data(h5file):
    print("Getting data")
    X, y = [], []
    for i in tqdm(h5file.keys()):
        X.append(h5file[i]["x"])
        y.append(h5file[i]["y"])

    return np.concatenate(X), np.concatenate(y)


def get_ua():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", default=32, type=int)
    ap.add_argument("--epochs", default=10, type=int)
    ap.add_argument("--conv_blocks", default=4, type=int)
    ap.add_argument("--model", default="2dcnn", choices=["2dcnn", "1dcnn"])
    ap.add_argument("--nolog", action="store_true")
    ap.add_argument("--in_train", default="/pine/scr/d/d/ddray/demo_n512_1c.hdf5")
    ap.add_argument("--in_val", default="/pine/scr/d/d/ddray/demo_n512_1c_val.hdf5")

    return ap.parse_args()


def main():
    ua = get_ua()
    train_file = h5py.File(ua.in_train, "r")
    val_file = h5py.File(ua.in_val, "r")

    model = ua.model
    if ua.nolog:
        log = "nolog"
    else:
        log = "log"

    run_name = ua.in_train.split("/")[-1].split(".")[0]
    run_name = f"{run_name}_{model}_{log}_{ua.conv_blocks}blocks"
    os.makedirs(run_name, exist_ok=True)

    print(f"Running {run_name}")

    train_X, train_y = get_data(train_file)
    val_X, val_y = get_data(val_file)

    train_y = np.log(train_y)
    val_y = np.log(val_y)

    t_mean = np.mean(train_y, axis=0)
    t_std = np.std(train_y, axis=0)

    print("Mean", t_mean)
    print("std", t_std)

    train_yz = zscore(train_y, t_mean, t_std)
    val_yz = zscore(val_y, t_mean, t_std)

    if ua.model == "2dcnn":
        data_shape = (*train_X.shape[1:], 1)
    elif ua.model == "1dcnn":
        data_shape = train_X.shape[1:]

    cnn = get_DemoNet(model, data_shape, 3)

    print("Data shape:", data_shape)

    earlystop = keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0, patience=3, verbose=0, mode="auto"
    )
    checkpoint = keras.callbacks.ModelCheckpoint(
        f"{run_name}.hdf5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
    )
    callbacks = [earlystop, checkpoint]

    # train_X, val_X = [np.swapaxes(i, 1, 2) for i in [train_X, val_X]]

    cnn.fit(
        x=train_X,
        y=train_yz,
        validation_data=[val_X, val_yz],
        batch_size=ua.batch_size,
        validation_batch_size=ua.batch_size,
        epochs=ua.epochs,
        callbacks=callbacks,
    )

    names = ["N0", "t1", "N1", "t2", "N2"]

    preds = cnn.predict(val_X)

    raw_preds = r_zscore(preds, t_mean, t_std) + np.log(1000)
    raw_trues = r_zscore(val_yz, t_mean, t_std) + np.log(1000)

    write_preds(raw_trues, raw_preds, names, run_name)


if __name__ == "__main__":
    main()
