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
    AveragePooling2D,
    Conv1D,
    Conv2D,
    Dense,
    Input,
    Concatenate,
    Dropout,
    Flatten,
)
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tqdm import tqdm


class RhoGenerator(tf.keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        h5file,
        np_idxs,
        train_mean,
        train_std,
        network="1d",
        batch_size=32,
        shuffle=True,
    ):
        "Initialization"
        self.h5file = h5file
        self.batch_size = batch_size
        self.np_idxs = np_idxs
        self.shuffle = shuffle
        self.network = network
        self.train_mean = train_mean
        self.train_std = train_std
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.np_idxs) / self.batch_size))

    def __getitem__(self, index):
        indices = self.np_idxs[index * self.batch_size : (index + 1) * self.batch_size]
        X = []
        p = []
        y = []
        for i in indices:
            try:
                X.append(np.array(self.h5file[f"{i}/x"]))
                p.append(np.array(self.h5file[f"{i}/p"]))
                y.append(np.array(self.h5file[f"{i}/y"][:, 1]))
            except:
                continue

        if "2d" in self.network:
            x_arr = np.expand_dims(np.concatenate(X), axis=3)

        else:
            x_arr = np.concatenate(X)

        p_arr = np.concatenate(p)
        y_arr = zscore(np.log(np.concatenate(y)), self.train_mean, self.train_std)

        return (x_arr, p_arr), y_arr

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.np_idxs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


def get_DemoNet(
    convDim,
    inputShape,
    convSize,
    poolSize=2,
    useDropout=True,
    numParams=1,
):
    if convDim == "2d":
        convFunc = Conv2D
        poolFunc = AveragePooling2D
    else:
        convFunc = Conv1D
        poolFunc = AveragePooling1D

    l2_lambda = 0.0001

    i1 = Input(inputShape[0])
    b1 = convFunc(
        1250,
        kernel_size=convSize,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(l2_lambda),
        input_shape=inputShape,
    )(i1)
    b1 = convFunc(
        256,
        kernel_size=convSize,
        kernel_regularizer=keras.regularizers.l2(l2_lambda),
        activation="relu",
    )(b1)
    b1 = poolFunc(pool_size=poolSize)(b1)
    b1 = Dropout(0.25)(b1)
    b1 = convFunc(
        256,
        kernel_size=convSize,
        kernel_regularizer=keras.regularizers.l2(l2_lambda),
        activation="relu",
    )(b1)
    b1 = poolFunc(pool_size=poolSize)(b1)
    b1 = Dropout(0.25)(b1)
    b1 = Flatten()(b1)

    i2 = Input(inputShape[1])
    b2 = Dense(
        64,
        input_shape=(418,),
        kernel_regularizer=keras.regularizers.l2(l2_lambda),
        activation="relu",
    )(i2)
    b2 = Dropout(0.1)(b2)

    merged = Concatenate()([b1, b2])
    denseMerged = Dense(256, activation="relu", kernel_initializer="normal")(merged)
    if useDropout:
        denseMerged = Dropout(0.25)(denseMerged)
    denseOutput = Dense(numParams, activation="linear")(denseMerged)
    model = Model(inputs=[i1, i2], outputs=denseOutput)
    print(model.summary())

    model.compile(loss="mean_squared_error", optimizer="adam")

    return model


def write_preds(true, preds, names, run_name):
    print(true)
    print(preds)
    resdict = {}
    for i, n in enumerate(names):
        resdict[f"true_{n}"] = true
        resdict[f"pred_{n}"] = preds

    pd.DataFrame(resdict).to_csv(f"{run_name}_preds.csv", index=False)

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
            MSE: {mean_squared_error(preds[f"true_{name}"], preds[f"pred_{name}"]):.2f}""",
        )
        plt.legend()
        plt.plot()
        plt.title(name)
        plt.ylabel("True")
        plt.xlabel("Pred")
        plt.tight_layout()

        plt.savefig(f"{run_name}_{name}_preds.png")
        plt.clf()


def get_norm_params(h5file, log):
    print("Calculating normalization parameters")
    y = []
    for i in h5file.keys():
        y.append(h5file[i]["y"][:, 1])

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
    X, p, y = [], [], []
    for i in tqdm(h5file.keys()):
        X.append(h5file[i]["x"])
        p.append(h5file[i]["p"])
        y.append(h5file[i]["y"][:, 1])

    return np.concatenate(X), np.concatenate(p), np.concatenate(y)


def get_ua():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", default=32, type=int)
    ap.add_argument("--epochs", default=10, type=int)
    ap.add_argument("--net", default="2d", choices=["2d", "1d"])
    ap.add_argument("--nolog", action="store_true")
    ap.add_argument("--in-train")
    ap.add_argument("--in-val")

    return ap.parse_args()


def main():
    ua = get_ua()
    train_file = h5py.File(ua.in_train, "r")
    val_file = h5py.File(ua.in_val, "r")

    model = ua.net
    if ua.nolog:
        log = "nolog"
    else:
        log = "log"

    run_name = ua.in_train.split("/")[-1].split(".")[0]
    run_name = f"rho/{run_name}_{model}_{log}/ones_{run_name}_{model}_{log}"

    print(f"Running {run_name}")

    train_X, train_p, train_y = get_data(train_file)
    val_X, val_p, val_y = get_data(val_file)

    train_y = np.log(train_y)
    val_y = np.log(val_y)

    t_mean = np.mean(train_y, axis=0)
    t_std = np.std(train_y, axis=0)

    print("Mean", t_mean)
    print("std", t_std)

    val_yz = zscore(val_y, t_mean, t_std)

    print("Train X shape:", train_X.shape)
    print("Train p shape:", train_p.shape)

    if model == "2d":
        data_shape = ((*train_X.shape[1:], 1), train_p.shape[1])
    elif model == "1d":
        data_shape = (train_X.shape[1:], train_p.shape[1])

    del train_y, train_X, train_p

    print(data_shape)
    cnn = get_DemoNet(model, data_shape, 3)

    print("Data shape:", data_shape)

    train_dl = RhoGenerator(
        train_file, np.arange(len(train_file)), t_mean, t_std, model, ua.batch_size
    )
    val_dl = RhoGenerator(
        val_file, np.arange(len(val_file)), t_mean, t_std, model, ua.batch_size
    )

    i = train_dl[0]
    print(
        "Data shape out of generator (X, p, y):",
        i[0][0].shape,
        i[0][1].shape,
        i[1].shape,
    )

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

    cnn.fit(
        train_dl,
        validation_data=val_dl,
        epochs=ua.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    names = ["rho"]

    preds = cnn.predict((val_X, val_p)).flatten()

    raw_preds = np.exp(r_zscore(preds, t_mean, t_std)) / 20000
    raw_trues = np.exp(r_zscore(val_yz, t_mean, t_std)) / 20000

    write_preds(raw_trues, raw_preds, names, run_name)


if __name__ == "__main__":
    main()
