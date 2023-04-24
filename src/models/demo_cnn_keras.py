import numpy as np
import tensorflow as tf
from glob import glob
import argparse
from tensorflow import keras

from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Conv2D,
    MaxPooling2D,
    Dropout,
    AveragePooling1D,
    Flatten,
    Dense,
    concatenate,
)
from tensorflow.keras import Model
import h5py
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd


class DataGenerator(tf.keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(self, filelist, training, sort=True, pad_size=512, shuffle=True):
        "Initialization"
        self.filelist = filelist
        self.shuffle = shuffle
        self.training = training
        self.pad_size = pad_size
        self.sort = sort
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return len(self.filelist)

    def resort_min_diff(self, amat):
        ###assumes your snp matrix is indv. on rows, snps on cols
        mb = NearestNeighbors(n_neighbors=len(amat), metric="manhattan").fit(amat)
        v = mb.kneighbors(amat)
        smallest = np.argmin(v[0].sum(axis=1))
        return amat[v[1][smallest]]

    def __getitem__(self, index):
        f = h5py.File(self.filelist[index], "r")
        X = []
        y = []

        keys = sorted(list(f["demo"].keys()))

        X = np.zeros((len(keys), f["demo/0/x_0"].shape[0], self.pad_size))

        for idx, i in enumerate(keys):
            x_arr = np.array(f[f"demo/{i}/x_0"])

            if self.sort:
                x = self.resort_min_diff(x_arr)
            else:
                x = x_arr

            if x.shape[1] > self.pad_size:
                x = x[:, : self.pad_size]

            X[idx, : x.shape[0], : x.shape[1]] += x
            y.extend(np.array(f[f"demo/{i}/y"]))

        y = np.stack(y)

        if self.training:
            np.random.shuffle(X)
            np.random.shuffle(y)

        return X, np.stack(y)

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        if self.shuffle == True:
            np.random.shuffle(self.filelist)


def get_LexNet(ksize=2, l2_lambda=0.0001):
    b1_0 = Input(shape=(5000, 208))
    b1 = Conv1D(
        128 * 2,
        kernel_size=ksize,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(l2_lambda),
    )(b1_0)
    b1 = Conv1D(
        128 * 2,
        kernel_size=ksize,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(l2_lambda),
    )(b1)
    b1 = MaxPooling1D(pool_size=ksize)(b1)
    b1 = Dropout(0.2)(b1)

    b1 = Conv1D(
        128 * 2,
        kernel_size=ksize,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(l2_lambda),
    )(b1)
    b1 = MaxPooling1D(pool_size=ksize)(b1)
    b1 = Dropout(0.2)(b1)

    b1 = Conv1D(
        128 * 2,
        kernel_size=ksize,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(l2_lambda),
    )(b1)
    b1 = AveragePooling1D(pool_size=ksize)(b1)
    b1 = Dropout(0.2)(b1)

    b1 = Conv1D(
        128 * 2,
        kernel_size=ksize,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(l2_lambda),
    )(b1)
    b1 = AveragePooling1D(pool_size=ksize)(b1)
    b1 = Dropout(0.2)(b1)
    b1 = Flatten()(b1)

    b2_0 = Input(shape=(5000,))
    b2 = Dense(
        64,
        input_shape=(5000,),
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(l2_lambda),
    )(b2_0)
    b2 = Dropout(0.1)(b2)

    merged = Concatenate(axis=1)([b1, b2])
    merged = Dense(
        256,
        activation="relu",
        kernel_initializer="normal",
        kernel_regularizer=keras.regularizers.l2(l2_lambda),
    )(merged)
    merged = Dropout(0.25)(merged)
    merged_output = Dense(5, activation="softmax")(merged)

    model = Model(inputs=[b1_0, b2_0], outputs=merged_output)
    print(model.summary())
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

    return model


def get_DemoNet(
    convDim, imgRows, imgCols, convSize, poolSize=2, useDropout=True, numParams=5
):
    if convDim == "2d":
        inputShape = (imgRows, imgCols, 1)
        convFunc = Conv2D
        poolFunc = MaxPooling2D
    else:
        inputShape = (imgRows, imgCols)
        convFunc = Conv1D
        poolFunc = AveragePooling1D

    b1 = Input(shape=inputShape)
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

    # b2 = Input(shape=(imgRows,))
    # dense21 = Dense(32, activation="relu")(b2)
    # if useDropout:
    #    dense21 = Dropout(0.25)(dense21)

    merged = concatenate([flat11])  # , dense21])
    denseMerged = Dense(256, activation="relu", kernel_initializer="normal")(merged)
    if useDropout:
        denseMerged = Dropout(0.25)(denseMerged)
    denseOutput = Dense(numParams)(denseMerged)
    model = Model(inputs=[b1], outputs=denseOutput)
    print(model.summary())

    model.compile(loss="mean_squared_error", optimizer="adam")

    return model


def write_preds(true, preds, names):
    resdict = {}
    for i, n in enumerate(names):
        resdict[f"true_{n}"] = true[:, i]
        resdict[f"pred_{n}"] = preds[:, i]

    pd.DataFrame(resdict).to_csv("preds.csv", index=False)


def plot_preds(true, pred, names):
    fig, axes = plt.subplots(len(names))
    for i, n in enumerate(names):
        axes[i].scatter(true, pred)
        axes[i].annotate(
            f"MSE: {mean_squared_error(true[:, i], pred[:, i])}", (0.1, 0.5)
        )
        axes[i].set_title(n)

    plt.tight_layout()

    plt.savefig("Predplots.png")


def get_ua():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", default=10)
    ap.add_argument(
        "--data_dir", default="/pine/scr/d/d/ddray/demography_regression_sims_1e5_h5/"
    )
    ap.add_argument("--model", default="DemoNet")

    return ap.parse_args()


def main():
    ua = get_ua()
    all_files = glob(f"{ua.data_dir}/*.hdf5")
    all_files = [i for i in all_files if "demo" in list(h5py.File(i, "r").keys())]
    train_files = [i for i in all_files if "val" not in i]
    val_files = [i for i in all_files if "val" in i]

    train_dl = DataGenerator(train_files, training=True)
    val_dl = DataGenerator(val_files, training=False, shuffle=False)

    model = get_DemoNet("2d", 50, 512, 3)

    earlystop = keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0, patience=3, verbose=0, mode="auto"
    )
    checkpoint = keras.callbacks.ModelCheckpoint(
        "weights.hdf5", monitor="val_loss", verbose=1, save_best_only=True, mode="min"
    )
    callbacks = [earlystop, checkpoint]

    model.fit(
        train_dl,
        validation_data=val_dl,
        epochs=ua.epochs,
        callbacks=callbacks,
    )

    names = ["N0", "t1", "N1", "t2", "N2"]

    all_preds = []
    all_trues = []
    for i in range(val_dl.__len__()):
        X, y = val_dl[i]
        all_preds.append(model.predict(X))
        all_trues.append(y)

    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_trues)

    write_preds(trues, preds, names)
    plot_preds(trues, preds, names)


if __name__ == "__main__":
    main()
