import argparse
import random

import h5py
import numpy as np
import pandas as pd
import plotting_utils as pu
import tensorflow as tf
from keras import Model, Sequential
from keras.layers import (
    AveragePooling1D,
    AveragePooling2D,
    Concatenate,
    Conv1D,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D,
    MaxPooling2D,
)
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tqdm import tqdm


class IntroGenerator(tf.keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        data_shape,
        h5file,
        np_idxs,
        encoding,
        network="1d",
        batch_size=32,
        shuffle=True,
    ):
        "Initialization"
        self.data_shape = data_shape
        self.h5file = h5file
        self.batch_size = batch_size
        self.np_idxs = np_idxs
        self.encoding = encoding
        self.shuffle = shuffle
        self.classes = h5file.keys()
        self.network = network
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.np_idxs) / self.batch_size))

    def __getitem__(self, index):
        indices = self.np_idxs[index * self.batch_size : (index + 1) * self.batch_size]
        X = []
        y = []
        for i in indices:
            for idx, model in enumerate(self.classes):
                try:
                    x_arr = np.array(self.h5file[f"{model}/{i}/x"])
                    X.append(x_arr)
                    y.append(
                        to_categorical(
                            [idx] * x_arr.shape[0], num_classes=len(self.classes)
                        )
                    )
                except:
                    continue

        if "2d" in self.network:
            x_arr = np.concatenate(X)
            x_arr = x_arr.reshape((len(X) * 4, *self.data_shape))
        else:
            x_arr = np.swapaxes(np.concatenate(X), 1, 2).reshape(
                (len(X) * 4, *self.data_shape)
            )

        if self.encoding == "01":
            pass
        elif self.encoding == "0255":
            x_arr = np.where(x_arr > 0, 255, 0)
        elif self.encoding == "neg11":
            x_arr = np.where(x_arr > 0, 1, -1)

        return x_arr, np.concatenate(y)

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.np_idxs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


def get_CNN(input_shape, num_classes):
    if len(input_shape) == 3:
        convf = Conv2D
        poolf = AveragePooling2D
    else:
        convf = Conv1D
        poolf = AveragePooling1D

    model = Sequential()
    model.add(convf(256, kernel_size=2, activation="relu", input_shape=(input_shape)))
    model.add(convf(128, kernel_size=2, activation="relu"))
    model.add(poolf(pool_size=2))
    model.add(Dropout(0.25))
    model.add(convf(128, kernel_size=2, activation="relu"))
    model.add(poolf(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

    print(model.summary())

    return model


def get_ua():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", default=32, type=int)
    ap.add_argument("--epochs", default=10, type=int)
    ap.add_argument("--in-train", help="Path to training data hdf5 file")
    ap.add_argument("--in-val", help="Path to validation data hdf5 file")
    ap.add_argument("--out-prefix", help="Prefix for output files")
    ap.add_argument("--net", choices=["1d", "2d"])
    ap.add_argument(
        "--encoding",
        choices=["01", "0255", "neg11"],
        default="01",
        help="Encoding of data",
    )

    return ap.parse_args()


def main():
    ua = get_ua()
    train_file = h5py.File(ua.in_train, "r")
    val_file = h5py.File(ua.in_val, "r")

    conv = ua.net

    classes = list(train_file.keys())
    if "1d" in conv:
        data_shape = train_file[f"{classes[0]}/0/x"].shape[1:][::-1]
        # flatten channels, e.g. (256, 32, 2) -> (256, 64)
        data_shape = (data_shape[0], data_shape[1] * data_shape[2])
    else:
        data_shape = train_file[f"{classes[0]}/0/x"].shape[1:][::-1]

    print("Data shape:", data_shape)

    # They're in batches of 4 but that only matters in the stack
    train_idxs = list(train_file[f"{classes[0]}"].keys())
    val_idxs = random.sample(
        list(val_file[f"{classes[0]}"].keys()), int(6144 / 4)
    )  # Dylan needed to subset for comp time with GCN

    if "1d" in conv:
        model = get_CNN(data_shape, len(classes))
    elif "2d" in conv:
        model = get_CNN(data_shape, len(classes))

    train_dl = IntroGenerator(
        data_shape, train_file, train_idxs, network=conv, batch_size=ua.batch_size
    )  # type: ignore
    val_dl = IntroGenerator(
        data_shape, val_file, val_idxs, network=conv, batch_size=ua.batch_size
    )

    i = train_dl[0]

    print("Data shape out of generator (X, y):", i[0].shape, i[1].shape)

    earlystop = keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0, patience=3, verbose=0, mode="auto"
    )
    checkpoint = keras.callbacks.ModelCheckpoint(
        f"{ua.out_prefix}_intro_{conv}.hdf5",
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
    )
    callbacks = [earlystop, checkpoint]

    history = model.fit(
        x=train_dl,
        validation_data=val_dl,
        epochs=ua.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    trues = []
    preds = []
    for X, y in tqdm(val_dl, desc="Predicting"):
        preds.append(np.array(model(X)))
        trues.append(y)

    preds_arr = np.concatenate(preds)
    pred_classes = np.argmax(preds_arr, axis=1)
    trues = np.concatenate(trues)
    true_classes = np.argmax(trues, axis=1)

    lab_dict = {i: c for i, c in enumerate(classes)}

    true_labs = [lab_dict[i] for i in true_classes]
    pred_labs = [lab_dict[i] for i in pred_classes]

    pd.DataFrame(
        {
            "true": true_labs,
            "pred": pred_labs,
        }
    ).to_csv(f"{ua.out_prefix}_{conv}_results.csv", index=False, sep="\t")

    pu.plot_confusion_matrix(
        ".",
        confusion_matrix(
            true_labs,
            pred_labs,
            labels=["none", "ba", "ab"],
        ),
        ["No introgression", "sech-to-sim", "sim-to-sech"],
        normalize=True,
        title=f"{ua.out_prefix}_intro_{conv}_confmat",
    )
    print(classification_report(true_labs, pred_labs))


if __name__ == "__main__":
    main()
