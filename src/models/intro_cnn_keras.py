import numpy as np
import tensorflow as tf
import argparse
from tensorflow import keras
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Dropout,
    AveragePooling1D,
    Flatten,
    Dense,
    Concatenate,
)
from tensorflow.keras import Model, Sequential
from tensorflow.keras.utils import to_categorical
import h5py


class LexNetDataGenerator(tf.keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(self, h5file, np_idxs, batch_size=32, shuffle=True):
        "Initialization"
        self.h5file = h5file
        self.batch_size = batch_size
        self.np_idxs = np_idxs
        self.shuffle = shuffle
        self.classes = h5file.keys()
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
                x_arr = np.array(self.h5file[f"{model}/{i}/x"])
                X.append(x_arr)
                y.extend(
                    to_categorical(
                        [idx] * x_arr.shape[0], num_classes=len(self.classes)
                    )
                )

        return np.array(X), np.array(y)

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.np_idxs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


class CNNGenerator(LexNetDataGenerator):
    def __init__(self, h5file, np_idxs, batch_size=32, shuffle=True):
        "Initialization"
        super().__init__(h5file, np_idxs, batch_size, shuffle)

    def __getitem__(self, index):
        indices = self.np_idxs[index * self.batch_size : (index + 1) * self.batch_size]
        X = []
        y = []
        for i in indices:
            for idx, model in enumerate(self.classes):
                x_arr = np.array(self.h5file[f"{model}/{i}/x"])
                X.append(x_arr.T)
                yvals = [idx] * x_arr.shape[0]
                y.extend(yvals)

                print(x_arr.shape, len(yvals))

        print(np.concatenate(X).shape, np.array(to_categorical(y, num_classes=3)).shape)
        return np.concatenate(X), np.concatenate(y)


def get_net(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(256, kernel_size=2, activation="relu", input_shape=(input_shape)))
    model.add(Conv1D(128, kernel_size=2, activation="relu"))
    model.add(AveragePooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Conv1D(128, kernel_size=2, activation="relu"))
    model.add(AveragePooling1D(pool_size=2))
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

    return model


def get_ua():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", default=32)
    ap.add_argument("--epochs", default=10)
    ap.add_argument("--in_train", default="/pine/scr/d/d/ddray/dros_n512.hdf5")
    ap.add_argument("--in_val", default="/pine/scr/d/d/ddray/dros_n512_val.hdf5")
    ap.add_argument("--model", default="LexNet")

    return ap.parse_args()


def main():
    ua = get_ua()
    train_file = h5py.File(ua.in_train, "r")
    val_file = h5py.File(ua.in_val, "r")

    classes = list(train_file.keys())
    train_len = len(train_file[f"{classes[0]}"].keys())
    val_len = len(val_file[f"{classes[0]}"].keys())

    model = get_net((512, 34), 3)
    train_dl = CNNGenerator(train_file, range(train_len), batch_size=ua.batch_size)
    val_dl = CNNGenerator(val_file, range(val_len), batch_size=ua.batch_size)

    model.fit(x=train_dl, validation_data=val_dl, epochs=ua.epochs)


if __name__ == "__main__":
    main()
