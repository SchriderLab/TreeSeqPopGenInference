import argparse
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model

from train_intro_cnn_keras import IntroGenerator


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
    input_data = h5py.File(ua.input_data, "r")
    cnn = load_model(ua.trained_model)

    if not os.path.exists(ua.output_dir):
        os.makedirs(ua.output_dir, exist_ok=True)

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
        data_shape=data_shape,
        h5file=input_data,
        np_idxs=idxs,
        batch_size=ua.batch_size,
        shuffle=False,
        encoding=ua.encoding,
    )

    true_labs = np.concatenate([i[1] for i in test_gen])
    pred_probs = cnn.predict(test_gen, verbose=1)
    pred_labs = np.argmax(pred_probs, axis=1)

    pred_dict = {f"{k}_prob": pred_probs[:, i] for i, k in enumerate(classes)}
    pred_dict["true_labs"] = np.argmax(true_labs, axis=1)
    pred_dict["pred_labs"] = pred_labs

    pd.DataFrame(pred_dict).to_csv(
        f"{ua.output_dir}/test_intro_{ua.net}_{ua.encoding}_preds.csv", index=False
    )


if __name__ == "__main__":
    main()
