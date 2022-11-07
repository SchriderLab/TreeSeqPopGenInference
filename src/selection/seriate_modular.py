import argparse
import multiprocessing as mp
import random
import sys

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from seriate import seriate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

# from scipy.optimize import linear_sum_assignment


def seriate_xpos(x, pos, metric="cosine"):
    Dx = pdist(x, metric=metric)
    Dx[np.where(np.isnan(Dx))] = 0.0
    ix = seriate(Dx, timeout=0)

    return x[ix], pos[ix], ix


def pad_matrix_resnet(x, new_size, axis):
    # expects a genotype matrix (channels,sites,n_individuals,) shaped
    s = x.shape[axis]

    if new_size > s:
        x_ = np.zeros((x.shape[0], new_size - s))
        x = np.concatenate([x, x_], axis=axis)
    elif new_size < s:
        segment = s - new_size
        start = random.randint(0, segment)
        return x[:, start : start + new_size, :]
    return x

def pad_matrix_x(x, new_size, axis):
    # expects a genotype matrix (channels,sites,n_individuals,) shaped
    s = x.shape[axis]

    if new_size > s:
        x_ = np.zeros((new_size-s,x.shape[1]))
        x = np.concatenate([x, x_], axis=axis)
    elif new_size < s:
        segment = s - new_size
        start = random.randint(0, segment)
        return x[:, start : start + new_size, :]
    return x

def pad_matrix_pos(x, new_size, axis):
    # expects a genotype matrix (channels,sites,n_individuals,) shaped
    s = x.shape[axis]

    if new_size > s:
        x_ = np.zeros((new_size-s))
        x = np.concatenate([x, x_], axis=axis)
    elif new_size < s:
        segment = s - new_size
        start = random.randint(0, segment)
        return x[:, start : start + new_size, :]
    return x

def get_ua():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i",
        "--input",
        dest="input_npz",
        type=str,
        default="final.split.up.seln.big.npz",
    )
    ap.add_argument("-o", "--outdir", dest="outdir", type=str)
    ap.add_argument("--dataset-index", dest="dataset_ind", type=int)
    ap.add_argument("--sample-index", dest="sample_ind", type=int)
    ap.add_argument("--test", dest="test", action="store_true")
    uap = ap.parse_args()
    return uap


def main():
    # Get indices for sample and dataset
    ua = get_ua()

    u = np.load(ua.input_npz, encoding="latin1", allow_pickle=True)

    if ua.test:
        xtest_, ytest_, postest_ = [u[i][:2000] for i in "xtest ytest postest".split()]

        postest_ = pad_sequences(
            postest_, padding="post", value=-1.0, dtype="float32", maxlen=5000
        )  # unpack and pad from (2000,) to (2000,5000)
        xtest_ = pad_sequences(
            xtest_, padding="post", maxlen=5000
        )  # unpack and pad from (2000,) to (2000,5000,208)

        np.savez(
            "test_sel.npz", **{"xtest": xtest_, "ytest": ytest_, "postest": postest_}
        )

    else:
        if not ua.sample_ind and ua.dataset_ind:
            raise argparse.ArgumentError(
                ua.dataset_ind,
                "Training mode was selected but no indices were provided",
            )

        xtrain = u[f"xtrain_{ua.dataset_ind}"][ua.sample_ind]
        ytrain = u[f"ytrain_{ua.dataset_ind}"][ua.sample_ind]
        postrain = u[f"postrain_{ua.dataset_ind}"][ua.sample_ind]

        postrain = pad_matrix_pos(postrain,5000,axis=0)  # to (3000,5000)
        xtrain = pad_matrix_x(xtrain,5000,axis=0)
        xtrain = pad_matrix_resnet(xtrain, 256, axis=1)
        # postrain_ = pad_matrix_resnet(postrain_,256,axis=2)

        x_sorted, pos, idx = seriate_xpos(xtrain, postrain)

        np.savez(
            f"{ua.outdir}/train_{ua.dataset_ind}_{ua.sample_ind}.npz",
            **{
                "x_train": x_sorted,
                "y_train": ytrain,
                "pos_train": pos,
                "idx_train": idx,
            },
        )


if __name__ == "__main__":
    main()
