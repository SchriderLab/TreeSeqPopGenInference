import argparse
from model_viz import cm_analysis, count_parameters

import torch
import torch.nn.functional as F
import h5py
import configparser
from data_loaders import GenotypeMatrixGenerator
import torch.nn as nn

from torch.nn import CrossEntropyLoss, NLLLoss, DataParallel
from collections import deque

from sklearn.metrics import accuracy_score

import logging, os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from layers import LexStyleNet


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing."""

    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def make_batch(indices, idx, batch_size, generator):
    batch_X = []
    batch_y = []
    inds = indices[idx * batch_size : (idx + 1) * batch_size]

    for j in inds:
        x1, y = generator[j]
        batch_X.append(x1)
        batch_y.append(y)

    return torch.tensor(np.concatenate(batch_X)), torch.tensor(np.concatenate(batch_y))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument(
        "--ifile", default="/pine/scr/d/d/ddray/dros_n512.hdf5", help="training h5 file"
    )
    parser.add_argument(
        "--ifile_val",
        default="/pine/scr/d/d/ddray/dros_n512_val.hdf5",
        help="validation h5 file",
    )
    parser.add_argument("--odir", default="None")

    parser.add_argument("--n_epochs", default="100")
    parser.add_argument("--lr", default="0.00001")  # original is 0.00001
    parser.add_argument("--n_early", default="10")
    parser.add_argument("--lr_decay", default="None")

    parser.add_argument("--L", default="32", help="tree sequence length")
    parser.add_argument("--batch_size", default="16", type=int)

    parser.add_argument("--weights", default="None")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if args.odir != "None":
        if not os.path.exists(args.odir):
            os.system("mkdir -p {}".format(args.odir))
            logging.debug("root: made output directory {0}".format(args.odir))
        else:
            os.system("rm -rf {0}".format(os.path.join(args.odir, "*")))

    return args


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using " + str(device) + " as device")

    generator = GenotypeMatrixGenerator(h5py.File(args.ifile, "r"), models=None)
    validation_generator = GenotypeMatrixGenerator(
        h5py.File(args.ifile_val, "r"), models=None
    )
    model = LexStyleNet()

    if args.weights != "None":
        checkpoint = torch.load(args.weights, map_location=device)
        model.load_state_dict(checkpoint)

    classes = generator.models

    model = model.to(device)
    print(model)
    count_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    # for writing the training
    result = dict()
    result["epoch"] = []
    result["loss"] = []
    result["acc"] = []
    result["val_loss"] = []
    result["val_acc"] = []

    losses = deque(maxlen=500)
    accuracies = deque(maxlen=500)
    criterion = NLLLoss()

    if args.lr_decay != "None":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, float(args.lr_decay)
        )
    else:
        lr_scheduler = None

    min_val_loss = np.inf

    num_batches = int(generator.__len__() / args.batch_size)
    indices = range(num_batches)

    for epoch in range(int(args.n_epochs)):
        model.train()
        for idx in range(0, num_batches, args.batch_size):
            x1, y = make_batch(indices, idx, args.batch_size, generator)

            print(x1.shape)
            print(y.shape)

            y = y.to(device)
            x1 = x1.to(device)

            optimizer.zero_grad()
            y_pred = model(x1)

            loss = criterion(y_pred, y)

            y_pred = y_pred.detach().cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)
            y = y.detach().cpu().numpy()

            accuracies.append(accuracy_score(y, y_pred))

            losses.append(loss.detach().item())

            loss.backward()
            optimizer.step()

            # change back to 100
            if (idx) % 10 == 0:
                logging.info(
                    "root: Epoch: {}/{}, Step: {}/{}, Loss: {}, Acc: {}".format(
                        epoch + 1,
                        args.n_epochs,
                        idx,
                        num_batches,
                        np.mean(losses),
                        np.mean(accuracies),
                    )
                )

        train_loss = np.mean(losses)
        train_acc = np.mean(accuracies)

        val_losses = []
        val_accs = []

        logging.info("validating...")
        model.eval()

        Y = []
        Y_pred = []
        with torch.no_grad():
            for idx, batch in enumerate(range(0, num_batches, args.batch_size)):
                batch_X = []
                batch_y = []

                inds = indices[idx * args.batch_size : (idx + 1) * args.batch_size]

                for j in inds:
                    x1, y = generator[j]
                    batch_X.append(x1)
                    batch_y.append(y)

                x1 = torch.tensor(np.concatenate(batch_X))
                y = torch.tensor(np.concatenate(batch_y))

                y = y.to(device)
                x1 = x1.to(device)

                y_pred = model(x1)

                loss = criterion(y_pred, y)

                y_pred = y_pred.detach().cpu().numpy()
                y = y.detach().cpu().numpy()
                y_pred = np.argmax(y_pred, axis=1)

                Y.extend(y)
                Y_pred.extend(y_pred)

                val_accs.append(accuracy_score(y, y_pred))
                val_losses.append(loss.detach().item())

        val_loss = np.mean(val_losses)
        val_acc = np.mean(val_accs)

        result["epoch"].append(epoch)
        result["val_loss"].append(val_loss)
        result["val_acc"].append(val_acc)
        result["loss"].append(train_loss)
        result["acc"].append(train_acc)

        logging.info(
            "root: Epoch {}, Val Loss: {:.3f}, Val Acc: {:.3f}".format(
                epoch + 1, val_loss, val_acc
            )
        )

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            print("saving weights...")
            torch.save(model.state_dict(), os.path.join(args.odir, "best.weights"))

            # do this for all the examples:
            cm_analysis(
                Y,
                np.round(Y_pred),
                os.path.join(args.odir, "confusion_matrix_best.png"),
                classes,
            )

            early_count = 0
        else:
            early_count += 1

            # early stop criteria
            if early_count > int(args.n_early):
                break

        validation_generator.on_epoch_end()

        df = pd.DataFrame(result)
        df.to_csv(os.path.join(args.odir, "metric_history.csv"), index=False)

        if lr_scheduler is not None:
            logging.info("lr for next epoch: {}".format(lr_scheduler.get_lr()))
            lr_scheduler.step()

    """
    plt.rc('font', family = 'Helvetica', size = 12)
    plt.rcParams.update({'figure.autolayout': True})
    fig = plt.figure(figsize=(12, 8), dpi=100)
    
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('epoch')
    ax.set_ylabel('negative ll loss')
    ax.set_title('training loss history')
    
    ax
    """


if __name__ == "__main__":
    main()
