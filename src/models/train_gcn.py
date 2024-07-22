from datetime import datetime
from torch_geometric.utils import to_dense_batch
import time
import matplotlib.pyplot as plt
import argparse
from model_viz import cm_analysis, count_parameters

import torch
import torch.nn.functional as F
import h5py
import configparser
from data_loaders import TreeSeqGenerator, TreeSeqGeneratorV2, TreeSeqGeneratorV3
#from gcn import GCN, Classifier, SequenceClassifier
import torch.nn as nn
from gcn_layers import GATSeqClassifier, GATConvClassifier

from torch.nn import CrossEntropyLoss, NLLLoss, DataParallel, BCEWithLogitsLoss
from collections import deque

from sklearn.metrics import accuracy_score

import logging
import os

import numpy as np
import pandas as pd
import matplotlib
# cluster safe
matplotlib.use('Agg')


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """

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


"""
example command:
"python3 src/models/train_gcn.py --ifile /pine/scr/d/d/ddray/seln_trees_i2_l128_scattered.hdf5 --ifile_val /pine/scr/d/d/ddray/seln_trees_i2_l128_scattered.hdf5 
    --odir training_results/seln_rnn_i6/ --n_steps 1000 --lr 0.0001 --L 128 --n_gcn_iter 32 --lr_decay 0.98 --pad_l --in_dim 3 --n_classes 5 --n_per_batch 4"
"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--verbose", action="store_true",
                        help="display messages")
    parser.add_argument("--ifile", default="None", help="training h5 file")
    parser.add_argument("--ifile_val", default="None",
                        help="validation h5 file")
    parser.add_argument("--odir", default="None",
                        help="output directory where we save weights and logs of training progress")

    parser.add_argument("--n_epochs", default="100",
                        help="number of training epochs to perform")
    parser.add_argument("--lr", default="0.00001",
                        help="learning rate for Adam optimizer")
    parser.add_argument("--n_early", default="10",
                        help="number of epochs to early stop if validation loss hasn't gone down")
    parser.add_argument("--lr_decay", default="None",
                        help="if specified as a float will apply exponential learning rate decay (not recommended).  other learning schedules could help in theory, but arent currently implemented")

    parser.add_argument("--n_per_batch", default="4",
                        help="number of h5 chunks per batch.  batch size will be chunk_size * n_per")
    parser.add_argument("--L", default="128", help="deprecated...")
    parser.add_argument("--n_steps", default="3000",
                        help="number of steps per epoch (if -1 all training examples are run each epoch)")
    parser.add_argument("--label_smoothing", default="0.0",
                        help="whether to use label smoothing in classification tasks.  if non zero")

    # data parameter
    parser.add_argument("--in_dim", default="4",
                        help="number of input dimensions")
    parser.add_argument("--n_classes", default="3",
                        help="number of output dimensions of the network")
    parser.add_argument("--regression", action="store_true",
                        help="specifies that were doing regression of a vector or scalar rather than logistic scores.  important for specifying the right loss function")
    parser.add_argument("--classes", default="ab,ba,none",
                        help="class labels if doing classification")
    parser.add_argument("--y_ix", default="None",
                        help="for regression.  if predicting a single scalar, its the desired index of the y vectors saved to the h5 file")
    parser.add_argument("--log_y", action="store_true",
                        help="for regression. whether the dataloader should return log scaled values of the y variables")

    # hyper-parameters
    parser.add_argument("--model", default="gru",
                        help="gru | conv. Type of architecture to use, specifying the type of sequential downsampling or processing employed (gated recurrent or convolutional). we recommend the GRU")
    parser.add_argument("--hidden_dim", default="128", help="for gru.")
    parser.add_argument("--n_gru_layers", default="1",
                        help="for gru.  the number of gru layers to use")
    parser.add_argument("--n_gcn_iter", default="6",
                        help="the number of gcn convolutional layers used")
    parser.add_argument("--gcn_dim", default="26",
                        help="the output dimension of the gcn layers")
    parser.add_argument("--n_conv", default="4",
                        help="for conv. number of 1d convolution layers in each block")

    parser.add_argument("--weights", default="None",
                        help="pre-trained weights to load to resume training or fine tune a model")
    parser.add_argument("--weight_decay", default="0.0",
                        help="weight decay for Adam optimizer. see https://pytorch.org/docs/stable/generated/torch.optim.Adam.html")
    parser.add_argument("--momenta_dir", default="None", help="deprecated...")
    parser.add_argument("--save_momenta_every",
                        default="250", help="deprecated...")

    parser.add_argument("--means", default="None",
                        help="prewritten mean-std values for the inputs and ouputs (in the case of regression)")
    parser.add_argument("--n_val_steps", default="-1",
                        help="in the case you want to validate on a smaller set than the one written")
    parser.add_argument("--skip_info", action = "store_true", help = "skip the graph level features passed the GRU")
    parser.add_argument("--skip_global", action = "store_true", help = "skip the sequence level features passed to the GRU")
    parser.add_argument("--skip_gcn", action = "store_true")
    parser.add_argument("--loss", default = "smooth_l1", help = "smooth_l1 | l1 | l2.  specifies the loss for regression problems")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if args.odir != "None":
        if not os.path.exists(args.odir):
            os.system('mkdir -p {}'.format(args.odir))
            logging.info('root: made output directory {0}'.format(args.odir))

    return args


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using " + str(device) + " as device")
    # model = Classifier(config)

    L = int(args.L)
    if args.y_ix == "None":
        y_ix = None
    else:
        y_ix = int(args.y_ix)

    logging.info(args)

    # write the training setting etc to a txt file
    io_file = open(os.path.join(args.odir, 'info.txt'), 'w')
    io_file.write(
        "Starting training at: {:%B %d, %Y}\n".format(datetime.now()))
    io_file.write(str(args) + '\n')

    generator = TreeSeqGeneratorV3(h5py.File(args.ifile, 'r'), means=args.means, n_samples_per=int(args.n_per_batch), regression=args.regression,
                                   models=args.classes, y_ix=y_ix, log_y=args.log_y)
    validation_generator = TreeSeqGeneratorV3(h5py.File(args.ifile_val, 'r'), means=args.means, n_samples_per=int(args.n_per_batch), regression=args.regression,
                                              models=args.classes, y_ix=y_ix, log_y=args.log_y)

    batch, x1, x2, y = generator[0]
    generator.on_epoch_end()

    bs, n_nodes, n_features = to_dense_batch(batch.x, batch.batch)[0].shape

    if args.model == 'gru':
        model = GATSeqClassifier(generator.batch_size, n_classes=int(args.n_classes), L=L,
                                 n_gcn_iter=int(args.n_gcn_iter), in_dim=int(args.in_dim), hidden_size=int(args.hidden_dim),
                                 use_conv=False, num_gru_layers=int(args.n_gru_layers), gcn_dim=int(args.gcn_dim), 
                                 skip_info = args.skip_info, skip_global = args.skip_global, skip_gcn = args.skip_gcn)
    elif args.model == 'conv':
        model = GATConvClassifier(generator.batch_size, n_classes=int(args.n_classes), L=L, n_nodes=n_nodes,
                                  n_gcn_iter=int(args.n_gcn_iter), in_dim=int(args.in_dim), hidden_size=int(args.hidden_dim),
                                  gcn_dim=int(args.gcn_dim), conv_dim=int(args.conv_dim))

    classes = generator.models

    if not args.regression:
        logging.info('have classes: {}'.format(classes))

    model = model.to(device)
    logging.info(model)
    io_file.write(str(model))

    io_file.write(str(count_parameters(model)))
    io_file.close()

    if args.weights != "None":
        checkpoint = torch.load(args.weights, map_location=device)
        model.load_state_dict(checkpoint)

    # momenta stuff
    save_momenta_every = int(args.save_momenta_every)
    momenta_count = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=float(
        args.lr), weight_decay=float(args.weight_decay))

    # for writing the training
    result = dict()
    result['epoch'] = []
    result['loss'] = []
    result['acc'] = []
    result['val_loss'] = []
    result['val_acc'] = []
    result['time'] = []

    losses = deque(maxlen=500)
    accuracies = deque(maxlen=500)

    if int(args.n_classes) > 1 and not args.regression:
        loss_str = 'nll loss'
        criterion = LabelSmoothing(float(args.label_smoothing))
        classification = True
    else:
        if args.loss == 'smooth_l1':
            loss_str = 'l1 loss'
            criterion = nn.SmoothL1Loss()
        elif args.loss == 'l1':
            loss_str = 'l1'
            criterion = nn.L1Loss()
        elif args.loss == 'l2':
            loss_str = 'mse'
            criterion = nn.MSELoss()
        classification = False

    # print(criterion)
    if args.lr_decay != "None":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, float(args.lr_decay))
    else:
        lr_scheduler = None

    min_val_loss = np.inf

    if int(args.n_steps) > 0:
        n_steps = min([int(args.n_steps), len(generator)])
    else:
        n_steps = len(generator)

    for epoch in range(int(args.n_epochs)):
        t0 = time.time()

        model.train()

        for j in range(int(args.n_steps)):
            batch, x1, x2, y = generator[j]

            if batch is None:
                break

            #print(batch.edge_index.shape, batch.x.shape, batch.edge_index.max())
            batch = batch.to(device)
            y = y.to(device)
            x1 = x1.to(device)
            x2 = x2.to(device)

            optimizer.zero_grad()

            y_pred = model(batch.x, batch.edge_index, batch, x1, x2)

            loss = criterion(y_pred, y)

            if classification:
                y_pred = y_pred.detach().cpu().numpy()
                y_pred = np.argmax(y_pred, axis=1)
                y = y.detach().cpu().numpy()

                accuracies.append(accuracy_score(y, y_pred))

            losses.append(loss.detach().item())

            loss.backward()

            if args.momenta_dir != "None":
                ret = dict()
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        ret[name] = param.grad.detach().cpu().numpy()
                model.update_momenta(ret)

                if (j + 1) % save_momenta_every == 0:
                    np.savez(os.path.join(args.momenta_dir, '{0:06d}.npz'.format(
                        momenta_count)), **model.momenta)
                    momenta_count += 1

            optimizer.step()

            # change back to 100
            if (j + 1) % 25 == 0:
                logging.info("root: Epoch: {}/{}, Step: {}/{}, Loss: {}, Acc: {}".format(epoch+1,
                                                                                         args.n_epochs, j + 1, n_steps,
                                                                                         np.mean(losses), np.mean(accuracies)))

        generator.on_epoch_end()

        train_loss = np.mean(losses)
        train_acc = np.mean(accuracies)

        val_losses = []
        val_accs = []

        logging.info('validating...')
        logging.info('have {} validation steps...'.format(
            len(validation_generator)))
        model.eval()

        Y = []
        Y_pred = []
        with torch.no_grad():
            for j in range(len(validation_generator)):
                batch, x1, x2, y = validation_generator[j]

                if batch is None:
                    break

                batch = batch.to(device)
                y = y.to(device)
                x1 = x1.to(device)
                x2 = x2.to(device)

                y_pred = model(batch.x, batch.edge_index, batch, x1, x2)

                loss = criterion(y_pred, y)

                if classification:
                    y_pred = y_pred.detach().cpu().numpy()
                    y = y.detach().cpu().numpy().flatten()

                    y_pred = np.argmax(y_pred, axis=1)
                    val_accs.append(accuracy_score(y, y_pred))
                else:
                    y_pred = y_pred.detach().cpu().numpy()
                    y = y.detach().cpu().numpy()

                Y.extend(y)
                Y_pred.extend(y_pred)

                val_losses.append(loss.detach().item())

        val_loss = np.mean(val_losses)
        val_acc = np.mean(val_accs)

        result['epoch'].append(epoch)
        result['val_loss'].append(val_loss)
        result['val_acc'].append(val_acc)
        result['loss'].append(train_loss)
        result['acc'].append(train_acc)
        result['time'].append(time.time() - t0)

        logging.info('root: Epoch {}, Val Loss: {:.3f}, Val Acc: {:.3f}'.format(
            epoch + 1, val_loss, val_acc))

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            print('saving weights...')
            torch.save(model.state_dict(), os.path.join(
                args.odir, 'best.weights'))

            # do this for all the examples:
            if classification:
                cm_analysis(Y, np.round(Y_pred), os.path.join(
                    args.odir, 'confusion_matrix_best.png'), classes)

            early_count = 0
        else:
            early_count += 1

            # early stop criteria
            if early_count > int(args.n_early):
                break

        # re-set the index of the generator without shuffling
        validation_generator.on_epoch_end(False)

        df = pd.DataFrame(result)
        df.to_csv(os.path.join(args.odir, 'metric_history.csv'), index=False)

        if lr_scheduler is not None:
            logging.info('lr for next epoch: {}'.format(
                lr_scheduler.get_last_lr()))
            lr_scheduler.step()

        plt.rc('font', family='Helvetica', size=12)
        plt.rcParams.update({'figure.autolayout': True})
        fig = plt.figure(figsize=(12, 8), dpi=100)

        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('epoch')

        ax.set_ylabel(loss_str)
        ax.set_title('training loss history')

        ax.plot(result['epoch'], result['loss'], label='training')
        ax.plot(result['epoch'], result['val_loss'], label='validation')
        ax.legend()
        plt.savefig(os.path.join(args.odir, 'training_loss.png'), dpi=100)
        plt.close()


if __name__ == "__main__":
    main()
