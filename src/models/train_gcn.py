import argparse
from model_viz import cm_analysis, count_parameters

import torch
import torch.nn.functional as F
import h5py
import configparser
from data_loaders import TreeSeqGenerator
#from gcn import GCN, Classifier, SequenceClassifier
import torch.nn as nn
from gcn_layers import GATSeqClassifier

from torch.nn import CrossEntropyLoss, NLLLoss, DataParallel, BCEWithLogitsLoss
from collections import deque

from sklearn.metrics import accuracy_score

import logging, os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--ifile", default="None", help = "training h5 file")
    parser.add_argument("--ifile_val", default="None", help = "validation h5 file")
    parser.add_argument("--odir", default="None")
    
    parser.add_argument("--n_epochs", default="100")
    parser.add_argument("--lr", default="0.00001")
    parser.add_argument("--n_early", default = "10")
    parser.add_argument("--lr_decay", default = "None")
    
    parser.add_argument("--n_per_batch", default = "16")
    parser.add_argument("--L", default = "32", help = "tree sequence length")
    parser.add_argument("--n_steps", default = "3000", help = "number of steps per epoch (if -1 all training examples are run each epoch)")
    parser.add_argument("--n_gcn_iter", default = "6")
    parser.add_argument("--in_dim", default = "6")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if args.odir != "None":
        if not os.path.exists(args.odir):
            os.system('mkdir -p {}'.format(args.odir))
            logging.debug('root: made output directory {0}'.format(args.odir))
        else:
            os.system('rm -rf {0}'.format(os.path.join(args.odir, '*')))

    return args


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using " + str(device) + " as device")
    # model = Classifier(config)

    L = int(args.L)

    generator = TreeSeqGenerator(h5py.File(args.ifile, 'r'), sequence_length = L)
    validation_generator = TreeSeqGenerator(h5py.File(args.ifile_val, 'r'), sequence_length = L)
    model = GATSeqClassifier(L = L, n_gcn_iter = int(args.n_gcn_iter), in_dim = int(args.in_dim))
    
    classes = generator.models
    
    model = model.to(device)
    print(model)
    count_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    
    # for writing the training 
    result = dict()
    result['epoch'] = []
    result['loss'] = []
    result['acc'] = []
    result['val_loss'] = []
    result['val_acc'] = []

    losses = deque(maxlen=500)
    accuracies = deque(maxlen=500)
    criterion = NLLLoss()
    
    if args.lr_decay != "None":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, float(args.lr_decay))
    else:
        lr_scheduler = None
        
    min_val_loss = np.inf
    
    for epoch in range(int(args.n_epochs)):
        model.train()
        
        n_steps = int(args.n_steps)
        for j in range(int(args.n_steps)):
            batch, y, x1, bl = generator[j]
            
            if batch is None:
                break
            
            #print(batch.edge_index.shape, batch.x.shape, batch.edge_index.max())
            batch = batch.to(device)
            y = y.to(device)
            x1 = x1.to(device)

            optimizer.zero_grad()

            y_pred = model(batch.x, batch.edge_index, batch.batch, x1, bl)

            loss = criterion(y_pred, y)

            y_pred = y_pred.detach().cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)
            y = y.detach().cpu().numpy()

            accuracies.append(accuracy_score(y, y_pred))

            losses.append(loss.detach().item())

            loss.backward()
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
        model.eval()
        
        Y = []
        Y_pred = []
        with torch.no_grad():
            for j in range(len(validation_generator)):
                batch, y, x1, bl = validation_generator[j]
                
                if batch is None:
                    break
                
                batch = batch.to(device)
                y = y.to(device)
                x1 = x1.to(device)

                y_pred = model(batch.x, batch.edge_index, batch.batch, x1, bl)

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
        
        result['epoch'].append(epoch)
        result['val_loss'].append(val_loss)
        result['val_acc'].append(val_acc)
        result['loss'].append(train_loss)
        result['acc'].append(train_acc)
        
        logging.info('root: Epoch {}, Val Loss: {:.3f}, Val Acc: {:.3f}'.format(epoch + 1, val_loss, val_acc))
        
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            print('saving weights...')
            torch.save(model.state_dict(), os.path.join(args.odir, 'best.weights'))
            
            # do this for all the examples:
            cm_analysis(Y, np.round(Y_pred), os.path.join(args.odir, 'confusion_matrix_best.png'), classes)

            early_count = 0
        else:
            early_count += 1

            # early stop criteria
            if early_count > int(args.n_early):
                break
        
        validation_generator.on_epoch_end()
    
        df = pd.DataFrame(result)
        df.to_csv(os.path.join(args.odir, 'metric_history.csv'), index = False)
        
        if lr_scheduler is not None:
            logging.info('lr for next epoch: {}'.format(lr_scheduler.get_lr()))
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