# -*- coding: utf-8 -*-
import argparse
from model_viz import cm_analysis, count_parameters

import torch
import torch.nn.functional as F
import h5py
import configparser
from data_loaders import GenomatGenerator, GenomatClassGenerator
#from gcn import GCN, Classifier, SequenceClassifier
import torch.nn as nn

from torch.nn import CrossEntropyLoss, NLLLoss, DataParallel, BCEWithLogitsLoss
from collections import deque

from sklearn.metrics import accuracy_score

import logging, os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time

from train_gcn import LabelSmoothing
from torchvision_mod_layers import resnet34, LexStyleNet

from datetime import datetime

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

"""
Trains a CNN on either a regression or classification task with data already formatted as an h5 file.
 
"""

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "None", help = "data from src/data/format_genomat.  has the genotype matrices for loading during training")
    parser.add_argument("--ifile_val", default = "None", help = "same as ifile but for validation")
    
    parser.add_argument("--in_channels", default = "1", help = "number of population channels in the image")
    parser.add_argument("--n_classes", default = "1", help = "number of classes to predict or the dimension of the y variable")
    parser.add_argument("--y_ix", default = "None")
    parser.add_argument("--log_y", action = "store_true", help = "whether the y variables should be in log space.  only applicable to regression tasks")
    parser.add_argument("--means", default = "None", help = "y means and stds for normalization.  only applicable to regression tasks")
    
    
    parser.add_argument("--n_epochs", default = "100")
    parser.add_argument("--lr", default = "0.001")
    parser.add_argument("--weight_decay", default = "0.0")
    parser.add_argument("--label_smoothing", default = "0.0")
    
    parser.add_argument("--n_steps", default = "None")
    parser.add_argument("--n_early", default = "10")
    
    parser.add_argument("--model", default = "res")
    
    parser.add_argument("--batch_size", default = "32")
    
    parser.add_argument("--weights", default = "None")
    
    parser.add_argument("--regression", action = "store_true")

    parser.add_argument("--odir", default = "None")
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
    # ${odir_del_block}

    return args

def main():
    args = parse_args()
    
    # write the training setting etc to a txt file
    io_file = open(os.path.join(args.odir, 'info.txt'), 'w')
    io_file.write("Starting training at: {:%B %d, %Y}\n".format(datetime.now()))
    io_file.write(str(args) + '\n')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using " + str(device) + " as device")
    # model = Classifier(config)
    
    logging.info('reading data keys...')
    if args.y_ix == "None":
        y_ix = None
    else:
        y_ix = int(args.y_ix)
    
    if args.regression:
        generator = GenomatGenerator(args.ifile, args.means, y_ix = y_ix, batch_size = int(args.batch_size) // 4, log_y = args.log_y)
        generator_val = GenomatGenerator(args.ifile_val, args.means, y_ix = y_ix, batch_size = int(args.batch_size) // 4, log_y = args.log_y)
    else:
        generator = GenomatClassGenerator(args.ifile, batch_size = int(args.batch_size))
        generator_val = GenomatClassGenerator(args.ifile_val, batch_size = int(args.batch_size))
        
        classes = generator.classes
        
    x, y = generator[0]
    logging.info('input shape {}...'.format(x.shape))
            
    logging.info('making model...')
    if args.model == "res":
        model = resnet34(in_channels = int(args.in_channels), num_classes = int(args.n_classes)).to(device)
    elif args.model == "lex":
        _, c, n_pop, n_sites = x.shape
        model = LexStyleNet(h = c * n_pop, n_classes = int(args.n_classes)).to(device)
    
    logging.info(model)
    io_file.write(str(model))
    
    io_file.write(str(count_parameters(model)))
    io_file.close()

    if args.weights != "None":
        checkpoint = torch.load(args.weights, map_location = device)
        model.load_state_dict(checkpoint)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay = float(args.weight_decay))
    
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
        criterion = LabelSmoothing(float(args.label_smoothing))
        classification = True
    else:
        criterion = nn.SmoothL1Loss()
        classification = False
        
    min_val_loss = np.inf
    
    if args.n_steps != "None":
        n_steps = min([len(generator), int(args.n_steps)])
    else:
        n_steps = len(generator)
    
    for epoch in range(int(args.n_epochs)):
        t0 = time.time()
        
        model.train()

        for j in range(n_steps):
            x, y = generator[j]
            
            if x is None:
                break
            
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()

            y_pred = model(x)
            loss = criterion(y_pred, y)

            if classification:
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
        
        model.eval()
        
        generator.on_epoch_end()
        
        train_loss = np.mean(losses)
        train_acc = np.mean(accuracies)
        
        val_losses = []
        val_accs = []

        logging.info('validating...')
        logging.info('have {} validation steps...'.format(len(generator_val)))
        model.eval()
        
        Y = []
        Y_pred = []
        with torch.no_grad():
            for j in range(len(generator_val)):
                x, y = generator_val[j]
                
                if x is None:
                    break
                
                x = x.to(device)
                y = y.to(device)
                
                y_pred = model(x)
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
                
        generator_val.on_epoch_end()        
        
        val_loss = np.mean(val_losses)
        val_acc = np.mean(val_accs)
        
        result['epoch'].append(epoch)
        result['val_loss'].append(val_loss)
        result['val_acc'].append(val_acc)
        result['loss'].append(train_loss)
        result['acc'].append(train_acc)
        result['time'].append(time.time() - t0)
        
        logging.info('root: Epoch {}, Val Loss: {:.3f}, Val Acc: {:.3f}'.format(epoch + 1, val_loss, val_acc))
        
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            print('saving weights...')
            torch.save(model.state_dict(), os.path.join(args.odir, 'best.weights'))
            
            # do this for all the examples:
            if classification:
                cm_analysis(Y, np.round(Y_pred), os.path.join(args.odir, 'confusion_matrix_best.png'), classes)
            
            
            early_count = 0
        else:
            early_count += 1

            # early stop criteria
            if early_count > int(args.n_early):
                break
    
        df = pd.DataFrame(result)
        df.to_csv(os.path.join(args.odir, 'metric_history.csv'), index = False)

    # ${code_blocks}

if __name__ == '__main__':
    main()


