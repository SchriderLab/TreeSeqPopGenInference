# -*- coding: utf-8 -*-
import os
import argparse
import logging

import sys
sys.path.append('src/models')

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

from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

import logging, os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}
from scipy.special import softmax

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--weights", default = "None")

    parser.add_argument("--n_per_batch", default = "4")
    parser.add_argument("--L", default = "128", help = "tree sequence length")
    parser.add_argument("--n_steps", default = "3000", help = "number of steps per epoch (if -1 all training examples are run each epoch)")
    
    # data parameter
    parser.add_argument("--in_dim", default = "4")
    parser.add_argument("--n_classes", default = "5")
    parser.add_argument("--n_samples", default = "34")
    
    # hyper-parameters
    parser.add_argument("--use_conv", action = "store_true")
    parser.add_argument("--hidden_dim", default = "128")
    parser.add_argument("--n_gru_layers", default = "1")
    parser.add_argument("--n_gcn_iter", default = "6")
    parser.add_argument("--gcn_dim", default = "26")
    parser.add_argument("--conv_dim", default = "4")
    parser.add_argument("--classes", default = "hard,hard-near,neutral,soft,soft-near")

    parser.add_argument("--chunk_size", default = "4")

    parser.add_argument("--means", default = "None")
    parser.add_argument("--model", default = "gru")

    parser.add_argument("--odir", default = "None")
    parser.add_argument("--return_params", action = "store_true")
    
    parser.add_argument("--ofile", default = "result.csv")
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using " + str(device) + " as device")
    # model = Classifier(config)

    L = int(args.L)

    generator = TreeSeqGeneratorV3(h5py.File(args.ifile, 'r'), means = args.means, n_samples_per = int(args.n_per_batch), regression = False, 
                                              models = args.classes, return_params = args.return_params)
    
    n_nodes = int(args.n_samples) * 2 - 1
    
    classes = generator.models
    args.n_classes = len(classes)
    
    if args.model == 'gru':
        model = GATSeqClassifier(n_nodes, n_classes = int(args.n_classes), L = L, 
                             n_gcn_iter = int(args.n_gcn_iter), in_dim = int(args.in_dim), hidden_size = int(args.hidden_dim),
                             use_conv = args.use_conv, num_gru_layers = int(args.n_gru_layers), gcn_dim = int(args.gcn_dim))
    elif args.model == 'conv':
        model = GATConvClassifier(n_nodes, n_classes = int(args.n_classes), L = L, 
                             n_gcn_iter = int(args.n_gcn_iter), in_dim = int(args.in_dim), hidden_size = int(args.hidden_dim),
                             gcn_dim = int(args.gcn_dim), conv_dim = int(args.conv_dim))
    
    

    model = model.to(device)
    checkpoint = torch.load(args.weights, map_location = device)
    model.load_state_dict(checkpoint)

    model.eval()
    
    Y = []
    Y_pred = []
    params = []

    accs = []
    
    ix = 0
    while True:
        with torch.no_grad():
            if not args.return_params:
                batch, x1, x2, y = generator[ix]
            else:
                batch, x1, x2, y, params_ = generator[ix]                
                if params_ is not None:
                    params.extend(params_)
                
            if batch is None:
                break
            
            t0 = time.time()
            batch = batch.to(device)
            y = y.to(device)
            x1 = x1.to(device)
            x2 = x2.to(device)

            y_pred = model(batch.x, batch.edge_index, batch, x1, x2)
            logging.debug('took {} s to forward...'.format(time.time() - t0))
            
            y_pred = y_pred.detach().cpu().numpy()
            y = y.detach().cpu().numpy().flatten()
        
            accs.append(accuracy_score(y, np.argmax(y_pred, axis=1)))
    
            Y.extend(y)
            Y_pred.extend(softmax(y_pred, axis = -1))
            
    
        ix += 1
    
    Y = np.array(Y)
    Y_pred = np.array(Y_pred)
    params = np.array(params)
    
    result = dict()
    
    for c in classes:
        result[c] = []
   
    result['y'] = Y
    if args.return_params:
        for ix in range(params.shape[1]):
            result['param_{0:02d}'.format(ix)] = params[:,ix]
    
    for ix, c in enumerate(classes):    
        result[c] = Y_pred[:,ix]
        
    df = pd.DataFrame(result)
    df.to_csv(args.ofile, index = False)
    
    logging.info('have shape of predictions: {}'.format(Y_pred.shape))
    logging.info('mean accuracy score: {}'.format(np.mean(accs)))
    
    """
    Yh = np.zeros((len(Y), len(classes)))    
    Yh[range(len(Y)),Y] = 1.
    
    roc = roc_auc_score(Yh, Y_pred)
    aupr = average_precision_score(Yh, Y_pred)

    print('mean accuracy: {}'.format(np.mean(accs)))
    print('roc: {}'.format(roc))
    print('aupr: {}'.format(aupr))

    cm_analysis(Y, np.argmax(Y_pred, axis=1), os.path.join(args.odir, 'confusion_matrix_best.png'), classes)
    """
    
if __name__ == '__main__':
    main()