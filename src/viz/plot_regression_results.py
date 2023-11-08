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
from data_loaders import TreeSeqGenerator, TreeSeqGeneratorV2
#from gcn import GCN, Classifier, SequenceClassifier
import torch.nn as nn
from gcn_layers import GATSeqClassifier, GATConvClassifier

from torch.nn import CrossEntropyLoss, NLLLoss, DataParallel, BCEWithLogitsLoss
from collections import deque

from sklearn.metrics import accuracy_score

import logging, os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--weights", default = "None")

    parser.add_argument("--n_per_batch", default = "16")
    parser.add_argument("--L", default = "128", help = "tree sequence length")
    parser.add_argument("--n_samples", default = "50", help = "number of present day individuals")
    parser.add_argument("--n_steps", default = "3000", help = "number of steps per epoch (if -1 all training examples are run each epoch)")
    
    # data parameter
    parser.add_argument("--in_dim", default = "4")
    parser.add_argument("--n_classes", default = "5")
    
    # hyper-parameters
    parser.add_argument("--use_conv", action = "store_true")
    parser.add_argument("--hidden_dim", default = "128")
    parser.add_argument("--n_gru_layers", default = "1")
    parser.add_argument("--n_gcn_iter", default = "6")
    parser.add_argument("--gcn_dim", default = "26")
    parser.add_argument("--conv_dim", default = "4")

    parser.add_argument("--chunk_size", default = "4")

    parser.add_argument("--means", default = "None")
    parser.add_argument("--model", default = "gru")

    parser.add_argument("--odir", default = "None")
    parser.add_argument("--ofile", default = "results.npz")
    
    parser.add_argument("--log_y", action = "store_true")
    parser.add_argument("--y_ix", default = "None")
    
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
    n_nodes = int(args.n_samples) * 2 - 1
    
    if args.y_ix == "None":
        y_ix = None
    else:
        y_ix = int(args.y_ix)

    generator = TreeSeqGeneratorV2(h5py.File(args.ifile, 'r'), means = args.means, n_samples_per = int(args.n_per_batch), regression = True, 
                                              chunk_size = int(args.chunk_size), y_ix = y_ix, log_y = args.log_y)
    
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
    
    classification = False
    
    
    ix = 0
    while True:
        with torch.no_grad():
            batch, x1, x2, y = generator[ix]
            
            if batch is None:
                break
            
            batch = batch.to(device)
            y = y.to(device)
            x1 = x1.to(device)
            x2 = x2.to(device)

            y_pred = model(batch.x, batch.edge_index, batch.batch, x1, x2)
            
            y_pred = y_pred.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            
            Y.extend(y)
            Y_pred.extend(y_pred)
            
        ix += 1
            
    Y = (np.array(Y) * generator.y_std + generator.y_mean)
    Y_pred = (np.array(Y_pred) * generator.y_std + generator.y_mean)

    print(Y.shape, Y_pred.shape)
    rmse = np.sqrt(np.mean((Y - Y_pred)**2, axis = 0))
    print(rmse)
    print(np.median(rmse))

    np.savez(args.ofile, y = Y, y_pred = Y_pred)

    """
    print(np.mean((Y - Y_pred)**2, axis = 0))

    plot, axes = plt.subplots(1, 5)
    plot.set_size_inches(25, 5)
        
    for ix in range(Y.shape[1]):
        axes[ix].plot([np.min(Y[:,ix]), np.min(Y[:,ix])], [np.max(Y[:,ix]), np.max(Y[:,ix])])
        axes[ix].scatter(Y[:,ix], Y_pred[:,ix], alpha = 0.7)
        print([np.min(Y[:,ix]), np.min(Y[:,ix])], [np.max(Y[:,ix]), np.max(Y[:,ix])])
        
        
    plt.tight_layout()
    plt.savefig('demo_results.png', dpi = 100)
    plt.close()
    """
    # ${code_blocks}

if __name__ == '__main__':
    main()
