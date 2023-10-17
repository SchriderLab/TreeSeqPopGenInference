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
from data_loaders import TreeSeqGenerator, TreeSeqGeneratorV2, GenomatGenerator
#from gcn import GCN, Classifier, SequenceClassifier
import torch.nn as nn
from gcn_layers import GATSeqClassifier, GATConvClassifier

from torchvision_mod_layers import resnet34

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
    parser.add_argument("--in_channels", default = "1")

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
    
    if args.y_ix == "None":
        y_ix = None
    else:
        y_ix = int(args.y_ix)
    
    logging.info('reading data keys...')
    
    model = resnet34(in_channels = int(args.in_channels), num_classes = int(args.n_classes)).to(device)
        
    model = model.to(device)
    checkpoint = torch.load(args.weights, map_location = device)
    model.load_state_dict(checkpoint)

    model.eval()
    
    generator = GenomatGenerator(args.ifile, args.means, y_ix = y_ix, batch_size = 32 // 4, log_y = args.log_y)
    
    Y = []
    Y_pred = []
    
    for ix in range(len(generator)):
        with torch.no_grad():
            t0 = time.time()
            x, y = generator[ix]
            
            x = x.to(device)
            y = y.to(device)
            
            y_pred = model(x)
            
            logging.debug('took {} s to forward...'.format(time.time() - t0))
            
            y_pred = y_pred.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
        
    
            Y.extend(y)
            Y_pred.extend(y_pred)

    Y = (np.array(Y) * generator.y_std + generator.y_mean)
    Y_pred = (np.array(Y_pred) * generator.y_std + generator.y_mean)
    print(Y.shape, Y_pred.shape)

    print(np.sqrt(np.mean((Y - Y_pred)**2, axis = 0)))

    np.savez(args.ofile, y = Y, y_pred = Y_pred)
    
if __name__ == '__main__':
    main()
    
