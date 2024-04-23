# -*- coding: utf-8 -*-
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
from data_loaders import TreeSeqGenerator, TreeSeqGeneratorV2, GenomatClassGenerator
#from gcn import GCN, Classifier, SequenceClassifier
import torch.nn as nn
from torchvision_mod_layers import resnet34

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
    
    parser.add_argument("--in_channels", default = "1")
    
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
    
    logging.info('reading data keys...')
    generator = GenomatClassGenerator(args.ifile, batch_size = 1)
    
    classes = generator.classes
    args.n_classes = len(classes)
    
    model = resnet34(in_channels = int(args.in_channels), num_classes = int(args.n_classes)).to(device)
        
    model = model.to(device)
    checkpoint = torch.load(args.weights, map_location = device)
    model.load_state_dict(checkpoint)

    model.eval()
    
    Y = []
    Y_pred = []

    
    accs = []
    
    ix = 0
    while True:
        with torch.no_grad():
            t0 = time.time()
            x, y = generator[ix]
            
            if x is None:
                break
            
            x = x.to(device)
            y = y.to(device)
            
            y_pred = model(x)
            
            logging.debug('took {} s to forward...'.format(time.time() - t0))
            
            y_pred = y_pred.detach().cpu().numpy()
            y = y.detach().cpu().numpy().flatten()
        
            accs.append(accuracy_score(y, np.argmax(y_pred, axis=1)))
    
            Y.extend(y)
            Y_pred.extend(softmax(y_pred, axis = -1))
            
        ix += 1
                
    Y = np.array(Y)
    Y_pred = np.array(Y_pred)
    
    result = dict()
    
    for c in classes:
        result[c] = []
   
    result['y'] = Y
    
    for ix, c in enumerate(classes):    
        result[c] = Y_pred[:,ix]
        
    df = pd.DataFrame(result)
    df.to_csv(args.ofile, index = False)
    
    logging.info('have shape of predictions: {}'.format(Y_pred.shape))
    logging.info('mean accuracy score: {}'.format(np.mean(accs)))
    
if __name__ == '__main__':
    main()
        
    
    
    
