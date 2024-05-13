# -*- coding: utf-8 -*-
import os
import argparse
import logging

import h5py
import sys
sys.path.insert(0, './src/models')

import torch
from gcn_layers import GATSeqClassifier, GATConvClassifier
from data_loaders import TreeSeqGenerator
import glob

from torch_geometric.data import Data, Batch, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def get_params(cmd):
    cmd = cmd.split()
    theta_factor = float(cmd[5]) / 220.
    rho_factor = float(cmd[8]) / 1008.33
    
    return theta_factor, rho_factor

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--i", default = "None")

    parser.add_argument("--weights", default = "None")
    parser.add_argument("--means", default = "None")

    parser.add_argument("--model", default = "gru")
    
    # data parameter
    parser.add_argument("--in_dim", default = "4")
    parser.add_argument("--n_classes", default = "5")
    parser.add_argument("--n_samples", default = "104")
    
    # hyper-parameters
    parser.add_argument("--use_conv", action = "store_true")
    parser.add_argument("--hidden_dim", default = "128")
    parser.add_argument("--n_gru_layers", default = "1")
    parser.add_argument("--n_gcn_iter", default = "6")
    parser.add_argument("--gcn_dim", default = "26")
    parser.add_argument("--conv_dim", default = "4")
    parser.add_argument("--classes", default = "hard,hard-near,neutral,soft,soft-near")
    
    parser.add_argument("--L", default = "128")
    parser.add_argument("--odir", default = "None")
    parser.add_argument("--sampling_mode", default = "none")
    
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
    
    L = int(args.L)
    n_nodes = int(args.n_samples) * 2 - 1
    if '.hdf5' in args.i:
        ifiles = [args.i]
    else:
        ifiles = glob.glob(os.path.join(args.i, '*/*.hdf5')) + glob.glob(os.path.join(args.i, '*.hdf5'))
    
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
    
    classes = list(args.classes.split(','))
    
    for ifile in ifiles:
        logging.info('working on {}...'.format(ifile))
        generator = TreeSeqGenerator(h5py.File(ifile, 'r'), n_samples_per = 1, means = args.means, sequence_length = L, pad = True, categorical = True)
        
        hfile = h5py.File(ifile, 'r')
        keys = list(hfile.keys())
        
        Y_pred = []
        Y = []
        params = []
        for key in keys:
            skeys = hfile[key].keys()
            # each is a tree seq
            
            for skey in skeys:
                x, x1, edge_index, mask, x2, y = generator.get_seq(key, skey, args.sampling_mode, normalize = True)
                
                print(x.shape, x1.shape, edge_index.shape, x2.shape)
                
                Y.append(classes.index(key))
                cmd = hfile[key][skey].attrs['cmd']
                
                params.append(get_params(cmd))
                
                # we have to remove the root node edge
                _ = []
                for k in range(edge_index.shape[0]):
                    e = edge_index[k]
                    ii = np.where(e >= 0)[0]
                
                    _.append(torch.LongTensor(e[:, ii]))
    
                # use PyTorch Geometrics batch object to make one big graph
                batch = Batch.from_data_list(
                    [
                        Data(x=torch.FloatTensor(x[k]), edge_index=torch.LongTensor(_[k])) for k in range(x.shape[0])
                    ]
                )
                ii = 0
                
                batch_indices = []
                l = x.shape[0]
                batch_indices.append(torch.LongTensor(np.array(range(ii, ii + l))))
                ii += l
                batch.batch_indices = batch_indices
                
                x1 = torch.FloatTensor(x1).to(device)
                x2 = torch.FloatTensor(x2).to(device).unsqueeze(0)
                batch = batch.to(device)
                
                y_pred = model(batch.x, batch.edge_index, batch, x1, x2)
                
                y_pred = y_pred.detach().cpu().numpy()

                Y_pred.append(y_pred[0])
    
        Y_pred = np.array(Y_pred)
        Y = np.array(Y)
        
        print(accuracy_score(Y, np.argmax(Y_pred, axis = -1)))
        
        params = np.array(params)
        
        print(Y_pred.shape)
        np.savez(ifile.replace('.hdf5', '.npz'), y_pred = Y_pred, y = Y, params = params)
    
    # ${code_blocks}

if __name__ == '__main__':
    main()
