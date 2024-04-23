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

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--i", default = "None")

    parser.add_argument("--weights", default = "None")
    parser.add_argument("--means", default = "None")
    
    parser.add_argument("--n_samples", default = "34")
    parser.add_argument("--odir", default = "None")
    parser.add_argument("--model", default = "gru")
    parser.add_argument("--L", default = "128")
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
    
    for ifile in ifiles:
        logging.info('working on {}...'.format(ifile))
        generator = TreeSeqGenerator(h5py.File(ifile, 'r'), n_samples_per = 1, means = args.means, sequence_length = L, pad = True, categorical = True)
        
        hfile = h5py.File(ifile, 'r')
        keys = list(hfile.keys())
        
        for key in keys:
            skeys = hfile[key].keys()
            # each is a tree seq
            
            for skey in skeys:
                x, x1, edge_index, mask, global_vec, y = generator.get_seq(key, skey, args.sampling_mode)
                
                print(x.shape)
                sys.exit()
                
    
    # ${code_blocks}

if __name__ == '__main__':
    main()