# -*- coding: utf-8 -*-
import os
import argparse
import logging

import h5py
import numpy as np

import sys
sys.path.insert(0, './src/models')

from data_loaders import TreeSeqGenerator
from torch_geometric.utils import to_dense_batch, unbatch_edge_index

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--ofile", default = "None")
    parser.add_argument("--classes", default = "hard,hard-near,neutral,soft,soft-near")
    
    parser.add_argument("--L", default = "128")
    parser.add_argument("--n_sample_iter", default = "3") # number of times to sample sequences > L
    parser.add_argument("--chunk_size", default = "5")
    
    parser.add_argument("--n_train", default = "100000")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()

    ifiles = sorted([os.path.join(args.idir, u) for u in os.listdir(args.idir) if u.split('.')[-1] == 'hdf5'])
    counts = dict()
    
    classes = args.classes.split(',')
    
    data = dict()
    for c in classes:
        data[c] = dict()
        data[c]['x'] = []
        data[c]['x1'] = []
        data[c]['edge_index'] = []
        
    
    ofile = h5py.File(args.ofile, 'w')
    ofile_val = h5py.File('/'.join(args.ofile.split('/')[:-1]) + '/' + args.ofile.split('/')[-1].split('.')[0] + '_val.hdf5', 'w')
    
    L = int(args.L)
    
    counter = 0

    for ifile in ifiles:
        generator = TreeSeqGenerator(h5py.File(ifile, 'r'), n_samples_per = 1, sequence_length = L, pad = args.pad_l)
        val = '_val' in ifile        
        
        for j in range(len(generator)):
            batch, y, x1, bl = generator[j]
            
            if batch is None:
                break
            
            x1 = x1.detach().cpu().numpy()
            x = batch.x
            edge_index = batch.edge_index
            y = list(y.detach().cpu().numpy().flatten())
            
            classes_ = [classes[u] for u in y]
            
            x = to_dense_batch(x, bl)
            #edge_index = unbatch_edge_index(edge_index, bl)
            
            print(x.shape)
            print(edge_index.shape)
            sys.exit()
            
            
            
            
            
    ofile.close()
    ofile_val.close()
        
        
        
        
    # ${code_blocks}

if __name__ == '__main__':
    main()


