# -*- coding: utf-8 -*-
import os
import argparse
import logging

import h5py
import numpy as np

import sys
sys.path.insert(0, './src/models')

from data_loaders import TreeSeqGenerator
from typing import List

import torch
from torch import Tensor


import matplotlib.pyplot as plt
from collections import deque
import glob
import random

def get_params(cmd):
    cmd = cmd.split()
    theta_factor = float(cmd[5]) / 220.
    rho_factor = float(cmd[8]) / 1008.33
    
    return theta_factor, rho_factor

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--i", default = "None", help = "input file or directory of h5 files to process")
    parser.add_argument("--ofile", default = "None", help = "h5 file to output the results to")
    parser.add_argument("--classes", default = "hard,hard-near,neutral,soft,soft-near")
    
    parser.add_argument("--L", default = "128", help = "max length of tree sequence")
    #parser.add_argument("--n_sample_iter", default = "3") # number of times to sample sequences > L
    parser.add_argument("--chunk_size", default = "5", help = "write replicates to chunks, drastically speeds up read time (only applies to the regression case as chunk size in the classification case is always set to the number of classes)")
    parser.add_argument("--sampling_mode", default = "sequential", help = "sequential | equi, if equi sample trees with replacement based on how much of the chrom they take up")
    
    parser.add_argument("--val_prop", default = "0.1", help = "proportion of data to put in val file which is saved with the suffix _val.hdf5")
    parser.add_argument("--cmd_parser", default = "get_params")
    parser.add_argument("--write_params", action = "store_true")
    
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()
    
    if '.hdf5' in args.i:
        ifiles = [args.i]
    else:
        ifiles = glob.glob(os.path.join(args.i, '*/*.hdf5')) + glob.glob(os.path.join(args.i, '*.hdf5'))
    
    random.shuffle(ifiles)
    counts = dict()
    
    # classification
    if ',' in args.classes:
        classes = args.classes.split(',')
        
        data = dict()
        for c in classes:
            data[c] = dict()
            data[c]['x'] = deque()
            data[c]['x1'] = deque()
            data[c]['edge_index'] = deque()
            data[c]['mask'] = deque()
            data[c]['global_vec'] = deque()
            
            if args.write_params:
                data[c]['params'] = deque()
            
        classification = True
    # regression
    else:
        classification = False
        
        data = dict()
        data['x'] = deque()
        data['x1'] = deque()
        data['edge_index'] = deque()
        data['mask'] = deque()
        data['global_vec'] = deque()
        data['y'] = deque()
        
        if args.write_params:
            data['params'] = deque()
        
    ofile = h5py.File(args.ofile, 'w')
    if float(args.val_prop) > 0:
        ofile_val = h5py.File('/'.join(args.ofile.split('/')[:-1]) + '/' + args.ofile.split('/')[-1].split('.')[0] + '_val.hdf5', 'w')
    
    L = int(args.L)
    
    # counters for dataset labels
    counter = 0
    val_counter = 0
    
    lengths = []
    
    x1_means = []
    x1_stds = []
    bls = []
    n_muts = []
    yl = []
    
    val_prop = float(args.val_prop)
    chunk_size = int(args.chunk_size)
    
    for ifile in ifiles:
        logging.info('working on {}...'.format(ifile))
        generator = TreeSeqGenerator(h5py.File(ifile, 'r'), n_samples_per = 1, sequence_length = L, pad = True, categorical = classification)
        
        hfile = h5py.File(ifile, 'r')
        keys = list(hfile.keys())
        
        for key in keys:
            skeys = hfile[key].keys()
            # each is a tree seq
            
            for skey in skeys:
                x, x1, edge_index, mask, global_vec, y = generator.get_seq(key, skey, args.sampling_mode)
                if args.write_params:
                    cmd = hfile[key][skey].attrs['cmd']
                    
                    params = globals()[args.cmd_parser](cmd)
                    print(params, x.shape, x1.shape)
                
                if classification:
                    c = y
                    
                    data[c]['x'].append(x)
                    data[c]['x1'].append(x1)
                    data[c]['edge_index'].append(edge_index)
                    data[c]['mask'].append(mask)
                    data[c]['global_vec'].append(global_vec)
                
                    if args.write_params:
                        data[c]['params'].append(params)
                else:
                    data['x'].append(x)
                    data['x1'].append(x1)
                    data['edge_index'].append(edge_index)
                    data['mask'].append(mask)
                    data['global_vec'].append(global_vec)
                    data['y'].append(y)
                    
                    if args.write_params:
                        data['params'].append(params)
        
        
            if classification:
                cond = all([len(data[u]['x']) > 0 for u in classes])
            else:
                cond = (len(data['x']) > 0)
            
            while cond:
                X = []
                Ds = []
                edge_index = []
                masks = []
                X1 = []
                global_vec = []
                y = []
                params = []
                
                if classification:
                    for c in classes:
                        X.append(data[c]['x'].pop())
                        edge_index.append(data[c]['edge_index'].pop())
                        X1.append(data[c]['x1'].pop())
                        y.append(classes.index(c))
                        global_vec.append(data[c]['global_vec'].pop())
                        masks.append(data[c]['mask'].pop())
                        if args.write_params:
                            params.append(data[c]['params'].pop())
                else:
                    for c in range(chunk_size):
                        if len(data['x']) == 0:
                            break
                        
                        X.append(data['x'].pop())
                        edge_index.append(data['edge_index'].pop())
                        X1.append(data['x1'].pop())
                        y.append(data['y'].pop()[0])
                        global_vec.append(data['global_vec'].pop())
                        masks.append(data['mask'].pop())
                        if args.write_params:
                            params.append(data['params'].pop())
                        
                    
    
                X = np.array(X, dtype = np.float32)
                edge_index = np.array(edge_index, dtype = np.int32)
                X1 = np.array(X1)
                
                if classification:
                    y = np.array(y, dtype = np.uint8)
                else:
                    y = np.array(y, dtype = np.float32)
                global_vec = np.array(global_vec, dtype = np.float32)
                masks = np.array(masks, dtype = np.uint8)
                
                if args.write_params:
                    params = np.array(params, dtype = np.float32)
                
                val = np.random.uniform() < val_prop
                
                if not val:
                    ofile.create_dataset('{0:06d}/x'.format(counter), data = X, compression = 'lzf')
                    ofile.create_dataset('{0:06d}/x1'.format(counter), data = X1, compression = 'lzf')
                    ofile.create_dataset('{0:06d}/edge_index'.format(counter), data = edge_index, compression = 'lzf')
                    ofile.create_dataset('{0:06d}/mask'.format(counter), data = np.array(masks), compression = 'lzf')
                    ofile.create_dataset('{0:06d}/global_vec'.format(counter), data = global_vec, compression = 'lzf')
                    ofile.create_dataset('{0:06d}/y'.format(counter), data = y, compression = 'lzf')
                    if args.write_params:
                        ofile.create_dataset('{0:06d}/params'.format(counter), data = params, compression = 'lzf')
                    
                    ofile.flush()
                
                    counter += 1
                else:
                    ofile_val.create_dataset('{0:06d}/x'.format(val_counter), data = X, compression = 'lzf')
                    ofile_val.create_dataset('{0:06d}/x1'.format(val_counter), data = X1, compression = 'lzf')
                    ofile_val.create_dataset('{0:06d}/edge_index'.format(val_counter), data = edge_index, compression = 'lzf')
                    ofile_val.create_dataset('{0:06d}/mask'.format(val_counter), data = np.array(masks), compression = 'lzf')
                    ofile_val.create_dataset('{0:06d}/global_vec'.format(val_counter), data = global_vec, compression = 'lzf')
                    ofile_val.create_dataset('{0:06d}/y'.format(val_counter), data = y, compression = 'lzf')
                    if args.write_params:
                        ofile_val.create_dataset('{0:06d}/params'.format(counter), data = params, compression = 'lzf')
                    
                    ofile_val.flush()
                
                    val_counter += 1
                    
                if classification:
                    cond = all([len(data[u]['x']) > 0 for u in classes])
                else:
                    cond = (len(data['x']) > 0)
                    
            
        logging.info('have {} training, {} validation chunks...'.format(counter, val_counter))
            
    ofile.close()
    if val_prop > 0:
        ofile_val.close()
    

        
    # ${code_blocks}

if __name__ == '__main__':
    main()


