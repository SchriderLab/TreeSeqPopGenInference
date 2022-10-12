# -*- coding: utf-8 -*-
import os
import argparse
import logging

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import h5py

import numpy as np
from scipy.spatial.distance import squareform

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "None")

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
    
    ifile = h5py.File(args.ifile, 'r')
    
    keys = list(ifile.keys())
    
    d_mean = np.zeros((18366,))
    count = 0
    
    for key in keys:
        d = np.array(ifile[key]['D']).reshape(20, -1)
        print(d.shape)
        print(np.mean(d, axis = 0).shape)
        
        
        d_mean += np.mean(d, axis = 0)
        count += 1
        
    D = squareform(d_mean / count)
    
    plt.imshow(D)
    plt.colorbar()
    
    plt.savefig(os.path.join(args.odir, 'D.png'), dpi = 100)
    plt.close()
    
    d_mean = d_mean / count
    d_mean = d_mean.reshape((1, d_mean.shape[0]))
    
    d_var = np.zeros(d_mean.shape)
    count = 0
    
    for key in keys:
        d = np.array(ifile[key]['D']).reshape(20, -1)
        
        d = (d - d_mean)**2
        d = np.mean(d, axis = 0)
        
        d_var += d
        count += 1
    
    d_var = d_var / (count - 1)
    
    np.savez(os.path.join(args.odir, 'D_mean.npz'), mean = d_mean, var = d_var)
    

    # ${code_blocks}

if __name__ == '__main__':
    main()
