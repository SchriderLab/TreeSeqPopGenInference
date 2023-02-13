# -*- coding: utf-8 -*-

import os
import argparse
import logging

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import cv2
import glob

import numpy as np

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")

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
    
    ifiles = glob.glob(os.path.join(args.idir, '*.npz'))
    
    mean = []
    var = []
    
    maxs = []
    mins = []
    
    l = []
    for ifile in ifiles:
        x = np.load(ifile)
        
        D = x['D']
        loc = x['loc']
        l.append(loc[-1])
        
        Dmax = np.max(np.log(D))
        Dmin = np.min(np.log(D))

        maxs.append(Dmax)
        mins.append(Dmin)
        
        mean.append(np.mean(Dmax))
        var.append(np.std(Dmax))
        
    plt.scatter(mean, var, c = l, cmap = 'viridis')
    plt.savefig('mean_std_a001.png', dpi = 100)
    plt.close()

    print(np.max(maxs))
    print(np.mean(maxs))
    print(np.std(maxs))
    print(np.mean(mins))
    print(np.std(mins))
    print(np.min(mins))

    # ${code_blocks}

if __name__ == '__main__':
    main()

