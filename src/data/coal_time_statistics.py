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
    
    bins = np.linspace(-7, 7., 100)
    h = np.zeros(len(bins) - 1)
    
    l = []
    for ifile in ifiles:
        x = np.load(ifile)
        
        D = x['D']
        loc = x['loc']
        l.append(loc[0])
        
        h += np.histogram(np.log(D[:,-64 * 63 / 2:]).flatten(), bins, density = True)[0]
        
        Dmax = np.max(np.log(D), axis = -1)
        Dmin = np.min(np.log(D), axis = -1)

        maxs.append(np.max(Dmax))
        mins.append(np.min(Dmin))
        
        mean.append(np.mean(D))
        var.append(np.std(D))
        
    print(h)
    plt.bar(bins[:-1] + np.diff(bins) / 2., h)
    plt.savefig('log_hist.png', dpi = 100)
    plt.close()
        
    plt.scatter(mean, var, c = l, cmap = 'viridis')
    plt.savefig('mean_std_a001.png', dpi = 100)
    plt.close()

    print(np.max(maxs))
    print(np.mean(maxs))
    print(np.std(maxs))
    print(np.mean(mins))
    print(np.median(mins))
    print(np.percentile(mins, 90))
    print(np.std(mins))
    print(np.min(mins))

    # ${code_blocks}

if __name__ == '__main__':
    main()

