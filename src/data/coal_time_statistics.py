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
    for ifile in ifiles:
        x = np.load(ifile)
        
        D = x['D']
        D = np.max(np.log(D), axis = 1)

        mean.append(np.mean(D))
        var.append(np.std(D))
        
    plt.scatter(mean, var)
    plt.savefig('mean_std_maxD.png', dpi = 100)
    
    plt.close()

    # ${code_blocks}

if __name__ == '__main__':
    main()

