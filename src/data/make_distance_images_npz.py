# -*- coding: utf-8 -*-
import os
import argparse
import logging

import glob
import numpy as np
import cv2

from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from scipy.special import expit

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--cdf", default = "None")
    
    parser.add_argument("--mean_max_log", default = "6.252627993915886")
    parser.add_argument("--min_clip", default = "-2.0")

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

import pickle

def main():
    args = parse_args()
    
    cdf = pickle.load(open(args.cdf, 'rb'))

    ifiles = sorted(glob.glob(os.path.join(args.idir, '*.npz')))
    
    for ifile in ifiles:
        logging.info('writing images for {}...'.format(ifile))
        
        odir = os.path.join(args.odir, ifile.split('/')[-1].split('.')[0])
        
        x = np.load(ifile)
        D = x['D']

        for ix in range(len(D)):
            d = D[ix]
            d[d < cdf.x[0]] = cdf.x[0]
            d[d > cdf.x[-1]] = cdf.x[-1]
            
            d = cdf(D[ix])
            
            cv2.imwrite('{1}_{0:05d}.png'.format(ix, odir), (squareform(d) * 255).astype(np.uint8))
            
            #np.savez('{1}_{0:05d}.npz'.format(ix, odir), d = d, loc = loc)
            
            

if __name__ == '__main__':
    main()
