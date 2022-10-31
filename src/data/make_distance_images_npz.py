# -*- coding: utf-8 -*-
import os
import argparse
import logging

import glob
import numpy as np
import cv2

from scipy.spatial.distance import squareform

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

    ifiles = sorted(glob.glob(os.path.join(args.idir, '*.npz')))
    
    for ifile in ifiles:
        logging.info('writing images for {}...'.format(ifile))
        
        odir = os.path.join(args.odir, ifile.split('/')[-1].split('.')[0])
        
        os.system('mkdir -p {}'.format(odir))
        
        x = np.load(ifile)
        D = x['D']
        
        for ix in range(len(D)):
            d = squareform(D[ix])
            
            i, j = np.where(d > 0)
            d[i, j] = np.log(d[i, j])
            
            d = ((d - np.min(d)) / (np.max(d) - np.min(d)) * 255.).astype(np.uint8)
            
            d = np.array([d, d, d], dtype = np.uint8).transpose(1, 2, 0)
            
            cv2.imwrite(os.path.join(odir, '{0:05d}.png'.format(ix)), d)
            
        
            
            

if __name__ == '__main__':
    main()
