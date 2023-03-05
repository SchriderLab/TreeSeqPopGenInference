# -*- coding: utf-8 -*-
import os
import argparse
import logging

import h5py
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import squareform
import cv2
import pickle
import glob

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--classes", default = "hard,hard-near,neutral,soft,soft-near")
    parser.add_argument("--cdf", default = "None")
    parser.add_argument("--n_sample", default = "75")
    
    parser.add_argument("--cl", default = "neutral")

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
        else:
            os.system('rm -rf {}'.format(args.odir))
    # ${odir_del_block}

    return args

def main():
    args = parse_args()
    
    ifiles = sorted(glob.glob(os.path.join(args.idir, '*.hdf5')))
    cdf = pickle.load(open(args.cdf, 'rb'))['cdf']
    
    counter = 0
    for ifile in ifiles:
        print(ifile)
        
        ifile = h5py.File(ifile, 'r')
        logging.info('reading keys...')
        classes = list(ifile.keys())
        classes.remove('max_min')
        
        logging.info('writing images...')
        for c in classes:
            keys = list(ifile[c].keys())
            keys = np.random.choice(keys, int(args.n_sample), replace = False)
            
            c_name = c
                        
            for key in keys:
                D = np.array(ifile[c][key]['D'])
                D = np.log(D)
                D[D < cdf.x[0]] = cdf.x[0]
                D[D > cdf.x[-1]] = cdf.x[-1]
                D = cdf(D)
                
                for k in range(D.shape[0]):
                    d = cv2.resize(squareform(D[k]), (256, 256))
                    
                    cv2.imwrite(os.path.join(args.odir, '{1}_{0:07d}.png'.format(counter, c_name)), (d * 65535).astype(np.uint16))
                    counter += 1

    # ${code_blocks}

if __name__ == '__main__':
    main()
