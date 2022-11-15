# -*- coding: utf-8 -*-
import os
import argparse
import logging

import h5py
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import squareform
import cv2

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "seln_trees_auto.hdf5")
    parser.add_argument("--classes", default = "hard,hard-near,neutral,soft,soft-near")
    
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
    
    classes = args.classes.split(',')
    
    try:
        c_ix = [classes.index(args.cl)]
    except:
        c_ix = list(range(len(classes)))
            
    ifile = h5py.File(args.ifile, 'r')
    keys = list(ifile.keys())
    
    i, j = np.triu_indices(191)
    ii = list(np.where(i != j)[0])
    
    for c_ix_ in c_ix:
        counter = 0
        
        c_name = classes[c_ix_]
        
        odir = os.path.join(args.odir, c_name)
        os.system('mkdir -p {}'.format(odir))
        
        for key in keys:
            d = np.array(ifile[key]['D'])[c_ix_].reshape(4, -1)[:,ii]
            
            i, j = np.where(d > 0)
            d[i, j] = np.log(d[i, j])
            
            for k in range(d.shape[0]):
                # a colormap and a normalization instance
                cmap = plt.cm.viridis
                norm = plt.Normalize(vmin = np.min(d[k]), vmax = np.max(d[k]))
                
                image = cmap(norm(squareform(d[k])))
                
                image = (image[:,:,:3] * 255).astype(np.uint8)[:,:,::-1]
                
                cv2.imwrite(os.path.join(odir, '{0:06d}.png'.format(counter)), image)
                counter += 1

    # ${code_blocks}

if __name__ == '__main__':
    main()
