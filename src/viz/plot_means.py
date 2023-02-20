# -*- coding: utf-8 -*-
import os
import argparse
import logging

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
import glob

from scipy.spatial.distance import squareform
import pickle
import numpy as np

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "/pine/scr/d/d/ddray/msprime_grid_m03")
    parser.add_argument("--cdf", default = "None")
    
    parser.add_argument("--grid", default = "m03")

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
    cdf = pickle.load(open(args.cdf, 'rb'))
    
    if args.grid == "m03":
        N = 12
        
        mean_images = []
        for ix, ifile in enumerate(ifiles):
            x = np.load(ifile)
                
            D = x['D']
            D[D < cdf.x[0]] = cdf.x[0]
            D[D > cdf.x[-1]] = cdf.x[-1]
            D = cdf(D)
            
            _ = []
            for d in D:
                im = squareform(d)
                
                _.append(im)
                
            mean_images.append(np.mean(np.array(_), axis = 0))
            
        mean_images = torch.FloatTensor(np.expand_dims(np.array(mean_images), 1))
        
        ret = make_grid(mean_images, nrow = N).numpy()[0,:,:]
        
        plt.imshow(ret)
        plt.colorbar()
        plt.savefig('m03_grid.png', dpi = 100)
        plt.close()
                
    

    # ${code_blocks}

if __name__ == '__main__':
    main()

