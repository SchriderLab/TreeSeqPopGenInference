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

from torchvision.utils import make_grid
import torch

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
    
    parser.add_argument("--sample_dir", default = "None")
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
            
    if not os.path.exists(args.sample_dir):
        os.system('mkdir -p {}'.format(args.sample_dir))
        logging.debug('root: made output directory {0}'.format(args.sample_dir))
    # ${odir_del_block}

    return args

import pickle

def main():
    args = parse_args()
    
    cdf = pickle.load(open(args.cdf, 'rb'))['cdf']
    
    ifiles = sorted(glob.glob(os.path.join(args.idir, '*.npz')))
    
    counter = 0
    for ifile in ifiles:
        logging.info('writing images for {}...'.format(ifile))
        
        odir = os.path.join(args.odir, ifile.split('/')[-1].split('.')[0])
        
        x = np.load(ifile)
        D = np.exp(x['W'])
        D[D < cdf.x[0]] = cdf.x[0]
        D[D > cdf.x[-1]] = cdf.x[-1]

        D = cdf(D)
        
        F = x['F']
        F /= np.max(F)
        
        im_sample = []
        for ix in range(len(D)):
            d = D[ix]
            f = F[ix]
            
            im = np.zeros(d.shape + (3,))
            im[:,:,0] = f * d
            im[:,:,1] = d
            im[:,:,2] = f
            
            if ix < 64:
                im_sample.append(im)
            
            cv2.imwrite('{1}_{0:05d}.png'.format(ix, odir), (cv2.resize(im, (128, 128)) * 255).astype(np.uint8))

        im = torch.FloatTensor(np.array(im_sample).transpose(0,3,1,2))
        
        grid_im = (make_grid(im, nrow = 8).numpy()[0,:,:] * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(args.sample_dir, '{0:03d}.png'.format(counter)), grid_im)
        counter += 1

if __name__ == '__main__':
    main()
