# -*- coding: utf-8 -*-
import os
import argparse
import logging

import os
import argparse
import logging

import math

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import glob

import lpips
from swagan import Generator
import numpy as np
import random

from torchvision_mod_layers import resnet34

from project_images import noise_regularize, noise_normalize_, even_chunks, make_image, latent_noise, get_lr

import h5py
import matplotlib.pyplot as plt

from torch.nn import SmoothL1Loss

import cv2
import pickle
from scipy.spatial.distance import squareform

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def chunks(lst, n):
    _ = []
    
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        _.append(lst[i:i + n])

    return _

def map_to_im(x, size = 256, viridis = True):
    # x is (n, n) matrix of floats to (n, n, 3)
    i, j = np.where(x > 0)
    x[i, j] = np.log(x[i, j])
    
    x = ((x - np.min(x)) / (np.max(x) - np.min(x)) * 255.).astype(np.uint8)

    im = np.array([x, x, x], dtype = np.uint8).transpose(1, 2, 0)
    
    return im    


def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    # ${args}
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--latent",
        type=int,
        default=32,
        help="dimensionality of the latent space",
    )
    parser.add_argument(
        "--n_mlp",
        type=int,
        default=6,
        help="dimensionality of the latent space",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=1,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--size", type=int, default=128, help="image sizes for the model"
    )
    parser.add_argument("--lr", default = "0.001")
    parser.add_argument("--weight_decay", default = "0.0")
    
    parser.add_argument("--n_steps", default = "80000")
    parser.add_argument("--ckpt", default = "None")
    parser.add_argument("--proj_ckpt", default = "None")
    parser.add_argument("--idir", default = "None")
    
    parser.add_argument("--cdf", default = "None")
    
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
            logging.info('root: made output directory {0}'.format(args.odir))
    # ${odir_del_block}

    return args

def main():
    args = parse_args()
    
    cdf = pickle.load(open(args.cdf, 'rb'))['cdf']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using " + str(device) + " as device")
    
    model = resnet34(num_classes = args.latent, in_channels = 1).to(device)
    checkpoint = torch.load(args.proj_ckpt, map_location = device)
    model.load_state_dict(checkpoint)
    model.eval()
        
    ifiles = sorted(glob.glob(os.path.join(args.idir, '*.npz')))
    random.shuffle(ifiles)
    
    for ifile in ifiles:
        logging.info('on file {}...'.format(ifile))
        
        ofile = os.path.join(args.odir, ifile.split('/')[-1])        
        
        if os.path.exists(ofile):
            logging.info('already written. moving on...')
            continue
        
        ifile = np.load(ifile)
        D = ifile['D']
        D = cdf(np.log(D))
        
        c = chunks(list(range(D.shape[0])), 64)
        
        ret = []        
        for chunk in c:
            X = []
            
            D_ = D[chunk,:]
            for d in D_:
                X.append(np.expand_dims(cv2.resize(squareform(d), (128, 128)), 0))
                
            X = torch.FloatTensor(X).to(device) * 2 - 1
        
            with torch.no_grad():
                v = model(X)
                
            v = v.detach().cpu().numpy()
            ret.extend(v)
                
        np.savez(ofile, w = np.array(ret))
    
            

    # ${code_blocks}

if __name__ == '__main__':
    main()


