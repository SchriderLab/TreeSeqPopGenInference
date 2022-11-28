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

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def map_to_im(x, size = 256, viridis = True):
    # x is (n, n) matrix of floats to (n, n, 3)
    i, j = np.where(x > 0)
    x[i, j] = np.log(x[i, j])
    
    x = ((x - np.min(x)) / (np.max(x) - np.min(x)) * 255.).astype(np.uint8)

    im = np.array([x, x, x], dtype = np.uint8)
    
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
        default=128,
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
        "--size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument("--lr", default = "0.001")
    parser.add_argument("--weight_decay", default = "0.0")
    
    parser.add_argument("--n_steps", default = "80000")
    parser.add_argument("--ckpt", default = "None")
    parser.add_argument("--proj_ckpt", default = "None")
    parser.add_argument("--ifile", deafult = "None")
    
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using " + str(device) + " as device")
    
    ckpt = torch.load(args.ckpt, map_location = device)
    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    generator.load_state_dict(ckpt["g_ema"], strict=False)
    generator.eval()
    
    model = resnet34(num_classes = args.latent)
    checkpoint = torch.load(args.proj_ckpt, map_location = device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    ifile = h5py.File(args.ifile, 'r')
    ofile = h5py.File(args.ofile, 'w')
    
    keys = list(ifile.keys())
    for key in keys:
        skeys = list(ifile[key].keys())
    
    L = SmoothL1Loss()
    
    for key in keys:
        skeys = list(ifile[key].keys())
        random.shuffle(skeys)
        
        logging.info('on key {}...'.format(key))
        for skey in skeys:
            logging.info('on skey {}...'.format(skey))
            
            D = np.array(ifile[key][skey]['D'])[:,-2,:,:]
            logging.info('shape: {}'.format(D.shape))
            
            global_v = np.array(ifile[key][skey]['global_vec'])
            info_v = np.array(ifile[key][skey]['info'])
            
            _ = []
            for k in range(D.shape[0]):
                _.append(map_to_im(D[k]))
             
            # down-project
            x = torch.FloatTensor(np.array(_, dtype = np.float32))
            
            with torch.no_grad():
                v = model(x)
            
                # up-project:
                x_pred = generator([v], input_is_latent = True)
                
                loss = L(x_pred, x)
                
                
            v = v.detach().cpu().numpy()
            
            logging.info('got l1 loss of {}...'.format(loss.item()))
            ofile.create_dataset('{}/{}/x'.format(key, skey), data = v)
            ofile.create_dataset('{}/{}/x1'.format(key, skey), data = info_v)
            ofile.create_dataset('{}/{}/x2'.format(key, skey), data = global_v)
            
            ofile.flush()

    ofile.close()
    
            

    # ${code_blocks}

if __name__ == '__main__':
    main()

