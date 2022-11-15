# -*- coding: utf-8 -*-
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

from project_images import noise_regularize, noise_normalize_, even_chunks, make_image, latent_noise, get_lr

import h5py
import matplotlib.pyplot as plt

import cv2

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def map_to_im(x, size = 256, viridis = True):
    # x is (n, n) matrix of floats to (n, n, 3)
    if viridis:
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin = np.min(x), vmax = np.max(x))
        
        im = cmap(norm(x))

    im = (im * 255).astype(np.uint8)
    
    im = cv2.resize(im, (size, size))
    plt.imshow(im)
    plt.show()
    
    im = (im.astype(np.float32) / 255.)[:,:,:3].transpose(2, 0, 1)

    return im    

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "0410_val.hdf5")
    
    parser.add_argument("--size", default = "256")
    parser.add_argument("--classes", default = "hard,hard-near,neutral,soft,soft-near")
    
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--e_tol", default = "0.01")
    parser.add_argument("--max_step", default = "1250")
    
    parser.add_argument("--device", default = "cuda")
    
    parser.add_argument("--ckpt", default = "120000.pt")
    
    parser.add_argument("--batch_size", default = "8")
    parser.add_argument("--lr", default = "0.1")
    parser.add_argument("--step", type=int, default=1000, help="optimize iterations")


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

def optimize(imgs, latent_mean, latent_std, lr, max_step, e_tol, g_ema, step = 1000):
    noises_d = dict()
    
    noises_single = g_ema.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)
    latent_in.requires_grad = True
    
    for noise in noises:
        noise.requires_grad = True

    optimizer = optim.Adam([latent_in] + noises, lr=lr)

    for i in range(max_step):
        t = i / step
        lr = get_lr(t, lr)
        optimizer.param_groups[0]["lr"] = lr
        noise_strength = latent_std * 0.05 * max(0, 1 - t / 0.75) ** 2
        latent_n = latent_noise(latent_in, noise_strength.item())

        img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)

        batch, channel, height, width = img_gen.shape

        #p_loss = percept(img_gen, imgs).sum()
        n_loss = noise_regularize(noises)
        mse_loss = F.mse_loss(img_gen, imgs)

        loss = mse_loss + 1e-5 * n_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        noise_normalize_(noises)

        if (i + 1) % 100 == 0:
            logging.info('step {}: mse loss {}...'.format(i + 1, mse_loss.item()))

        if mse_loss.item() < e_tol:
            break
    
    for k in range(len(noises)):
        if str(k) not in noises_d.keys():
            noises_d[str(k)] = [noises[k].detach().cpu().numpy()]
        else:
            noises_d[str(k)].append(noises[k].detach().cpu().numpy())
    
    return latent_in.detach().cpu().numpy(), noises_d, mse_loss.item()

def main():
    args = parse_args()
    
    resize = int(args.size)
    device = torch.device(args.device)
    
    g_ema = Generator(int(args.size), 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)
    
    max_step = int(args.max_step)
    e_tol = float(args.e_tol)
    n_mean_latent = 10000
    
    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = g_ema.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5
        
    ifile = h5py.File(args.ifile, 'r')
    keys = list(ifile.keys())
    
    for key in keys:
        skeys = list(ifile[key].keys())
        
        for skey in skeys:
            D = np.array(ifile[key][skey]['D'])[:,-2,:,:]
            D[np.where(D > 0.)] = np.log(D[np.where(D > 0.)])
            global_v = np.array(ifile[key][skey]['global_vec'])

            x = []
            for k in range(D.shape[0]):
                x.append(map_to_im(D[k], size = int(args.size)))
                
            x = torch.FloatTensor(np.array(x))
            
            ii = even_chunks(list(range(x.shape[0])), int(args.batch_size))

            v = []
            for ix in ii:
                x_ = x[ix].to(device)
                print(x_.shape)
                
                v_, _, _ = optimize(x_, latent_mean, latent_std, float(args.lr), max_step, e_tol, g_ema)
                
                v.append(v_)
                
            v = np.hstack(v)
            
            print(v.shape, x.shape)
            sys.exit()
            

    # ${code_blocks}

if __name__ == '__main__':
    main()
