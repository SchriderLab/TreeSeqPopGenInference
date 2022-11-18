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

from torch.nn import SmoothL1Loss

import cv2

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

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
    parser.add_argument("--ifile", default = "0410_val.hdf5")
    
    parser.add_argument("--size", default = "256")
    parser.add_argument("--classes", default = "hard,hard-near,neutral,soft,soft-near")
    
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--e_tol", default = "0.002")
    parser.add_argument("--max_step", default = "1250")
    parser.add_argument("--device", default = "cuda")
    
    parser.add_argument("--ckpt", default = "050000.pt")
    
    parser.add_argument("--batch_size", default = "32")
    parser.add_argument("--lr", default = "0.15")
    parser.add_argument("--step", type=int, default=1000, help="optimize iterations")

    parser.add_argument("--latent", default = "512")


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
    print(latent_in.shape)
    latent_in.requires_grad = True
    
    for noise in noises:
        noise.requires_grad = True

    optimizer = optim.Adam([latent_in] + noises, lr=lr)
    
    L = SmoothL1Loss()

    for i in range(max_step):
        optimizer.zero_grad()
        
        t = i / step
        lr = get_lr(t, lr)
        optimizer.param_groups[0]["lr"] = lr
        noise_strength = latent_std * 0.05 * max(0, 1 - t / 0.75) ** 2
        latent_n = latent_noise(latent_in, noise_strength.item())

        img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)

        batch, channel, height, width = img_gen.shape

        #p_loss = percept(img_gen, imgs).sum()
        n_loss = noise_regularize(noises)
        mse_loss = L(img_gen, imgs)

        loss = mse_loss + 1e-5 * n_loss
        loss.backward()
        optimizer.step()

        noise_normalize_(noises)

        if mse_loss.item() < e_tol:
            break
    
    for k in range(len(noises)):
        if str(k) not in noises_d.keys():
            noises_d[str(k)] = [noises[k].detach().cpu().numpy()]
        else:
            noises_d[str(k)].append(noises[k].detach().cpu().numpy())
    
    return latent_in.detach().cpu().numpy(), noises, mse_loss.item()

import copy
from auction_lap import auction_lap

def get_starting_points(imgs, g_ema, 
                        n_steps = 100, device = torch.device('cuda')):
    
    # on CPU regardless
    L = np.zeros((imgs.shape[0], 512))
    
    imgs = imgs.to(device)
    imgs = torch.unsqueeze(torch.flatten(imgs, 1, -1), 0)
    g_ema.eval()
    
    N = imgs.shape[-1]
    
    min_errs = np.array([np.inf for k in range(imgs.shape[1])])
    for ix in range(n_steps):
        with torch.no_grad():
            noise_sample = torch.randn(imgs.shape[1], 512, device=device)
            l = g_ema.style(noise_sample)
            
            imgs_, _ = g_ema([l], input_is_latent = True)
            imgs_ = torch.unsqueeze(torch.flatten(imgs_, 1, -1), 0)
            
            D = torch.cdist(imgs, imgs_)[0]
            
            score, ii, _ = auction_lap(torch.round(D * 100))
            D = D.detach().cpu().numpy()
            l = l.detach().cpu().numpy()
            ii = ii.detach().cpu().numpy()
            ij = list(zip(list(range(imgs.shape[1])), list(ii)))

            l = l[ii]
            D = np.array([D[i, j] / N for i, j in ij])

            L[np.where(D < min_errs)] = l[np.where(D < min_errs)]
            min_errs[np.where(D < min_errs)] = D[np.where(D < min_errs)]

    return L

def main():
    args = parse_args()
    
    resize = int(args.size)
    device = torch.device(args.device)
    
    g_ema = Generator(int(args.size), int(args.latent), 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema = g_ema.to(device)
    g_ema.eval()
    
    max_step = int(args.max_step)
    e_tol = float(args.e_tol)
    n_mean_latent = 10000
    
    classes = args.classes.split(',')
    
    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    
    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = g_ema.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5
        
    ifile = h5py.File(args.ifile, 'r')
    keys = list(ifile.keys())
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

            x = []
            
            ims = []
            for k in range(D.shape[0]):
                im = map_to_im(D[k], size = int(args.size))
                #plt.imshow(im)
                #plt.show()
                
                im = transform(Image.fromarray(im))
                im_ = im.detach().cpu().numpy()
                ims.append(im_)
                
                #plt.imshow(im_.transpose(1, 2, 0))
                #plt.show()
                
                
                x.append(im)
                
            x = torch.stack(x, 0)
            
            """
            errs = []
            
            min_err = np.inf
            for ix in range(500):
                with torch.no_grad():
                    noise_sample = torch.randn(1, 512, device=device)
                    l = g_ema.style(noise_sample)
                    print(l.shape)
                    
                    img_, _ = g_ema([l], input_is_latent = True)
                    
                    err = F.mse_loss(img_, img.to(device))
                    errs.append(err.item())
                    
                    if errs[-1] < min_err:
                        latent_ = l
                        min_err = copy.copy(errs[-1])
            
            img_, _ = g_ema([latent_], input_is_latent = True)
            img_ = img_.detach().cpu().numpy()
            
            print(min_err)
            
            plt.hist(errs, bins = 35)
            plt.show()
            
            plt.imshow(img_[0].transpose(1, 2, 0))
            plt.show()
            
            plt.imshow(img[0].detach().cpu().numpy().transpose(1, 2, 0))
            plt.show()
            """
            ii = even_chunks(list(range(x.shape[0])), int(args.batch_size))

            v = []
            mses = []
            
            for ix in ii:
                imgs = x[ix]
                
                logging.info('getting starting points via random search...')
                latent_in = torch.FloatTensor(get_starting_points(imgs, g_ema)).to(device)
                latent_in.requires_grad = True

                logging.info('optimizing...')
                imgs = imgs.to(device)
                
                noises_single = g_ema.make_noise()
                noises = []
                for noise in noises_single:
                    noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())
                
                for noise in noises:
                    noise.requires_grad = True

                optimizer = optim.Adam([latent_in] + noises, lr=float(args.lr))

                min_loss = np.inf
                for i in range(max_step):
                    optimizer.zero_grad()
                    
                    t = i / 1000
                    lr = get_lr(t, float(args.lr))
                    optimizer.param_groups[0]["lr"] = lr
                    noise_strength = latent_std * 0.05 * max(0, 1 - t / 0.75) ** 2
                    latent_n = latent_noise(latent_in, noise_strength.item())

                    img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)

                    batch, channel, height, width = img_gen.shape

                    #p_loss = percept(img_gen, imgs).sum()
                    n_loss = noise_regularize(noises)
                    mse_loss = F.mse_loss(img_gen, imgs)

                    loss = mse_loss + 1e-5 * n_loss
                    loss.backward()
                    optimizer.step()

                    noise_normalize_(noises)
                    
                    if mse_loss.item() < e_tol:
                        break
                
                    if mse_loss.item() < min_loss:
                        min_loss = copy.copy(mse_loss.item())
                        V = latent_in.detach().cpu().numpy()
            
                logging.info('have min mse: {}...'.format(min_loss))
                v.append(V)
                
                mses.append(min_loss)
                
            v = np.concatenate(v, 0)
            
            np.savez(os.path.join(args.odir, '{}_{}.npz'.format(key, skey)), x = v, x1 = info_v, x2 = global_v, mses = np.array(mses))
                
            

    # ${code_blocks}

if __name__ == '__main__':
    main()
