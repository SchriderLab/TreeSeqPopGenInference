# -*- coding: utf-8 -*-
import os
import argparse
import logging


## StyleGAN 2 code
# this script will take a large directory of images and project them to latent space via
# gradient descent on some generator weights
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

def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def even_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    _ = []
    
    for i in range(0, len(lst), n):
        _.append(lst[i:i + n])

    return _

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--size", default = "256")
    
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--ckpt", default = "None")
    
    parser.add_argument("--batch_size", default = "8")

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
    
    resize = min(int(args.size), 256)
    device = torch.device('cuda')
    
    g_ema = Generator(int(args.size), 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)
    
    
    n_mean_latent = 10000
    
    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = g_ema.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5
    
    idirs = glob.glob(os.path.join(args.idir, '*'))

    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    
    result = dict()
    for idir in idirs:
        cl = idir.split('/')[-1]
        
        ifiles = sorted(glob.glob(os.path.join(args.idir, '*.png')))
        result[cl] = []
        
        indices = even_chunks(range(len(ifiles)), int(args.batch_size))
        
        for ii in indices:
            imgs = []
            
            ifiles_ = [ifiles[u] for u in ii]
            for imgfile in ifiles_:
                img = transform(Image.open(imgfile).convert("RGB"))
                imgs.append(img)

            imgs = torch.stack(imgs, 0).to(device)
            
            noises_single = g_ema.make_noise()
            noises = []
            for noise in noises_single:
                noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

            latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)
            latent_in.requires_grad = True
            
            for noise in noises:
                noise.requires_grad = True

            optimizer = optim.Adam([latent_in] + noises, lr=args.lr)
            
            pbar = tqdm(range(args.step))
            latent_path = []

            for i in pbar:
                t = i / args.step
                lr = get_lr(t, args.lr)
                optimizer.param_groups[0]["lr"] = lr
                noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
                latent_n = latent_noise(latent_in, noise_strength.item())

                img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)

                batch, channel, height, width = img_gen.shape

                if height > 256:
                    factor = height // 256

                    img_gen = img_gen.reshape(
                        batch, channel, height // factor, factor, width // factor, factor
                    )
                    img_gen = img_gen.mean([3, 5])

                #p_loss = percept(img_gen, imgs).sum()
                n_loss = noise_regularize(noises)
                mse_loss = F.mse_loss(img_gen, imgs)

                loss = mse_loss + 1e-5 * n_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                noise_normalize_(noises)

                if (i + 1) % 100 == 0:
                    latent_path.append(latent_in.detach().clone())

                pbar.set_description(
                    (
                        f" noise regularize: {n_loss.item():.4f};"
                        f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
                    )
                )
            
            print(latent_in.shape)
            
            
        

    

    # ${code_blocks}

if __name__ == '__main__':
    main()
