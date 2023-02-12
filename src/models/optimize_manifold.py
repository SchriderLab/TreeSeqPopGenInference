# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from torch.autograd import Variable
import itertools

# for now, distributions are uncorrelated i.e. sigma is zero everywhere but the diagonal
def KL_div_gaussian(mu0, mu1, sigma0, sigma1):
    k = mu0.shape[1]
    
    sigma0 = torch.diag_embed(sigma0, offset=0)
    sigma1 = torch.diag_embed(sigma1, offset=0)
    
    # trace term:
    t1 = torch.bmm(torch.linalg.inv(sigma1), sigma0)
    t1 = torch.einsum('bii->b', t1)
    
    # log-determinant term:
    t2 = torch.log(torch.linalg.det(sigma1) / torch.linalg.det(sigma0))
    
    # mu term:
    t3 = torch.bmm(torch.unsqueeze(mu1, 1) - torch.unsqueeze(mu0, 1), torch.linalg.inv(sigma1))
    t3 = torch.bmm(t3, torch.transpose(torch.unsqueeze(mu1, 1) - torch.unsqueeze(mu0, 1), 1, 2))
    
    
    return (t1 + t2 + torch.squeeze(torch.squeeze(t3)) - k) * 0.5

class KLGaussian(nn.Module):
    def __init__(self, n_points = 64, k = 16):
        
        return
    
    def forward(self):
        
        return

import os
import argparse
import logging
import glob
import numpy as np
import matplotlib.pyplot as plt

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "/mirror/GeneGraphs/a001_kl_matrix")
    parser.add_argument("--k", default = "16")
    parser.add_argument("--N", default = "128")
    
    parser.add_argument("--n_steps", default = "10000")

    parser.add_argument("--odir", default = "None")
    parser.add_argument("--ofile", default = "manifold.npz")
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
    device = torch.device('cuda')

    N = int(args.N)
    Z = []
    for ix in range(N):
        Z.append(np.load(ifiles[ix])['row'][:N])
        
    Z = np.array(Z, dtype = np.float32)
    
    k = int(args.k)
    
    n = N**2 - N
    
    mu = Variable(torch.randn((N, k)).to(device), requires_grad=True)
    sigma = Variable(torch.abs(torch.randn((N, k)).to(device)), requires_grad=True)

    indices = np.array(list(itertools.permutations(range(N), 2)), dtype = np.int32)
    i0_ = indices[:,0]
    i1_ = indices[:,1]
    
    Z_flat = Z[i0_, i1_]
    ix = list(np.where(~np.isnan(Z_flat))[0])
    
    Z_flat = torch.FloatTensor(Z_flat[ix]).to(device)
    i0 = torch.LongTensor(i0_[ix]).to(device)
    i1 = torch.LongTensor(i1_[ix]).to(device)
    
    i0_ = i0_[ix]
    i1_ = i1_[ix]
    
    optimizer = torch.optim.Adam([mu, sigma], lr = 0.001)
    
    n = i0.shape[0]
    
    losses = []
    for ix in range(int(args.n_steps)):
        optimizer.zero_grad()
        
        mu0 = mu[i0,:]
        s0 = sigma[i0,:]
        
        mu1 = mu[i1,:]
        s1 = sigma[i1,:]

        kl = KL_div_gaussian(mu0, mu1, s0, s1)

        loss = torch.mean((kl - Z_flat) ** 2)
        loss.backward()
        
        optimizer.step()
        print(loss.item())
        
        losses.append(loss.item())
        
    Z_ = kl.detach().cpu().numpy()
    
    Z_new = np.zeros(Z.shape)
    Z_new[::] = np.nan

    Z_new[i0_, i1_] = Z_

    fig, axes = plt.subplots(ncols = 2)
    axes[0].imshow(Z)
    axes[1].imshow(Z_new)
    
    plt.show()

    mu = mu.detach().cpu().numpy()
    sigma = mu.detach().cpu().numpy()
    
    np.savez(args.ofile, mu = mu, sigma = sigma)
    

    # ${code_blocks}

if __name__ == '__main__':
    main()
    
    """
    mu0 = torch.randn((16, 16))
    mu1 = torch.randn((16, 16))

    s0 = torch.abs(torch.randn((16, 16)))
    s1 = torch.abs(torch.randn((16, 16)))
    
    print(KL_div_gaussian(mu0, mu1, s0, s1)[0,::])
    """