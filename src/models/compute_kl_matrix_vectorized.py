# -*- coding: utf-8 -*-
import os
import argparse
import logging

import numpy as np
import glob

import copy
import sys

from scipy.special import softmax
import random
import scipy.integrate as integrate

import matplotlib.pyplot as plt

from torch import nn
import torch

import json

class CoalExponentialRateFunction(nn.Module):
    # alpha: (n, ) vector of exponential growth/decay rates where n = number of populations (alpha != 0.)
    # T: (n, ) vector of stopping times for growth or None
    # N: (n, ) vector of initial effective population sizes
    def __init__(self, alpha, N, T = None):
        super().__init__()
        # (1, n)
        self.alpha = nn.Parameter(torch.unsqueeze(torch.FloatTensor(alpha), 0))
        if T is not None:
            self.T = nn.Parameter(torch.unsqueeze(torch.FloatTensor(T), 0))
        else:
            self.T = None
        self.N = nn.Parameter(torch.unsqueeze(torch.FloatTensor(N), 0))
        
    # t is the vector (|e|,) of times in generations
    # s is the matrix of population sizes (|e|, n)
    def forward(self, t, t0, s):
        t = t.unsqueeze(1)
        t0 = t0.unsqueeze(1)
        s_ = s * (s - 1) / 2.
        
        N = (2 * self.N * torch.exp(-self.alpha * t0))
        
        y = s_ * (torch.exp(self.alpha * t) - 1) / (self.alpha * N)

        return y
    
    def intensity(self, t, s):
        t = t.unsqueeze(1)
        s = s * (s - 1) / 2.
        
        return (2. * torch.exp(-self.alpha * t) * self.N) ** -1
    
class MigExponentialRateFunction(nn.Module):
    def __init__(self, alpha, N, m, T = None):
        super().__init__()
        
        self.n_pops = alpha.shape[0]
        
        alpha_mat = np.tile(alpha.reshape(1, -1), (self.n_pops, 1))
        alpha_mat = alpha_mat - alpha_mat.T
        
        i, j = np.where(m != 0)
        self.n_nonzero = len(i)
        
        self.i = i
        self.j = j
        
        N = np.tile(N.reshape(1, -1), (self.n_pops, 1))
        N_mat = N / N.T

        self.alpha = nn.Parameter(torch.FloatTensor(alpha_mat[i, j].reshape(1, self.n_nonzero)))

        self.N = nn.Parameter(torch.FloatTensor(N_mat[i, j].reshape(1, self.n_nonzero)))
        self.m = nn.Parameter(torch.FloatTensor(m[i,j].reshape(1, self.n_nonzero)))
        if T is not None:
            self.T = nn.Parameter(torch.unsqueeze(torch.FloatTensor(T[i]), 0))
        else:
            self.T = None
    
    # t is the vector (|e|,) of times in generations
    # t0 is the vector (|e|,) of prior event times in generations
    # s is the matrix of population sizes (|e|, n)
    def forward(self, t, t0, s):
        t = t.unsqueeze(1)
        t0 = t.unsqueeze(1)
        
        alpha = torch.tile(self.alpha, (t.shape[0], 1))
        
        N = (self.N * torch.exp(self.alpha * t0))[:,:,0]
                
        y = s[:,self.i] * self.m * N * (torch.exp(self.alpha * t) - 1)
        y = torch.where(alpha == 0, s[:,self.i] * self.m * N * t, y / self.alpha)
        
        return y

    def intensity(self, t, s):
        t = t.unsqueeze(1)
        
        return s[:, self.i] * self.m * self.N * (torch.exp(self.alpha * t))

import math

class DemographyLL(nn.Module):
    def __init__(self, coal, mig):
        super().__init__()
        
        self.coal = coal
        self.mig = mig
        
    def forward(self, t0, t, s, k_coal, k_mig):
        d1 = self.coal(t - t0, t0, s)
        p_coal = torch.exp(-d1) * torch.pow(d1, k_coal)
        p_coal = torch.where(s <= 1, torch.tensor(1., dtype = p_coal.dtype), p_coal)
        p_coal = torch.where(p_coal == 0, torch.tensor(1., dtype = p_coal.dtype), p_coal)
        
        p = torch.log2(p_coal).sum()
        
        if self.mig is not None:
            d2 = self.mig(t - t0, t0, s)
            p_mig = torch.exp(-d2) * torch.pow(d2, k_mig)
            p_mig = torch.where(s[:,self.mig.i] == 0, torch.tensor(1., dtype = p_coal.dtype), p_mig)
            p_mig = torch.where(p_mig == 0, torch.tensor(1., dtype = p_coal.dtype), p_mig)
    
            p += torch.log2(p_mig).sum()
        
        
        return p 
    
def setup_ll_inputs(E, N, alpha, m):
    E = np.array(E)
    
    # time of the events
    t = np.array([u[3] for u in E])
    # time of the prior event or zero at the first one
    t0 = np.array([0.] + list(t))[:-1]

    n_pops = m.shape[0]
    n_events = len(t)
    
    pop_sizes = copy.copy(E[:,4:4 + n_pops])
    
    n_nonzero = len(np.where(m > 0.)[0])
    n_total = n_pops + n_nonzero
    
    # the counts of the Poisson process at each time of event
    counts = np.vstack((np.zeros((1, n_total)), copy.copy(E[:,-n_total:])))
    counts = np.diff(counts, axis = 0)
    
    print(counts.dtype, pop_sizes.dtype, t0.dtype)
    
    return torch.FloatTensor(t0), torch.FloatTensor(t), torch.FloatTensor(pop_sizes), torch.LongTensor(counts[:,:n_pops]), \
             torch.LongTensor(counts[:,n_pops:])
    
# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--i", default = "0")
    parser.add_argument("--ofile", default = "None")
    
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
    ifiles_loc = sorted(glob.glob(os.path.join(args.idir, '*.json')))
    ix = int(args.i)

    ret = np.zeros(len(ifiles))
    trees = np.load(ifiles[ix], allow_pickle = True)
    
    ifile = json.load(open(ifiles_loc[ix]))
    
    N = np.array(ifile['N'])
    M = np.array(ifile['M']).reshape(N.shape[0], N.shape[0])
    alpha = np.array(ifile['alpha'])
    
    p = []

    logging.info('getting probabilities for p...')
    if N.shape[0] > 1:
        model = DemographyLL(CoalExponentialRateFunction(alpha, N), MigExponentialRateFunction(alpha, N, M))
    else:
        model = DemographyLL(CoalExponentialRateFunction(alpha, N), None)
    
    with torch.no_grad():
        for events in trees['E']:
            p.append(model(*setup_ll_inputs(events, N, alpha, M)).item())

    logging.info('mean ll: {}'.format(np.mean(p)))

    ii = sorted(list(set(list(range(len(ifiles)))).difference([ix])))
    for ij in ii:
        logging.info('getting probabilities for q_i = {}...'.format(ij))
        #trees_ = np.load(ifiles[ij], allow_pickle = True)
        
        ifile = json.load(open(ifiles_loc[ij]))
        
        N = np.array(ifile['N'])
        M = np.array(ifile['M']).reshape(N.shape[0], N.shape[0])
        alpha = np.array(ifile['alpha'])
        
        q = []
        if N.shape[0] > 1:
            model = DemographyLL(CoalExponentialRateFunction(alpha, N), MigExponentialRateFunction(alpha, N, M))
        else:
            model = DemographyLL(CoalExponentialRateFunction(alpha, N), None)
        
        with torch.no_grad():
            for events in trees['E']:
                q.append(model(*setup_ll_inputs(events, N, alpha, M)).item())
            
        q = np.array(q)
        logging.info('mean ll: {}'.format(np.mean(q)))
        
        ii = list(np.where(np.isnan(q))[0])
        if len(ii) < 10:
            kl_ = np.nanmean(p - q)
        else:
            kl_ = np.nan
            
        logging.info('got kl divergence of {}...'.format(kl_))
        ret[ij] = kl_
        
    np.savez(args.ofile, row = ret)

    # ${code_blocks}

if __name__ == '__main__':
    main()
