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

def nC2(n):
    return n * (n - 1) / 2

def f_a001_coal(alpha, N, t): 
    # alpha > 0. = growth rate
    # t = time in generations
    # N = effective population size at t = 0.
    # dt = time since last event
    # n = sample size (unused here)

    ret = 1 / (N * np.exp(-alpha * t)) * np.exp((1 - np.exp(alpha * t)) / (N * alpha))
    
    return ret
    
def F_a001_coal(alpha, N, t):

    ret = 1 - np.exp((1 - np.exp(alpha * t)) / (alpha * N))
    return ret

def f_a001_mig(a0, a1, m, N1, N2, t):
    # a0 > 0. = growth rate for pop0
    # a1 > 0. = growth rate for pop1
    # a0 != a1.
    # m migration fraction
    # t = time in generations
    # dt = time since last event
    # n = sample size (unused here)

    ret = m * np.exp(-a1 * t + a0 * t) * (N2 / N1) * np.exp(m * (N2 / N1) * (np.exp(t * (a0 - a1)) - 1) / (a1 - a0))
    return ret

def F_a001_mig(a0, a1, m, N1, N2, t):
    
    ret = 1 - np.exp(m * (N2 / N1) * (np.exp(t * (a0 - a1)) - 1) / (a1 - a0))
    
    return ret

def prob_a001_coal(i, a0, a1, m, t0, t1, N, n1, n2):
    t = t1 - t0
    
    # p(t = t | coal first) = P(coal first | t = t) * P(t = t) (P_t_first) / (P(coal first) = P_coal_first)
    # P(coal first) = integral_0^inf {p(t) * product over possible {(1 - F(t))^k} where k is the number of samples or pairs of samples}
    if i == 0:
        if n2 > 1:
            #P_coal_first = integrate.quad(lambda u: F_a001_coal(a0, N * np.exp(-a0 * t0), u) * (1 - F_a001_coal(a0, N * np.exp(-a0 * t0), u)) ** (nC2(n1) - 1) * (1 - F_a001_coal(a1, N * np.exp(-a1 * t0), u)) ** (nC2(n2)) * (1 - F_a001_mig(a0, a1, m,  N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), u)) ** n1, 0, 1e5)[0]
            
            P_t_first = F_a001_coal(a0, N * np.exp(-a0 * t0), t) * (1 - F_a001_coal(a0, N * np.exp(-a0 * t0), t)) ** (nC2(n1) - 1) * (1 - F_a001_coal(a1, N * np.exp(-a1 * t0), t)) ** (nC2(n2)) * (1 - F_a001_mig(a0, a1, m,  N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), t)) ** n1
        else:
            #P_coal_first = integrate.quad(lambda u: F_a001_coal(a0, N * np.exp(-a0 * t0), u) * (1 - F_a001_coal(a0, N * np.exp(-a0 * t0), u)) ** (nC2(n1) - 1) * (1 - F_a001_mig(a0, a1, m,  N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), u)) ** n1, 0, 1e5)[0]
            P_t_first = F_a001_coal(a0, N * np.exp(-a0 * t0), t) * (1 - F_a001_coal(a0, N * np.exp(-a0 * t0), t)) ** (nC2(n1) - 1) * (1 - F_a001_mig(a0, a1, m,  N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), t)) ** n1
    else:
        if n1 > 1:
            #P_coal_first = integrate.quad(lambda u: F_a001_coal(a1, N * np.exp(-a1 * t0), u) * (1 - F_a001_coal(a1, N * np.exp(-a1 * t0), u)) ** (nC2(n2) - 1) * (1 - F_a001_coal(a0, N * np.exp(-a0 * t0), u)) ** (nC2(n1)) * (1 - F_a001_mig(a0, a1, m,  N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), u)) ** n1, 0, 1e5)[0]
            P_t_first = F_a001_coal(a1, N * np.exp(-a1 * t0), t) * (1 - F_a001_coal(a1, N * np.exp(-a1 * t0), t)) ** (nC2(n2) - 1) * (1 - F_a001_coal(a0, N * np.exp(-a0 * t0), t)) ** (nC2(n1)) * (1 - F_a001_mig(a0, a1, m, N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), t)) ** n1
        elif n1 == 1:
            #P_coal_first = integrate.quad(lambda u: F_a001_coal(a1, N * np.exp(-a1 * t0), u) * (1 - F_a001_coal(a1, N * np.exp(-a1 * t0), u)) ** (nC2(n2) - 1) * (1 - F_a001_mig(a0, a1, m,  N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), u)) ** n1, 0, 1e5)[0]
            P_t_first = F_a001_coal(a1, N * np.exp(-a1 * t0), t) * (1 - F_a001_coal(a1, N * np.exp(-a1 * t0), t)) ** (nC2(n2) - 1) * (1 - F_a001_mig(a0, a1, m, N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), t)) ** n1
        else:
            if n2 > 2:
                #P_coal_first = integrate.quad(lambda u: F_a001_coal(a1, N * np.exp(-a1 * t0), u) * (1 - F_a001_coal(a1, N * np.exp(-a1 * t0), u)) ** (nC2(n2) - 1), 0, 1e5)[0]
                P_t_first = F_a001_coal(a1, N * np.exp(-a1 * t0), t) * (1 - F_a001_coal(a1, N * np.exp(-a1 * t0), t)) ** (nC2(n2) - 1) * F_a001_coal(a1, N * np.exp(-a1 * t0), t)
            else:
                return F_a001_coal(a1, N * np.exp(-a1 * t0), t)
                
    return P_t_first

def prob_a001_mig(a0, a1, m, t0, t1, N, n1, n2):
    p_mig_t = 0.
    
    t = t1 - t0
    
    if n2 >= 2 and n1 >= 2:
        #p_mig_first = integrate.quad(lambda u: F_a001_mig(a0, a1, m, N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), u) * (1 - F_a001_coal(a1, N * np.exp(-a1 * t0), u)) ** (nC2(n2)) * (1 - F_a001_coal(a0, N * np.exp(-a0 * t0), u)) ** (nC2(n1)) * (1 - F_a001_mig(a0, a1, m, N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), u)) ** (n1 - 1), 0, 1e5)[0]
        p_mig_t = F_a001_mig(a0, a1, m, N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), t) * (1 - F_a001_coal(a1, N * np.exp(-a1 * t0), t)) ** (nC2(n2)) * (1 - F_a001_coal(a0, N * np.exp(-a0 * t0), t)) ** (nC2(n1)) * (1 - F_a001_mig(a0, a1, m, N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), t)) ** (n1 - 1)
    elif n2 >= 2 and n1 == 1:
        #p_mig_first = integrate.quad(lambda u: F_a001_mig(a0, a1, m, N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), u) * (1 - F_a001_coal(a1, N * np.exp(-a1 * t0), u)) ** (nC2(n2)) * (1 - F_a001_mig(a0, a1, m, N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), u)) ** (n1 - 1), 0, 1e5)[0]
        p_mig_t = F_a001_mig(a0, a1, m, N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), t) * (1 - F_a001_coal(a1, N * np.exp(-a1 * t0), t)) ** (nC2(n2)) * (1 - F_a001_mig(a0, a1, m, N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), t)) ** (n1 - 1)
    elif n2 <= 1 and n1 == 1:
        return F_a001_mig(a0, a1, m, N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), t)
    elif n2 <= 1 and n1 >= 2:
        #p_mig_first = integrate.quad(lambda u: F_a001_mig(a0, a1, m, N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), u) * (1 - F_a001_coal(a0, N * np.exp(-a0 * t0), u)) ** (nC2(n1)) * (1 - F_a001_mig(a0, a1, m, N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), u)) ** (n1 - 1), 0, 1e5)[0]
        p_mig_t = F_a001_mig(a0, a1, m, N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), t) * (1 - F_a001_coal(a0, N * np.exp(-a0 * t0), t)) ** (nC2(n1)) * (1 - F_a001_mig(a0, a1, m, N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), t)) ** (n1 - 1)
    
    return p_mig_t

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--sample_sizes", default = "32,32")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--i", default = "None")
    
    parser.add_argument("--K", default = "10")
    

    parser.add_argument("--ofile", default = "test.npz")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    # ${odir_del_block}

    return args

from scipy.stats import norm, gaussian_kde

def compute_P(events, n, a1, a2, m, pop_sizes, kde_ma, kde_mi, n_reps = 100):
    ret = []
    for k in range(n_reps):
        T_ma = kde_ma.resample(1)[0,0]
        T_mi = kde_mi.resample(1)[0,0]
        
        pops = copy.copy(pop_sizes)
    
        logP = 0.
        t = 0.
        
        # go to distance and normalize
        Tl = [u[-1] for u in events]
        Tl = np.log(2 * Tl)
        Tl = (Tl - np.min(Tl)) / (np.max(Tl) - np.min(Tl))
        
        Tl *= (T_ma - T_mi)
        Tl += T_mi
        Tl = np.exp(Tl)
        
        for ix in range(len(events)):
            e = events[ix]
            t_ = Tl[ix]
            
            if e[0] == 1:
                p_ = prob_a001_mig(a1, a2, m, t, t_, n, pops[0], pops[1])
                logP += np.log2(p_)
                if logP in [np.inf, -np.inf, np.nan]:
                    logP = np.nan
                    break
    
                pops[0] -= 1
                pops[1] += 1
            elif e[0] == 0 and e[1] == 0:
                _ = prob_a001_coal(0, a1, a2, m, t, t_, n, pops[0], pops[1])
                logP += np.log2(_)
                
                if logP in [np.inf, -np.inf, np.nan]:
                    logP = np.nan
                    break
      
                pops[0] -= 1
            elif e[0] == 0 and e[1] == 1:
                _ = prob_a001_coal(1, a1, a2, m, t, t_, n, pops[0], pops[1])
                logP += np.log2(_)
                
                if logP in [np.inf, -np.inf, np.nan]:
                    logP = np.nan
                    break
                
                pops[1] -= 1
            
                
            t = copy.copy(t_)
    
    
        
        if not np.isnan(logP):
            ret.append(logP)
        
    return np.nanmean(ret)

def main():
    args = parse_args()
    
    ifiles = sorted(glob.glob(os.path.join(args.idir, '*.npz')))
    ix = int(args.i)
    
    sample_sizes = tuple(map(int, args.sample_sizes.split(',')))
    s1, s2 = sample_sizes

    ret = np.zeros(len(ifiles))
    trees = np.load(ifiles[ix], allow_pickle = True)
    
    n, a1, a2, m = tuple(trees['loc'])
    ori_params = (n, a1, a2, m)
    
    p = []
    Ap = []
    
    maTs = []
    miTs = []
    for events in trees['E']:
        pop_sizes = copy.copy([s1, s2])
        events = sorted(events, key = lambda u: u[-1])
        
        T = [np.log(u[-1] * 2) for u in events]
        
        maTs.append(np.max(T))
        miTs.append(np.min(T))
    
    kde_ma = gaussian_kde(maTs)
    kde_mi = gaussian_kde(miTs)
    
    mu_ma = np.mean(maTs)
    std_ma = np.std(maTs)
    

    logging.info('getting probabilities for p...')
    for events in trees['E']:
        pop_sizes = copy.copy([s1, s2])
        events = list(sorted(events, key = lambda u: u[-1]))
        
        p_ = compute_P(events, n, a1, a2, m, pop_sizes, kde_ma, kde_mi)

        p.append(p_)
    
    p = np.array(p)
    logging.info('mean ll: {}'.format(np.mean(p)))

    ii = sorted(list(set(list(range(len(ifiles)))).difference([ix])))
    for ij in ii:
        logging.info('getting probabilities for q_i = {}...'.format(ij))
        trees_ = np.load(ifiles[ij], allow_pickle = True)
        
        n, a1, a2, m = tuple(trees_['loc'])
        
        maTs = []
        miTs = []
        for events in trees_['E']:
            pop_sizes = copy.copy([s1, s2])
            events = sorted(events, key = lambda u: u[-1])
            
            T = [np.log(u[-1] * 2) for u in events]
            
            maTs.append(np.max(T))
            miTs.append(np.min(T))
        
        kde_ma = gaussian_kde(maTs)
        kde_mi = gaussian_kde(miTs)
        
        q = []
        for events in trees['E']:
            pop_sizes = copy.copy([s1, s2])
            events = sorted(events, key = lambda u: u[-1])
            
            p_ = compute_P(events, n, a1, a2, m, pop_sizes, kde_ma, kde_mi)

            q.append(p_)
        
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

