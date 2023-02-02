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

# constants a0, a1, m
# N1 and N2 determined via the time of the last event
def F_a001_mig(a0, a1, m, N1, N2, t):
    ret = 1 - np.exp(m * (N2 / N1) * (np.exp(t * (a0 - a1)) - 1) / (a1 - a0))
    
    return ret

def F_mig(alpha, m, N_div, t):
    ret = np.ones(alpha.shape)
    i, j = np.where(alpha[0] != 0)
    
    _ = m[:,i,j] * (N_div[:,i,j]) * (np.exp(t[:,0] * alpha[:,i,j] * -1) - 1) / (alpha[:,i,j])
    ret[:, i, j] = 1 - np.exp(_)
    
    i, j = np.where(alpha[0] == 0 & ~np.eye(alpha[0].shape[0],dtype=bool))

    if len(i) > 0:
        ret[:,i,j] = 1 - np.exp(-m[:,i,j] * t[:,0])
    
    ret[:,range(ret.shape[1]),range(ret.shape[1])] = 1.
    
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

# --- replicate
# events = (|e|, 4 + n) array with 
# (coal or mig (0 or 1 respectively), 
### index of pop from (if mig) pop where coal if not, 
### index of pop mig to (0 if coal)
### time in generation of event)
### the n internal pop_sizes when each event occurs
# --------
# --- priors
# N = (n, ) vector of effective population sizes
# alpha = (n,) vector of population growth rates
# m = (n, n) matrix of migration rates (>= 0, asymmetric, diagonal is zero)

def compute_P(events, N, alpha, m):
    T = np.array([u[3] for u in events])
    
    n_pops = m.shape[0]
    n_events = len(T)
    
    t0 = np.array([0.] + list(T))
    t = np.diff(t0).astype(np.float64)
    
    N_t = np.tile(N, (n_events, 1)).astype(np.float64)
    mu = np.exp(np.tile(t0[:-1].reshape(-1, 1), (1, n_pops)) * alpha.reshape(1, -1) * -1).astype(np.float64)
    
    # effective size of populations at each time t0 (|e|, n)
    N_t = N_t * mu
    t_mat = np.tile(t.reshape(-1, 1), (1, n_pops)) # |e|, n time matrix 
    alpha_mat = np.tile(alpha.reshape(1, -1), (n_events, 1)).astype(np.float64) # |e|, n matrix of alpha coefficients
    
    ### Coalesence ###
    ### --------------- ###
    ## non-coal pops
    P_coal = 1 - F_a001_coal(alpha_mat, N_t, t_mat)
    P_coal_ = 1 - P_coal
    
    ii_coal = np.where(events[:,0] == 0)
    ii_pop = events[ii_coal,1].astype(np.int32)
    s = events[:,4:].astype(np.int32)
    s = nC2(s)
    s[ii_coal, ii_pop] -= 1
    
    P_coal = np.power(P_coal, s)
    P_coal[ii_coal, ii_pop] *= P_coal_[ii_coal, ii_pop]
    

    #p = np.sum(np.log2(_) * s, axis = 1)
    
    ### Migration ###
    # alpha_0, alpha_1, ...
    # alpha_0, alpha_1, ...
    alpha_mat = np.tile(alpha.reshape(1, -1), (n_pops, 1)).astype(np.float64)
    alpha_mat = np.tile(np.expand_dims(alpha_mat - alpha_mat.T, 2), (1, 1, n_events)).transpose(2, 0, 1).astype(np.float64)
    
    # 0, alpha_1 - alpha_0, alpha_2 - alpha_0
    # repeated for |e| on the first axis
    
    # ----
    # |e|, n repeated n times on the last axis    
    N_mat = np.tile(np.expand_dims(N_t, 2), (1, 1, n_pops)).astype(np.float32)
    # 1, N_1 / N_0, N_2 / N_0 @ t = t_0, t_1, ...
    # ...
    N_mat = N_mat / N_mat.transpose(0, 2, 1).astype(np.float32)
    
    # |e|, n, n migration rate repeated over event times
    m_mat = np.tile(np.expand_dims(m, 2), (1, 1, n_events)).transpose(2, 0, 1)
    t_mat = t.reshape(1, 1, -1).transpose(2, 0, 1)

    # (|e|,n,n) matrix of probability of migration happening at each event
    P_mig = 1 - F_mig(alpha_mat, m_mat, N_mat, t_mat)
    P_mig_ = 1 - P_mig
    
    s = events[:,4:]
    ii_mig = np.where(events[:,0] == 1)
    # from-to indices of migration
    ij_mig = events[ii_mig,1:3].astype(np.int32)[0]
    s[ii_mig,ij_mig[:,0]] -= 1
    
    i, j, k = np.where(m_mat > 0.)
    
    P_mig[i, j, k] = np.power(P_mig[i, j, k], s[i,j])
    
    P_mig[ii_mig,ij_mig[:,0],ij_mig[:,1]] *= P_mig_[ii_mig,ij_mig[:,0],ij_mig[:,1]]
    
    i, j, k = np.where(m_mat > 0.)
    P_mig = P_mig[i, j, k].reshape(-1, 1)
    P = np.concatenate([P_coal, P_mig], axis = 1)
    P = np.product(P, axis = 1)
    P = np.log2(P)
    
    # sum accross possibilties
    p = np.sum(P)
    
    return p

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
    ix = int(args.i)

    ret = np.zeros(len(ifiles))
    trees = np.load(ifiles[ix], allow_pickle = True)
    
    n, a1, a2, m = tuple(trees['loc'])
    ori_params = (n, a1, a2, m)
    
    p = []

    logging.info('getting probabilities for p...')
    M = np.array([[0., m], [0.,0.]])
    N = np.array([n, n])
    alpha = np.array([a1, a2])
    
    for events in trees['E']:
        p.append(compute_P(events, N, alpha, M))

    logging.info('mean ll: {}'.format(np.mean(p)))

    ii = sorted(list(set(list(range(len(ifiles)))).difference([ix])))
    for ij in ii:
        logging.info('getting probabilities for q_i = {}...'.format(ij))
        trees_ = np.load(ifiles[ij], allow_pickle = True)
        
        n, a1, a2, m = tuple(trees_['loc'])
        print(ori_params, (n, a1, a2, m))
        q = []
        M = np.array([[0., m], [0.,0.]])
        N = np.array([n, n])
        alpha = np.array([a1, a2])
        
        for events in trees['E']:
            q.append(compute_P(events, N, alpha, M))
            
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
