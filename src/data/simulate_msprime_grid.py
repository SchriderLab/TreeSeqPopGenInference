# -*- coding: utf-8 -*-
import os
import argparse
import logging

import msprime
import numpy as np
import itertools
import scipy.integrate as integrate

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

import copy
import random
import pickle

from io import StringIO
from skbio import read
from skbio.tree import TreeNode
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import squareform

import sys
sys.path.append('src/models/')

import matplotlib

from compute_kl_matrix_vectorized import DemographyLL, MigExponentialRateFunction, CoalExponentialRateFunction, setup_ll_inputs
cmap = matplotlib.cm.get_cmap('viridis')

import cv2

#cdf = pickle.load(open('n01_w_stats.pkl', 'rb'))['cdf']

def make_FW_rep(root, sample_sizes):
    if len(sample_sizes > 1):
        topo_tips = [u for u in root.levelorder() if u.is_tip()]
        topo_ids = [u.id for u in topo_tips]
        pop_vectors = [[u.pop for u in topo_tips]]
        
        ordered_nodes = sorted([u for u in root.levelorder() if not u.is_tip()], key = lambda x: x.age)
        
        for n in ordered_nodes:
            c1, c2 = n.children
            
            i1 = topo_ids.index(c1.id)
            i2 = topo_ids.index(c2.id)
            
            topo_ids[i1] = n.id
            del topo_ids[i2]
            
            _ = copy.copy(pop_vectors[-1])
            _[i1] = (_[i1] + _[i2]) / 2.
            del _[i2]
            
            pop_vectors.append(_)
        
        pop_matrix = np.zeros((len(topo_tips), len(topo_tips)))
        for k in range(len(pop_vectors)):
            i, j = np.diag_indices(len(topo_tips) - k)
            j = copy.copy(j) + k
            
            pop_matrix[i, j] = pop_vectors[k]
            
        i, j = np.triu_indices(pop_matrix.shape[0])
        pop_matrix = np.squeeze(pop_matrix[i, j])
    else:
        pop_matrix = None

    non_zero_ages = []
    
    ages = np.zeros(root.count())
    for node in root.traverse():
        ages[node.id] = node.age
        
        if node.age > 0.:
            non_zero_ages.append(node.age)
            
    non_zero_ages = sorted(non_zero_ages, reverse = True)

    # indexed by assinged id
    F = np.zeros((sum(sample_sizes) - 1, sum(sample_sizes) - 1))

    children = root.children
    c1, c2 = root.children
    
    extant = np.array(range(2, sum(sample_sizes) + 1))
    start_end = []
    
    todo = [root]
    while len(todo) != 0:
        root = todo[-1]
        del todo[-1]
        
        t_coal = ages[root.id]
        
        if root.has_children():
            c1, c2 = root.children
            
            start_end.append((t_coal, c1.age))
            
            if not c1.is_tip():
                todo.append(c1)
                
            start_end.append((t_coal, c2.age))
            
            if not c2.is_tip():
                todo.append(c2)

    start_end = np.array(start_end)
    s = np.array(non_zero_ages + [0.])
    
    F[list(range(F.shape[0])), list(range(F.shape[0]))] = extant
    
    i, j = np.tril_indices(F.shape[0], -1)
    ij = zip(i, j)
    
    for i, j in ij:
        start = s[j]
        end = s[i]

        F[i, j] = sum((start_end[:,1] <= end) & (start_end[:,0] >= start))
        F[j, i] = F[i, j]
    
    i, j = np.tril_indices(F.shape[0])

    W = s[j] - s[i + 1]

    """
    print(np.min(W))
    print(np.max(W))
    print(np.min(cdf.x), np.max(cdf.x))    
    W = cdf(W)
    
    F /= np.max(F)
        
    im = np.zeros(pop_matrix.shape + (3,))
    im[:,:,0] = pop_matrix
    im[:-1,:-1,1] = W
    im[1:,-1,1] = W[:,-1]
    
    im[:-1,:-1,2] = F
    im[-1,:-1,2] = F[-1]
    im[1:,-1,2] = F[:,-1]
    
    plt.imshow(im)
    plt.show()
    """
    
    """
    plt.scatter(F.flatten(), W.flatten())
    plt.show()   

    angle = (np.arctan2(W, F) + np.pi / 2.) / np.pi
    plt.imshow(angle)
    plt.colorbar()
    plt.show()
    
    mag = np.sqrt(F ** 2 + W ** 2)
    plt.imshow(mag)
    plt.show()
    
    mag = mag / np.max(mag)
    
    im = np.zeros(F.shape + (3,))
    im[:,:,0] = angle
    im[:,:,1] = 0.2
    im[:,:,2] = mag

    plt.imshow(im)
    plt.show()
    
    im = (im * 255).astype(np.uint8)
    
    im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
    
    plt.imshow(im[:,:,[2, 0, 1]])
    plt.show()    
    """
    
    return F, W, pop_matrix

def make_distance_matrix(root, sample_sizes):
    
    # include pop_labels
    if len(sample_sizes) > 1:
        ii_topological = [root.id] + [u.id for u in root.levelorder() if u.pop == -1] + [u.id for u in root.levelorder() if u.pop == 0] \
                        + [u.id for u in root.levelorder() if u.pop == 1]
    else:
        ii_topological = [u.id for u in root.levelorder()]
    
    ages = np.zeros(root.count())
    for node in root.traverse():
        ages[node.id] = node.age
        
    # indexed by assinged id
    D = np.zeros((2 * sum(sample_sizes) - 1, 2 * sum(sample_sizes) - 1))
    
    children = root.children
    c1, c2 = root.children
    
    todo = [root]
    while len(todo) != 0:
        root = todo[-1]
        del todo[-1]
        
        t_coal = ages[root.id]
        d_root = t_coal - ages
        
        if root.has_children():
            c1, c2 = root.children
            i1 = c1.id
            i2 = c2.id
            
            if c1.has_children():
                todo.append(c1)
            if c2.has_children():
                todo.append(c2)
            
            ii1 = [u.id for u in list(c1.traverse())]
            ii2 = [u.id for u in list(c2.traverse())]

            # n x m distance matrix for the descendants
            # of left and right child
            d1 = np.tile(d_root[ii1].reshape(-1, 1), (1, len(ii2))) 
            d2 = np.tile(d_root[ii2].reshape(1, -1), (len(ii1), 1))

            d = d1 + d2
            
            D[root.id,ii1 + ii2] = d_root[ii1 + ii2]
            D[ii1 + ii2,root.id] = d_root[ii1 + ii2]
            
            D[np.ix_(ii1, ii2)] = d
            D[np.ix_(ii2, ii1)] = d.T
    
    D = D[np.ix_(ii_topological, ii_topological)]
    
    return D

class NPopSimulator(object):
    # s is the vector of integer samples sizes in the present day
    # N is the vector of effective population sizes
    # alpha is the vector of exponential growth rates
    # m is the migration matrix (>= 0)
    # T_alpha is the time in generations that growth or decay concludes for a population
    def __init__(self, s, N, alpha, m, T_alpha = None, ploidy = 1):
        self.n_pop = s.shape[0]
        
        self.s = s
        self.N = N
        self.alpha = alpha
        self.T_alpha = T_alpha
        
        self.demography = msprime.Demography()
        self.samples = []
        for ix in range(self.n_pop):
            self.demography.add_population(name = 'pop_{0:03d}'.format(ix), initial_size = N[ix])
            self.demography.add_population_parameters_change(0., population = 'pop_{0:03d}'.format(ix), growth_rate = alpha[ix])
            
            if T_alpha is not None:
                if T_alpha[ix] != np.inf:
                    self.demography.add_population_parameters_change(T_alpha[ix], population = 'pop_{0:03d}'.format(ix), growth_rate = 0.)
        
            self.samples.append(msprime.SampleSet(s[ix], population = 'pop_{0:03d}'.format(ix), ploidy = ploidy))
        
        self.m = m
        self.i, self.j = np.where(m > 0)
        
        self.ij = list(zip(self.i, self.j))
        
        self.n_nonzero = len(self.i)
        for ix in range(self.n_nonzero):
            self.demography.set_migration_rate('pop_{0:03d}'.format(self.i[ix]), 'pop_{0:03d}'.format(self.j[ix]), self.m[self.i[ix], self.j[ix]])
            
    def simulate(self, sequence_length = 1000, mu = 0):
        ret = msprime.sim_ancestry(samples = self.samples, recombination_rate = 0., 
                                 sequence_length = sequence_length, demography = self.demography, record_migrations = True)
        
        if mu > 0:
            ret = msprime.sim_mutations(ret, mu)
        
            X = ret.genotype_matrix()
        else:
            X = None
        
        tables = ret.dump_tables()
        tables.sort()
        
        ages = tables.nodes.time
        pops = tables.nodes.population
        individuals = list(range(len(pops)))
        
        edges = np.array([tables.edges.parent, tables.edges.child], dtype = np.int32).T
        
        start = list(np.where(ages > 0)[0])[0]
        coals = []

        t = 0.        
        # get the coalescence events
        for ij in range(start, len(individuals)):
            i = individuals[ij]
            t1 = ages[ij]
            pop = pops[ij]
            
            e = edges[np.where(edges[:,0] == i)[0]]
            c1 = e[0,1] - 1
            c2 = e[1,1] - 1
            
            coals.append((0, pop, 0, t1))
            
            t = copy.copy(t1)
            
        migs = []
        
        
        # get the migration events
        
        
        time = tables.migrations.time
        node = tables.migrations.node
        src = tables.migrations.source
        dest = tables.migrations.dest

        for ij in range(len(time)):
            t = time[ij]
            i = node[ij]
            
            migs.append((1, src[ij], dest[ij], t))
            
        events = sorted(migs + coals, key = lambda u: u[3])

        count_vector = np.zeros(self.n_pop + self.n_nonzero)

        # append the population sizes in the event list for easier prob calculation later
        pops = list(self.s)
        for ix, e in enumerate(events): # the counts of the Poisson process at each time of event
            events[ix] = events[ix] + tuple(pops)
            
            ii, i, j = e[:3]
            
            if ii == 0:
                pops[i] -= 1
            
                count_vector[i] += 1
            else:
                pops[j] += 1
                pops[i] -= 1
                
                count_vector[self.n_pop + self.ij.index((i, j))] += 1
            
            # the observed counts of the Poisson processes at
            # each time of observation
            events[ix] = events[ix] + tuple(count_vector)
        
        events = np.array(events)
        
        return ret, events, X
        
import sys

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "src/json/n_001.json")
    parser.add_argument("--n_replicates", default = "2500")
    
    parser.add_argument("--pop_labels", action = "store_true")

    parser.add_argument("--ofile", default = "None")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    # ${odir_del_block}

    return args

import pickle
import json

def main():
    args = parse_args()
    
    ifile = json.load(open(args.ifile))
    
    N = np.array(ifile['N'])
    M = np.array(ifile['M']).reshape(N.shape[0], N.shape[0])
    sample_sizes = np.array(ifile['s'])
    alpha = np.array(ifile['alpha'])
    
    simulator = NPopSimulator(sample_sizes, N, alpha, M)
    if alpha.shape[0] > 1:
        ll = DemographyLL(CoalExponentialRateFunction(alpha, N), MigExponentialRateFunction(alpha, N, M))
    else:
        ll = DemographyLL(CoalExponentialRateFunction(alpha, N), None)
        
    E = []
    Ds = []
    P = []
    Xs = []
    Fs = []
    Ws = []
    pop_mats = []
    
    for j in range(int(args.n_replicates)):
        s, events, X = simulator.simulate(mu = 1e-5)
        X[X > 0] = 1.
        
        Xs.append(np.sum(X.T, axis = 0))
        
        tables = s.dump_tables()
        tables.sort()
        
        t = list(s.aslist())[0]
        
        f = StringIO(t.as_newick())  
        root = read(f, format="newick", into=TreeNode)
        root.assign_ids()
                
        children = root.children
        t = max(tables.nodes.time)
        
        fm = lambda x: int(x.replace('n', '')) - 1
        
        root.age = t
        populations = list(tables.nodes.population)

        while len(children) > 0:
            _ = []
            for c in children:
                
                c.age = c.parent.age - c.length
                if c.is_tip():
                    c.pop = populations[fm(c.name)]
                    c.age = 0.
                else:
                    c.pop = -1
                    
                _.extend(c.children)
                
            children = copy.copy(_)

        D = make_distance_matrix(root, sample_sizes)    
        F, W, pop_matrix = make_FW_rep(root, sample_sizes)

        Fs.append(F)
        Ws.append(W)
        if pop_matrix is not None:
            pop_mats.append(pop_matrix)
        
        Ds.append(squareform(D))
        
        E.append(events)
        
        P.append(ll(*setup_ll_inputs(events, N, alpha, M)).item())
        print('ll = {}'.format(P[-1]))
    
    # distance matrices
    D = np.array(Ds)
    # log-likelihood
    P = np.array(P)

    Dmean = np.mean(D)
    Dstd = np.std(D)
        
    # now saving the SFS
    np.savez_compressed(args.ofile, E = np.array(E, dtype = object), F = np.array(Fs, dtype = np.uint16),
                        D = D, stats = np.array([Dmean, Dstd]), P = P, X = np.array(Xs), W = np.array(Ws))
    
    
        
     
    """
    pop_sizes = [s1, s2]
    
    logP = 0.
    
    t = 0.
    for e in events:
        t_ = e[1]
        
        if len(e) == 2:
            p = prob_a001_mig(a1, a2, m, t, t_, n, pop_sizes[0], pop_sizes[1])
            logP += np.log(p)

            if p < 0:
                print(p)
                print(a1, a2, m, t, t_, n, pop_sizes[0], pop_sizes[1])
            pop_sizes[0] -= 1
            pop_sizes[1] += 1
        elif e[0] == 0:
            p = prob_a001_coal(0, a1, a2, m, t, t_, n, pop_sizes[0], pop_sizes[1])
            logP += np.log(p)
  
            if p < 0:
                print(p)
                print(0, a1, a2, m, t, t_, n, pop_sizes[0], pop_sizes[1])
            pop_sizes[0] -= 1
        else:

            p = prob_a001_coal(1, a1, a2, m, t, t_, n, pop_sizes[0], pop_sizes[1])
            logP += np.log(p)

            if p < 0:
                print(p)
                print(1, a1, a2, m, t, t_, n, pop_sizes[0], pop_sizes[1])
            
            pop_sizes[1] -= 1
            
        t = copy.copy(t_)
    
    print(logP)
    """

    # ${code_blocks}

if __name__ == '__main__':
    main()

