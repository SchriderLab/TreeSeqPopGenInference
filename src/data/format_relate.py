# -*- coding: utf-8 -*-
import os
import argparse
import logging

import networkx as nx

import sys
import copy
import h5py
import numpy as np
import glob

from data_functions import load_data
from scipy.stats import skew

from skbio.tree import TreeNode

import matplotlib.pyplot as plt
import random

from ete3 import Tree
import itertools
from scipy.spatial.distance import squareform
import time
from simulate_msprime_grid import make_distance_matrix

"""
notes:
training_results/seln_rnn_i2/ --n_steps 1000 --lr 0.00001 --L 128 --n_gcn_iter 16 --lr_decay 0.98 --pad_l --in_dim 3 --n_classes 5 --n_per_batch 4
training_results/seln_rnn_i3/ --n_steps 1000 --lr 0.00001 --L 92 --n_gcn_iter 32 --lr_decay 0.98 --pad_l --in_dim 3 --n_classes 5 --n_per_batch 4
training_results/seln_rnn_i4/ --n_steps 1000 --lr 0.00001 --L 128 --n_gcn_iter 32 --lr_decay 0.98 --pad_l --in_dim 3 --n_classes 5 --n_per_batch 4"
training_results/seln_rnn_i5/ --n_steps 1000 --lr 0.0001 --L 128 --n_gcn_iter 32 --lr_decay 0.98 --pad_l --in_dim 3 --n_classes 5 --n_per_batch 4"

"""

import gzip

def parse_line(line, s0, s1, ):
    nodes = []
    parents = []
    lengths = []
    n_mutations = []
    regions = []
    
    edges = []
    
    # new tree
    line = line.replace(':', ' ').replace('(', '').replace(')', '').replace('\n', '')
    line = line.split(' ')[:-1]
    
    start_snp = int(line[0])
    
    sk_nodes = dict()
    mut_dict = dict()
    try:
        for j in range(2, len(line), 5):
            nodes.append((j - 1) // 5)
            
            p = int(line[j])
            if p not in sk_nodes.keys():
                sk_nodes[p] = TreeNode(name = str(p))
                
            length = float(line[j + 1])
            
            if (j - 1) // 5 not in sk_nodes.keys():
                sk_nodes[(j - 1) // 5] = TreeNode(name = str((j - 1) // 5), parent = sk_nodes[p], length = length)
                sk_nodes[p].children.append(sk_nodes[(j - 1) // 5])
            else:
                sk_nodes[(j - 1) // 5].parent = sk_nodes[p]
                sk_nodes[(j - 1) // 5].length = length
                sk_nodes[p].children.append(sk_nodes[(j - 1) // 5])
                
            parents.append(p)
            lengths.append(float(line[j + 1]))
            n_mutations.append(float(line[j + 2]))
            
            mut_dict[nodes[-1]] = n_mutations[-1]
            regions.append((int(line[j + 3]), int(line[j + 4])))
            
            edges.append((parents[-1], nodes[-1]))
    except:
        return
    
    lengths.append(0.)
    
    root = None
    for node in sk_nodes.keys():
        node = sk_nodes[node]
        if node.is_root():
            root = node
            break
        
    root = root.children[0]
    T_present = [u for u in root.traverse() if u.is_tip()]
    T_names = sorted([int(u.name) for u in root.traverse() if u.is_tip()])
    
    data = dict()
    
    # pop_labels + mutation
    if s1 > 0:
        for node in T_names[:s0]:
            data[node] = np.array([0., 1., 0., 0., mut_dict[node]])
        
        for node in T_names[s0:s0 + s1]:
            data[node] = np.array([0., 0., 1., 0., mut_dict[node]])
    else:
        for node in T_names:
            data[node] = np.array([0., 1., 0., mut_dict[node]])
    
    edges = []
    while len(T_present) > 0:
        _ = []
        
        for c in T_present:
            c_ = int(c.name)
            branch_l = c.length
            
            p = c.parent
            if p is not None:
            
                p = int(c.parent.name)
                
                if p not in data.keys():
                    d = np.zeros(data[c_].shape)
                    # pop_label
                    d[-2] = 1.
                    # time
                    d[0] = data[c_][0] + branch_l
                    
                    if p in mut_dict.keys():
                        d[-1] = mut_dict[p]

                    data[p] = d
                
                    _.append(c.parent)
            
                edges.append((c_, p))
               
        T_present = copy.copy(_)
        
    X = []
    for node in nodes:
        X.append(data[node])
                            
    X = np.array(X)
    edges = edges[:X.shape[0]]

    return X, edges, lengths, start_snp

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ms_dir", default = "None")
    parser.add_argument("--idir", default = "None")
    
    parser.add_argument("--n_samples", default = "None")
    parser.add_argument("--val_prop", default = "0.1")
    parser.add_argument("--pop_sizes", default = "20,14")
    
    parser.add_argument("--n_sample", default = "None")
    
    parser.add_argument("--topological_order", action = "store_true")
    parser.add_argument("--bidirectional", action = "store_true")

    parser.add_argument("--ofile", default = "None")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()
    
    ofile = h5py.File(args.ofile, 'w')

    ifiles = glob.glob(os.path.join(args.ms_dir, '*.msOut.gz'))
    logging.info('have {} files to parse...'.format(len(ifiles)))
    
    tags = [u.split('/')[-1].split('.')[0] for u in ifiles]
    pop_sizes = list(map(int, args.pop_sizes.split(',')))
    
    s0, s1 = pop_sizes
    
    for ii in range(len(ifiles)):
        tag = tags[ii]
        ifile = ifiles[ii]
        print(tag)
        
        anc_file = sorted([os.path.join(args.idir, u) for u in os.listdir(args.idir) if (u.split('.')[-1] == 'gz')])[0]
        anc_file = gzip.open(anc_file, 'r')
        
        # load the genotype matrices that correspond to the trees
        logging.info('reading data, {}...'.format(ifile))
        x, y, p, params = load_data(ifile, None)
        
        del y
        
        times = []
        
        ix = 0
        
        logging.info('writing...')
        while True:
            # we're at the beginning of a block
            for k in range(3):
                line = anc_file.readline()
            
            if not '(' in line.decode('utf-8'):
                break
            
            current_day_nodes = list(range(sum(pop_sizes)))
                                        
            Xs = []
            Edges = []
            infos = []
            
            As = []
            Ds = []

            snps = []

            lines = []            
            while '(' in line.decode('utf-8'):
                lines.append(line.decode('utf-8'))
                line = anc_file.readline()
                
            iix = int(line.decode('utf-8').replace('\n', '').split()[-1]) - 1
            l = x[iix].shape[1]
            
            t0 = time.time()
            for ij in range(len(lines)):
                line = lines[ij]
                X, edges, lengths, start_snp = parse_line(line, s0, s1)
                
                if ij == len(lines) - 1:
                    end_snp = x[iix].shape[1]
                
                
                else:
                    next_line = lines[ij + 1]
                    next_line = next_line.replace(':', ' ').replace('(', '').replace(')', '').replace('\n', '')
                    next_line = next_line.split(' ')[:-1]
                
                    end_snp = int(next_line[0])
                    
                l_ = end_snp - start_snp
                    
                Xs.append(X)
                
                ii = list(np.where(X[:,0] > 0)[0])
                times.extend(X[ii,0])
                
                t_coal = np.max(X[:,0])
                mean_time = np.mean(X[ii,0])
                std_time = np.std(X[ii,0])
                median_time = np.median(X[ii,0])
                
                mean_branch_length = np.mean(lengths)
                median_branch_length = np.median(lengths)
                std_branch_length = np.std(lengths)
                skew_branch_length = skew(lengths)
                max_branch_length = np.max(lengths)
                position = ((start_snp + end_snp) / 2.) / l
                w = l_ / l
                
                info_vec = np.array([t_coal, mean_time, std_time, median_time, mean_branch_length, median_branch_length, std_branch_length, skew_branch_length, max_branch_length,
                                     position, w, l])
                
                # make edges bi-directional
                if args.bidirectional:
                    edges = edges + [(v,u) for u,v in edges]
                edges = np.array(edges).T
                
                Edges.append(edges)
                infos.append(info_vec)
            
            logging.info('iteration took {} seconds...'.format(time.time() - t0))
            infos = np.array(infos)
            global_vec = np.array(list(np.mean(infos, axis = 0)) + list(np.std(infos, axis = 0)) + list(np.median(infos, axis = 0)) + [infos.shape[0]], dtype = np.float32)
            
            if len(Xs) > 0:
                
                ofile.create_dataset('{1}/{0}/global_vec'.format(ix, tag), data = global_vec, compression = 'lzf')
                ofile.create_dataset('{1}/{0}/x'.format(ix, tag), data = np.array(Xs), compression = 'lzf')
                ofile.create_dataset('{1}/{0}/edge_index'.format(ix, tag), data = np.array(Edges).astype(np.int32), compression = 'lzf')
                ofile.create_dataset('{1}/{0}/info'.format(ix, tag), data = np.array(infos), compression = 'lzf')
                
            
            Xg = x[iix]
            #A = np.array(As)
            y = params[iix]
            
            ofile.create_dataset('{1}/{0}/x_0'.format(ix, tag), data = Xg.astype(np.uint8), compression = 'lzf')
            ofile.create_dataset('{1}/{0}/y'.format(ix, tag), data = np.array([y]), compression = 'lzf')
            ofile.flush()

            ix += 1
                
          
    ofile.close()
    
    # ${code_blocks}

if __name__ == '__main__':
    main()