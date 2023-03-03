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
    parser.add_argument("--sample_sizes", default = "104")
    
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
    ofile_val = h5py.File('/'.join(args.ofile.split('/')[:-1]) + '/' + args.ofile.split('/')[-1].split('.')[0] + '_val.hdf5', 'w')
    
    ifiles = glob.glob(os.path.join(args.ms_dir, '*.msOut.gz'))
    logging.info('have {} files to parse...'.format(len(ifiles)))
    
    tags = [u.split('/')[-1].split('.')[0] for u in ifiles]
    sample_sizes = list(map(int, args.sample_sizes.split(',')))
    
    for ii in range(len(ifiles)):
        tag = tags[ii]
        ifile = ifiles[ii]
        
        anc_files = sorted([os.path.join(args.idir, u) for u in os.listdir(args.idir) if (u.split('.')[-1] == 'anc' and tag == u.split('_')[0])])
        if len(anc_files) == 0:
            logging.info('ERROR: have no matching .anc files for {}...'.format(ifile))
            continue
        else:
            logging.info('have {} anc files...'.format(len(anc_files)))
        
        # load the genotype matrices that correspond to the trees
        logging.info('reading data, {}...'.format(ifile))
        x, y, p, intros = load_data(ifile, None)
        if args.n_samples == "None":
            N = len(anc_files)
        else:
            N = int(args.n_samples)
            
        N_val = int(N * float(args.val_prop))
        N = N - N_val
        
        logging.info('have {0} training and {1} validation replicates...'.format(N, N_val))
        logging.info('writing...')
        for ix in range(N + N_val):
            if (ix + 1) % 5 == 0:
                logging.info('on replicate {}...'.format(ix))
            
            if not os.path.exists(anc_files[ix].replace('.anc', '.mut')):
                logging.info('ERROR: {} has no matching .mut file!...'.format(anc_files[ix]))
                continue
            
            anc_file = open(anc_files[ix], 'r')
                            
            lines = anc_file.readlines()[2:]
            
            l = x[int(anc_files[ix].split('/')[-1].split('.')[0].split('chr')[-1]) - 1].shape[1]
                   
            Ds = []
            
            t0 = time.time()
            for ij in range(len(lines)):
                line = lines[ij]
                
                nodes = []
                parents = []
                lengths = []
                n_mutations = []
                regions = []
                
                edges = []
                
                # new tree
                line = line.replace(':', ' ').replace('(', '').replace(')', '').replace('\n', '')
                line = line.split(' ')[:-1]
                
                try:
                    if ij != len(lines) - 1:
                        next_line = lines[ij + 1]
                        next_line = next_line.replace(':', ' ').replace('(', '').replace(')', '').replace('\n', '')
                        next_line = next_line.split(' ')[:-1]
                    
                        end_snp = int(next_line[0])
                    else:
                        end_snp = x[int(anc_files[ix].split('/')[-1].split('.')[0].split('chr')[-1]) - 1].shape[1]
                except:
                    break
    
                start_snp = int(line[0])
                l_ = end_snp - start_snp
                
                sk_nodes = dict()
                mut_dict = dict()
                try:
                    for j in range(2, len(line), 5):
                        nodes.append((j - 1) // 5)
                        
                        p = int(line[j])
                        if p not in sk_nodes.keys():
                            sk_nodes[p] = TreeNode(name = p)
                            
                        length = float(line[j + 1])
                        
                        if (j - 1) // 5 not in sk_nodes.keys():
                            sk_nodes[(j - 1) // 5] = TreeNode(name = (j - 1) // 5, parent = sk_nodes[p], length = length)
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
                    break
                
                nodes.append(-1)
                master_nodes = copy.copy(nodes)
                
                lengths.append(0.)
                
                root = None
                for node in sk_nodes.keys():
                    node = sk_nodes[node]
                    if node.is_root() and hasattr(node, 'children'):
                        print(node.children)
                        if len(node.children) == 2:
                            root = node
                            break
                    
                D_ = root.tip_tip_distances().data
                root.age = np.max(D_) / 2.
                children = root.children
                
                while len(children) > 0:
                    _ = []
                    for c in children:
                        c.age = c.parent.age - c.length
                    
                        _.extend(c.children)
                    
                    children = copy.copy(_)
                    
                print(children)
                print(root.count())
                
                D = make_distance_matrix(root, sample_sizes)
                Ds.append(squareform(D))
                
            if len(Ds) > 0:
                if ix < N:
                    ofile.create_dataset('{1}/{0}/D'.format(ix, tag), data = np.array(Ds), compression = 'lzf')
                    ofile.flush()
                else:
                    ofile_val.create_dataset('{1}/{0}/D'.format(ix - N, tag), data = np.array(Ds), compression = 'lzf')
                    ofile_val.flush()
    
    ofile.close()
    ofile_val.close()

if __name__ == '__main__':
    main()