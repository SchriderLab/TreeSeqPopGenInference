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
# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--ofile", default = "None")
    parser.add_argument("--classes", default = "soft,hard,neutral")

    parser.add_argument("--n_sample", default = "250")
    parser.add_argument("--sample_sizes", default = "104")

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
    
    anc_files = sorted(glob.glob(os.path.join(args.idir, '*.anc')))
    random.shuffle(anc_files)
    
    ofile = h5py.File(args.ofile, 'w')
    
    sample_sizes = list(map(int, args.sample_sizes.split(',')))
    classes = args.classes.split(',')
    counter = dict()
    
    for c in classes:
        counter[c] = 0
        
    min_log = np.inf
    max_log = -np.inf
    
    for ix in range(len(anc_files)):
        if (ix + 1) % 5 == 0:
            logging.info('on replicate {}...'.format(ix))
        
        if not os.path.exists(anc_files[ix].replace('.anc', '.mut')):
            logging.info('ERROR: {} has no matching .mut file!...'.format(anc_files[ix]))
            continue
        
        anc_file = open(anc_files[ix], 'r')
        tag = anc_files[ix].split('/')[-1].split('_')[0]
        
        lines = anc_file.readlines()[2:]
                       
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
                    root = node.children[0]
                
            root.assign_ids()
                
            D_ = root.tip_tip_distances().data
            root.age = np.max(D_) / 2.
            children = root.children
            
            while len(children) > 0:
                _ = []
                for c in children:
                    c.age = c.parent.age - c.length
                
                    _.extend(c.children)
                
                children = copy.copy(_)
                                
            D = make_distance_matrix(root, sample_sizes)
            Ds.append(squareform(D))
            
        D = np.array(Ds)
        logD = np.log(D)
        
        if np.max(logD) > max_log:
            max_log = np.max(logD)
            
        if np.min(logD) < min_log:
            min_log = np.min(logD)
        
        print('have min and max of: {}, {}'.format(min_log, max_log))
        ofile.create_dataset('{0}/{1}/D'.format(tag, counter[tag]), data = D, compression = 'lzf')
        ofile.flush()
        
        counter[tag] += 1
        
    ofile.create_dataset('max_min', data = np.array([min_log, max_log]))
    ofile.close()

    # ${code_blocks}

if __name__ == '__main__':
    main()

