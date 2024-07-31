# -*- coding: utf-8 -*-
import os
import argparse
import logging

import pandas as pd
from data_functions import load_data

import glob
import tsinfer

from format_relate import parse_line
from scipy.stats import skew

import sys
import h5py
import numpy as np
# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")

    parser.add_argument("--r", default = "1e-7")
    parser.add_argument("--L", default = "20000")
    
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
    
    ifiles = glob.glob(os.path.join(args.idir, '*.msOut.gz'))
    L = int(args.L)
    r = float(args.r)
    
    ofile = h5py.File(args.ofile, 'w')
    
    print(ifiles)
    for ifile in ifiles:
        print('working on {}...'.format(ifile))
        sys.stdout.flush()
        
        tag = ifile.split('/')[-1].split('.')[0]
        
        
        X, Y, P, params = load_data(ifile)

        ix_ = 0
        for x, p, y in zip(X, P, params):
            p = (L * p).astype(np.int32)
            
            
            print(p)
            print(x.shape)
            
            # x (samples, sites)
            with tsinfer.SampleData(sequence_length=L) as sample_data:
                for k in range(x.shape[1]):
                    try:
                        sample_data.add_site(p[k], x[:,k], ["A", "T"], ancestral_allele = 0)
                    except Exception as e:
                        print(e)
                        pass
            
            inferred_ts = tsinfer.infer(sample_data, recombination_rate = r)
            inferred_ts = inferred_ts.simplify(keep_unary = False)
            
            trees = inferred_ts.aslist()
            
            Edges = []
            Xs = [] # node features
            X1 = [] # tree summary features

            l = x.shape[1]
            
            start_snp = 0
            current_node = 0
            for tree in trees:
                
                if tree.has_single_root:
                    X_ = []
                    
                    tree = tree.split_polytomies()
                    nodes = list(tree.nodes())
                    
                    mutation_ix = [nodes.index(u.node) for u in list(tree.mutations())]
                    
                    leaf_nodes = []
                    lengths = []
                    
                    sites = list(tree.sites())

                    e = []
                    for ix, n in enumerate(nodes):
                        c = tree.children(n)
                        lengths.append(tree.branch_length(n))
                   
                        # child to parent
                        if len(c) > 0:    
                            e.extend([(nodes.index(u), ix) for u in c])
                        else:
                            leaf_nodes.append(n)
                    
                        t = tree.time(n)
                        if t == 0:
                            X_.append([0., 1., 0., 0.])
                        else:
                            X_.append([t, 0., 1., 0.])
                        

                    X_ = np.array(X_)
                    X_[mutation_ix, -1] += 1.
                    e = np.array(e, dtype = np.int32) + current_node
                    
                    current_node += X_.shape[0]

                    Xs.append(X_)
                    
                    t_coal = np.max(X_[:,0])
                    ii = list(np.where(X_[:,0] > 0)[0])
                    
                    mean_time = np.mean(X_[ii,0])
                    std_time = np.std(X_[ii,0])
                    median_time = np.median(X_[ii,0])
                    
                    mean_branch_length = np.mean(lengths)
                    median_branch_length = np.median(lengths)
                    std_branch_length = np.std(lengths)
                    skew_branch_length = skew(lengths)
                    max_branch_length = np.max(lengths)
                    l_ = len(sites)
                    
                    w = l_ / l
                    position = (l_ * 0.5 + start_snp) / l
                    start_snp += l_

                    info_vec = np.array([t_coal, mean_time, std_time, median_time, mean_branch_length, median_branch_length, std_branch_length, skew_branch_length, max_branch_length,
                                         position, w, l])
                    
                    X1.append(info_vec)
                    Edges.append(e)

            #logging.info('iteration took {} seconds...'.format(time.time() - t0))
            infos = np.array(X1)
            global_vec = np.array(list(np.mean(infos, axis = 0)) + list(np.std(infos, axis = 0)) + list(np.median(infos, axis = 0)) + [infos.shape[0]], dtype = np.float32)
            if len(Xs) > 0:
                ofile.create_dataset('{1}/{0}/global_vec'.format(ix_, tag), data = global_vec, compression = 'lzf')
                ofile.create_dataset('{1}/{0}/x'.format(ix_, tag), data = np.array(Xs), compression = 'lzf')
                ofile.create_dataset('{1}/{0}/edge_index'.format(ix_, tag), data = np.array(Edges).astype(np.int32), compression = 'lzf')
                ofile.create_dataset('{1}/{0}/info'.format(ix_, tag), data = np.array(infos), compression = 'lzf')
                #ofile['{1}/{0}'.format(ix, tag)].attrs['cmd'] = cmd_line
            Xg = x
            
            ofile.create_dataset('{1}/{0}/x_0'.format(ix_, tag), data = Xg.astype(np.uint8), compression = 'lzf')
            ofile.create_dataset('{1}/{0}/y'.format(ix_, tag), data = np.array([y]), compression = 'lzf')
            
            ix_ += 1
            ofile.flush()
            sys.stdout.flush()
            
    ofile.close()
            
if __name__ == '__main__':
    main()
    