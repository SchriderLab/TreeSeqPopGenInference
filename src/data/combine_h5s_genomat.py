# -*- coding: utf-8 -*-
import os
import argparse
import logging

from seriate import seriate
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.optimize import linear_sum_assignment

import numpy as np

import h5py
from mpi4py import MPI
from collections import deque

def format_matrix(x, pop_sizes = (20, 14), out_shape = (2, 32, 128), metric = 'cosine', mode = 'seriate_match'):
    s0, s1 = pop_sizes
    n_pops, n_ind, n_sites = out_shape
    
    if x.shape[0] != s0 + s1:
        return None
    
    if mode == 'seriate_match':
        x0 = x[:s0,:]
        x1 = x[s0:s0 + s1,:]
        
        if s0 != n_ind:
            ii = np.random.choice(range(s0), n_ind)
            x0 = x0[ii,:]
        
        if s1 != n_ind:
            ii = np.random.choice(range(s1), n_ind)
            x1 = x1[ii,:]
    
        if x0.shape[1] > n_sites:
            ii = np.random.choice(range(x0.shape[1] - n_sites))
            
            x0 = x0[:,ii:ii + n_sites]
            x1 = x1[:,ii:ii + n_sites]
            
        else:
            to_pad = n_sites - x0.shape[1]
        
            if to_pad % 2 == 0:
                x0 = np.pad(x0, ((0,0), (to_pad // 2), (to_pad // 2)))
                x1 = np.pad(x1, ((0,0), (to_pad // 2), (to_pad // 2)))
            else:
                x0 = np.pad(x0, ((0,0), (to_pad // 2 + 1), (to_pad // 2)))
                x1 = np.pad(x1, ((0,0), (to_pad // 2 + 1), (to_pad // 2)))
    
        # seriate population 1
        D = squareform(pdist(x0, metric = metric))
        D[np.isnan(D)] = 0.
        
        ii = seriate(D, timeout = 0.)
        
        x0 = x0[ii]
        
        D = cdist(x0, x1, metric = metric)
        D[np.isnan(D)] = 0.
        
        i, j = linear_sum_assignment(D)
        
        x1 = x1[j]
        
        return np.concatenate([np.expand_dims(x0, 0), np.expand_dims(x1, 0)], 0)
    elif mode == 'pad':
        if x.shape[1] > n_sites:
            ii = np.random.choice(range(x.shape[1] - n_sites))
            
            x = x[:,ii:ii + n_sites]
        else:
            to_pad = n_sites - x.shape[1]

            if to_pad % 2 == 0:
                x = np.pad(x, ((0,0), (to_pad // 2), (to_pad // 2)))
            else:
                x = np.pad(x, ((0,0), (to_pad // 2 + 1), (to_pad // 2)))
    
        return x
    
# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--mode", default = "pad")
    
    parser.add_argument("--pop_sizes", default = "20,14")
    parser.add_argument("--chunk_size", default = "4")
    parser.add_argument("--out_shape", default = "(1, 34, 512)")

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
    # configure MPI
    comm = MPI.COMM_WORLD
    
    args = parse_args()
    
    counts = dict()
    ifiles = sorted([os.path.join(args.idir, u) for u in os.listdir(args.idir) if u.split('.')[-1] == 'hdf5'])
    
    if comm.rank == 0:
        counts = dict()
        
        ofile = h5py.File(args.ofile, 'w')
        ofile_val = h5py.File('/'.join(args.ofile.split('/')[:-1]) + '/' + args.ofile.split('/')[-1].split('.')[0] + '_val.hdf5', 'w')

    pop_sizes = list(map(int, args.pop_sizes.split(',')))
    chunk_size = int(args.chunk_size)

    for ifile in ifiles:
        if comm.rank == 0:
            X = deque()
        
        comm.Barrier()
        if comm.rank == 0:
            logging.info('reading {}...'.format(ifile))
        try:
            val = '_val' in ifile
            
            ifile = h5py.File(ifile, 'r')
            
            keys = list(ifile.keys())
            if len(keys) == 0:
                continue
        except:
            continue
        
        cases = sorted(keys)
        
        for case in cases:
            comm.Barrier()
                
            keys = list(ifile[case].keys())
            if comm.rank != 0:
                for ij in range(comm.rank - 1, len(keys), comm.size - 1):
                    key = keys[ij]
                    
                    if 'x_0' not in ifile[case][key].keys():
                        x = None
                        comm.send([x], dest = 0)
                        continue

                    x = ifile[case][key]['x_0']
                    x = format_matrix(x, pop_sizes, out_shape = tuple(map(int, args.out_shape.split(','))), mode = args.mode)
                    
                    comm.send([x], dest = 0)
                    
            else:
                n_received = 0
                
                if case not in counts.keys():
                    counts[case] = [0, 0]
                
                while n_received < len(keys):
                    x = comm.recv(source = MPI.ANY_SOURCE)[0]
                    
                    if x is not None:
                        X.append(x)
                        
                    n_received += 1
                    
                    if len(X) >= chunk_size:
                        x_ = np.array([X.pop() for k in range(chunk_size)])
                        
                        if val:
                            ofile_val.create_dataset('{0}/{1}/x'.format(case, counts[case][1]), data = x_.astype(np.uint8), compression = 'lzf')
                            counts[case][1] += 1
                        else:
                            ofile.create_dataset('{0}/{1}/x'.format(case, counts[case][0]), data = x_.astype(np.uint8), compression = 'lzf')
                            counts[case][0] += 1
                
    if comm.rank == 0:
        ofile.close()
        ofile_val.close()
                

    # ${code_blocks}

if __name__ == '__main__':
    main()

