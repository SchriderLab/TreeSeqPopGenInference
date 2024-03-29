# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import os
import argparse
import logging

import h5py

import glob
from data_functions import load_data
import numpy as np

from seriate import seriate
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.optimize import linear_sum_assignment

from collections import deque

"""

"""

def find_files(idir):
    matches = []
    
    if not os.path.isdir(idir):
        return matches
        
    for root, dirnames, filenames in os.walk(idir):
        filenames = [ f for f in filenames if os.path.splitext(f)[1] in ('.msOut.gz') ]
        for filename in filenames:
            matches.append(os.path.join(root, filename))
            
    return matches

def format_matrix(x, pos, pop_sizes = (20, 14), out_shape = (2, 32, 128), metric = 'cosine', mode = 'seriate_match'):
    s0, s1 = pop_sizes
    n_pops, n_ind, n_sites = out_shape
        
    pos = np.array(pos)
    
    if x.shape[0] != s0 + s1:
        logging.debug('have x with incorrect shape!: {} vs expected {}'.format(x.shape[0], s0 + s1))
        return None, None
    
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
            pos = pos[ii:ii + n_sites]
        else:
            to_pad = n_sites - x0.shape[1]
        
            if to_pad % 2 == 0:
                x0 = np.pad(x0, ((0,0), (to_pad // 2, to_pad // 2)))
                x1 = np.pad(x1, ((0,0), (to_pad // 2, to_pad // 2)))
                
                pos = np.pad(pos, (to_pad // 2, to_pad // 2))
            else:
                x0 = np.pad(x0, ((0,0), (to_pad // 2 + 1, to_pad // 2)))
                x1 = np.pad(x1, ((0,0), (to_pad // 2 + 1, to_pad // 2)))
                
                pos = np.pad(pos, (to_pad // 2 + 1, to_pad // 2))
    
        # seriate population 1
        D = squareform(pdist(x0, metric = metric))
        D[np.isnan(D)] = 0.
        
        ii = seriate(D, timeout = 0.)
        
        x0 = x0[ii]
        
        D = cdist(x0, x1, metric = metric)
        D[np.isnan(D)] = 0.
        
        i, j = linear_sum_assignment(D)
        
        x1 = x1[j]
        
        x = np.concatenate([np.expand_dims(x0, 0), np.expand_dims(x1, 0)], 0)
    elif mode == 'pad':
        if x.shape[1] > n_sites:
            ii = np.random.choice(range(x.shape[1] - n_sites))
            
            x = x[:,ii:ii + n_sites]
            pos = pos[ii:ii + n_sites]
        else:
            to_pad = n_sites - x.shape[1]

            if to_pad % 2 == 0:
                x = np.pad(x, ((0,0), (to_pad // 2, to_pad // 2)))
                pos = np.pad(pos, (to_pad // 2, to_pad // 2))
            else:
                x = np.pad(x, ((0,0), (to_pad // 2 + 1, to_pad // 2)))
                pos = np.pad(pos, (to_pad // 2 + 1, to_pad // 2))
    
        
    elif mode == 'seriate': # one population
        if x.shape[1] > n_sites:
            ii = np.random.choice(range(x.shape[1] - n_sites))
            
            x = x[:,ii:ii + n_sites]
            pos = pos[ii:ii + n_sites]
        else:
            to_pad = n_sites - x.shape[1]

            if to_pad % 2 == 0:
                x = np.pad(x, ((0,0), (to_pad // 2, to_pad // 2)))
                pos = np.pad(pos, (to_pad // 2, to_pad // 2))
            else:
                x = np.pad(x, ((0,0), (to_pad // 2 + 1, to_pad // 2)))
                pos = np.pad(pos, (to_pad // 2 + 1, to_pad // 2))
                
        D = squareform(pdist(x, metric = metric))
        D[np.isnan(D)] = 0.
        
        ii = seriate(D, timeout = 0.)
        
        x = x[ii,:]
        
    return x, pos

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}
def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--regression", action = "store_true")
    
    parser.add_argument("--pop_sizes", default = "20,14")
    parser.add_argument("--chunk_size", default = "4")
    parser.add_argument("--out_shape", default = "1,34,512")
    
    parser.add_argument("--classes", default = "None")
    parser.add_argument("--val_prop", default = "0.05")
    parser.add_argument("--mode", default = "seriate_match")

    parser.add_argument("--ofile", default = "None")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    # ${odir_del_block}

    return args

import time

def main():
    from mpi4py import MPI
    
    args = parse_args()
    
    # configure MPI
    comm = MPI.COMM_WORLD
    
    if comm.rank == 0:
        ofile = h5py.File(args.ofile, 'w')
        if float(args.val_prop) > 0:
            ofile_val = h5py.File('/'.join(args.ofile.split('/')[:-1]) + '/' + args.ofile.split('/')[-1].split('.')[0] + '_val.hdf5', 'w')

    pop_sizes = list(map(int, args.pop_sizes.split(',')))
    chunk_size = int(args.chunk_size)
    
    if not args.regression:
        ifiles = []
        
        classes = sorted(os.listdir(args.idir))
        
        for c in classes:
            idir = os.path.join(args.idir, c)
            ifiles.extend([(c, u) for u in find_files(idir)])
            
        if comm.rank == 0:
            logging.info('have {} files to parse...'.format(len(ifiles)))
    
        counts = dict()
        counts_val = dict()
        for c in classes:
            counts[c] = 0
            counts_val[c] = 0
    else:
        ifiles = []
        classes = sorted(os.listdir(args.idir))
        
        for c in classes:
            idir = os.path.join(args.idir, c)
            ifiles.extend([(None, u) for u in find_files(idir)])
            
        count = 0
        count_val = 0
            
        if comm.rank == 0:
            logging.info('have {} files to parse...'.format(len(ifiles)))
    
    if comm.rank != 0:
        for ij in range(comm.rank - 1, len(ifiles), comm.size - 1):
            tag, ifile = ifiles[ij]
            logging.info('{}: working on {}...'.format(comm.rank, ifile))
            
            t0 = time.time()
            
            try:
                X, Y, P, params = load_data(ifile)
            except:
                logging.info('could not read {}!'.format(ifile))
                comm.send([None, None, None], dest = 0)
                continue
            
            X_ = []
            P_ = []
            params_ = []
            ls = []
            
            for ix, x in enumerate(X):
                if x is None:
                    continue
                
                try:
                    ls.append(x.shape[1])
                except:
                    pass
                
                x, p = format_matrix(x, P[ix], pop_sizes, out_shape = tuple(map(int, args.out_shape.split(','))), mode = args.mode)
                                
                if x is not None:

                    X_.append(x)
                    P_.append(p)
                    params_.append(params[ix])
            
            logging.info('sending {} matrices from {}...'.format(len(X_), ifile))    
            print('time: {}'.format(time.time() - t0))
            
            
            if not args.regression:
                comm.send([X_, P_, tag], dest = 0)
            else:
                comm.send([X_, P_, params_], dest = 0)
                
            if len(ls) > 0:
                logging.info('have max shape: {}'.format(max(ls)))
    else:
        n_received = 0
                
        while n_received < len(ifiles):
            if not args.regression:
                Xf, p, tag = comm.recv(source = MPI.ANY_SOURCE)
            else:
                Xf, p, y = comm.recv(source = MPI.ANY_SOURCE)
                
            if Xf is None:
                n_received += 1
                continue
            
            logging.info('have len {}'.format(len(Xf)))
            while len(Xf) >= chunk_size:
                if np.random.uniform() < float(args.val_prop):
                    if not args.regression:
                        ofile_val.create_dataset('{}/{}/x'.format(tag, counts_val[tag]), data = np.array(Xf[-chunk_size:], dtype = np.uint8), compression = 'lzf')
                        ofile_val.create_dataset('{}/{}/p'.format(tag, counts_val[tag]), data = np.array(p[-chunk_size:], dtype = np.float32), compression = 'lzf')
                        
                        counts_val[tag] += 1
                    else:
                        ofile_val.create_dataset('{}/x'.format(count_val), data = np.array(Xf[-chunk_size:], dtype = np.uint8), compression = 'lzf')
                        ofile_val.create_dataset('{}/p'.format(count_val), data = np.array(p[-chunk_size:], dtype = np.float32), compression = 'lzf')
                        ofile_val.create_dataset('{}/y'.format(count_val), data = np.array(y[-chunk_size:], dtype = np.float32), compression = 'lzf')
                        
                        count_val += 1
                          
                    ofile_val.flush()
                    
                else:
                    if not args.regression:
                        ofile.create_dataset('{}/{}/x'.format(tag, counts[tag]), data = np.array(Xf[-chunk_size:], dtype = np.uint8), compression = 'lzf')
                        ofile.create_dataset('{}/{}/p'.format(tag, counts[tag]), data = np.array(p[-chunk_size:], dtype = np.float32), compression = 'lzf')
                        
                        counts[tag] += 1
                    else:
                        ofile.create_dataset('{}/x'.format(count), data = np.array(Xf[-chunk_size:], dtype = np.uint8), compression = 'lzf')
                        ofile.create_dataset('{}/p'.format(count), data = np.array(p[-chunk_size:], dtype = np.float32), compression = 'lzf')
                        ofile.create_dataset('{}/y'.format(count), data = np.array(y[-chunk_size:], dtype = np.float32), compression = 'lzf')
                        
                        count += 1    
                    
                    ofile.flush()
            
                del Xf[-chunk_size:]
                del p[-chunk_size:]
                
                if args.regression:
                    del y[-chunk_size:]
            
            n_received += 1
            if n_received % 10 == 0:
                logging.info('received {} files thus far...'.format(n_received))
                     
        if len(Xf) > 0:
            if not args.regression:
                ofile.create_dataset('{}/{}/x'.format(tag, counts[tag]), data = np.array(Xf, dtype = np.uint8), compression = 'lzf')
                ofile.create_dataset('{}/{}/p'.format(tag, counts[tag]), data = np.array(p, dtype = np.float32), compression = 'lzf')
                
                counts[tag] += 1
            else:
                ofile.create_dataset('{}/x'.format(count), data = np.array(Xf, dtype = np.uint8), compression = 'lzf')
                ofile.create_dataset('{}/p'.format(count), data = np.array(p, dtype = np.float32), compression = 'lzf')
                ofile.create_dataset('{}/y'.format(count), data = np.array(y, dtype = np.float32), compression = 'lzf')
                
                count += 1    
        
        ofile.flush()
                
    if comm.rank == 0:
        ofile.close()
        if float(args.val_prop) > 0:
            ofile_val.close()
            
    # ${code_blocks}

if __name__ == '__main__':
    main()

