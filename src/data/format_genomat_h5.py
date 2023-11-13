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

def find_files(idir, match = '.msOut.gz'):
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

def main():
    from mpi4py import MPI
    
    args = parse_args()
    
    # configure MPI
    comm = MPI.COMM_WORLD
    
    if comm.rank == 0:
        ofile = h5py.File(args.ofile, 'w')

    pop_sizes = list(map(int, args.pop_sizes.split(',')))
    chunk_size = int(args.chunk_size)
    
    if not args.regression:
        ifiles = []

        h5_files = find_files(args.idir, '.hdf5') + glob.glob(os.path.join(args.idir, '*.hdf5'))
        
        for ifile in h5_files:
            ifile_ = h5py.File(ifile, 'r')
            
            key0 = list(ifile_.keys())[0]
            
            keys = ifile_[key0].keys()
            
            ifiles.extend([(ifile, key0 + '/' + u, key0) for u in keys])
            
        if comm.rank == 0:
            logging.info('have {} files to parse...'.format(len(ifiles)))
    
        classes = sorted(list(set([u[-1] for u in ifiles])))
    
        counts = dict()
        counts_val = dict()
        for c in classes:
            counts[c] = 0
            counts_val[c] = 0
    else:
        ifiles = glob.glob(os.path.join(args.idir, '*.hdf5'))
        
        _ = []
        for ifile in ifiles:
            ifile_ = h5py.File(ifile, 'r')
            key0 = list(ifile_.keys())[0]
            
            keys = ifile_[key0].keys()
            
            _.extend([(ifile, key0 + '/' + u, None) for u in keys])
            
        ifiles = _
        
        count = 0
        count_val = 0
            
        if comm.rank == 0:
            logging.info('have {} files to parse...'.format(len(ifiles)))
    
    if comm.rank != 0:
        for ij in range(comm.rank - 1, len(ifiles), comm.size - 1):
            ifile, key, tag = ifiles[ij]
            
            logging.info('{}: working on {}, {}...'.format(comm.rank, ifile, key))
            
            try:
                ifile = h5py.File(ifile, 'r')
                X = [np.array(ifile[key]['x_0'])]
                P = [np.zeros(X[-1].shape[-1])]
                Y = [None]
                params = [np.array(ifile[key]['y'])]
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
            
            if not args.regression:
                comm.send([X_, P_, tag], dest = 0)
            else:
                comm.send([X_, P_, params_], dest = 0)
                
            if len(ls) > 0:
                logging.info('have max shape: {}'.format(max(ls)))
    else:
        n_received = 0
        
        data = dict()
        
        if not args.regression:
            for c in classes:
                data[c] = dict()
                data[c]['X'] = []
                data[c]['P'] = []
        else:
            data = dict()
            data['X'] = []
            data['P'] = []
            data['Y'] = []
        
        while n_received < len(ifiles):
            if not args.regression:
                X, P, tag = comm.recv(source = MPI.ANY_SOURCE)
            else:
                X, P, Y = comm.recv(source = MPI.ANY_SOURCE)
                
            if X is None:
                n_received += 1
                continue
            else:
                if args.regression:
                    data['X'].extend(X)
                    data['P'].extend(P)
                    data['Y'].extend(Y)
                
                    cond = (len(data['X']) > chunk_size)
                else:
                    data[tag]['X'].extend(X)
                    data[tag]['P'].extend(P)
                
                    cond = all([len(data[u]['X']) > chunk_size for u in classes])
            
            while cond:
                if not args.regression:
                    for tag in classes:
                        ofile.create_dataset('{}/{}/x'.format(tag, counts[tag]), data = np.array(data[tag]['X'][-chunk_size:], dtype = np.uint8), compression = 'lzf')
                                                
                        counts[tag] += 1
                        
                        del data[tag]['X'][-chunk_size:]
                        del data[tag]['P'][-chunk_size:]
                else:
                    ofile.create_dataset('{}/x'.format(count), data = np.array(data['X'][-chunk_size:], dtype = np.uint8), compression = 'lzf')
                    ofile.create_dataset('{}/y'.format(count), data = np.array(data['Y'][-chunk_size:], dtype = np.float32), compression = 'lzf')
                    
                    del data['X'][-chunk_size:]
                    del data['Y'][-chunk_size:]
                    
                    count += 1    
                
                ofile.flush()
            
                if args.regression:
                    logging.info('wrote chunk {}...'.format(count))
                    cond = (len(data['X']) > chunk_size)
                else:
                    logging.info('wrote chunk {} for class {}...'.format(counts[tag], tag))
                    cond = all([len(data[u]['X']) > chunk_size for u in classes])
                    
            
            n_received += 1
            if n_received % 10 == 0:
                logging.info('received {} files thus far...'.format(n_received))
   
        if args.regression:
            if len(data['X']) > 0:
                ofile.create_dataset('{}/x'.format(count), data = np.array(data['X'], dtype = np.uint8), compression = 'lzf')
                ofile.create_dataset('{}/y'.format(count), data = np.array(data['Y'], dtype = np.float32), compression = 'lzf')
                
                count += 1    
        
        ofile.flush()
                
    if comm.rank == 0:
        ofile.close()

            
    # ${code_blocks}

if __name__ == '__main__':
    main()

