# -*- coding: utf-8 -*-
import os
import argparse
import logging

import h5py
import numpy as np

import sys
sys.path.insert(0, './src/models')

from data_loaders import TreeSeqGenerator
from torch_geometric.utils import to_dense_batch
from typing import List

import torch
from torch import Tensor

from torch_geometric.utils import degree
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from collections import deque
import glob
import random

def unbatch(src: Tensor, batch: Tensor, dim: int = 0) -> List[Tensor]:
    r"""Splits :obj:`src` according to a :obj:`batch` vector along dimension
    :obj:`dim`.

    Args:
        src (Tensor): The source tensor.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            entry in :obj:`src` to a specific example. Must be ordered.
        dim (int, optional): The dimension along which to split the :obj:`src`
            tensor. (default: :obj:`0`)

    :rtype: :class:`List[Tensor]`

    Example:

        >>> src = torch.arange(7)
        >>> batch = torch.tensor([0, 0, 0, 1, 1, 2, 2])
        >>> unbatch(src, batch)
        (tensor([0, 1, 2]), tensor([3, 4]), tensor([5, 6]))
    """
    sizes = degree(batch, dtype=torch.long).tolist()
    return src.split(sizes, dim)

def unbatch_edge_index(edge_index: Tensor, batch: Tensor) -> List[Tensor]:
    r"""Splits the :obj:`edge_index` according to a :obj:`batch` vector.

    Args:
        edge_index (Tensor): The edge_index tensor. Must be ordered.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered.

    :rtype: :class:`List[Tensor]`

    Example:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 4, 5, 5, 6],
        ...                            [1, 0, 2, 1, 3, 2, 5, 4, 6, 5]])
        >>> batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
        >>> unbatch_edge_index(edge_index, batch)
        (tensor([[0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2]]),
        tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]]))
    """
    deg = degree(batch, dtype=torch.int64)
    ptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)[:-1]], dim=0)

    edge_batch = batch[edge_index[0]]
    edge_index = edge_index - ptr[edge_batch]
    sizes = degree(edge_batch, dtype=torch.int64).cpu().tolist()
    return edge_index.split(sizes, dim=1)

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--ofile", default = "seln_means.npz")
    parser.add_argument("--classes", default = "hard,hard-near,neutral,soft,soft-near")
    
    parser.add_argument("--L", default = "128")
    parser.add_argument("--n_sample_iter", default = "3") # number of times to sample sequences > L
    parser.add_argument("--chunk_size", default = "5")
    parser.add_argument("--sampling_mode", default = "equi")
    
    parser.add_argument("--n_sample", default = "None")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()
    
    ifiles = glob.glob(os.path.join(args.idir, '*/*.hdf5'))
    random.shuffle(ifiles)

    #ifiles = sorted([os.path.join(args.idir, u) for u in os.listdir(args.idir) if u.split('.')[-1] == 'hdf5'])
    counts = dict()
    
    # classification
    if ',' in args.classes:
        classes = args.classes.split(',')
        
        data = dict()
        for c in classes:
            data[c] = dict()
            data[c]['x'] = deque()
            data[c]['x1'] = deque()
            data[c]['edge_index'] = deque()
            data[c]['mask'] = deque()
            data[c]['global_vec'] = deque()
            
        classification = True
    # regression
    else:
        classification = False
        
        data = dict()
        data['x'] = deque()
        data['x1'] = deque()
        data['edge_index'] = deque()
        data['mask'] = deque()
        data['global_vec'] = deque()
        data['y'] = deque()
        
    ofile = h5py.File(args.ofile, 'w')
    ofile_val = h5py.File('/'.join(args.ofile.split('/')[:-1]) + '/' + args.ofile.split('/')[-1].split('.')[0] + '_val.hdf5', 'w')
    
    L = int(args.L)
    
    # counters for dataset labels
    counter = 0
    val_counter = 0
    
    lengths = []
    
    x1_means = []
    x1_stds = []
    bls = []
    n_muts = []
    yl = []
    
    val_prop = 0.1
    chunk_size = int(args.chunk_size)
    
    for ifile in ifiles:
        logging.info('working on {}...'.format(ifile))
        
        generator = TreeSeqGenerator(h5py.File(ifile, 'r'), n_samples_per = 1, sequence_length = L, pad = True, categorical = classification)
        
        if args.n_sample == "None":
            try:
                N = len(generator)
            except:
                continue
        else:
            N = int(args.n_sample)
        
        for j in range(N):
            x, x1, edge_index, masks, global_vecs, y = generator.get_single_model_batch(sample_mode = args.sampling_mode)
            
            if x is None:
                break
                    
            for k in range(len(x)):
                if np.random.uniform() < 0.02:
                    x1_means.append(np.mean(x1[k][np.where(masks[k] != 0.)[0],:], axis = 0))
                    x1_stds.append(np.std(x1[k][np.where(masks[k] != 0.)[0],:], axis = 0))
                    
                    # sequence, node, feature
                    ii = np.where(x[k][:,:,0] > 0)
                    bls.extend(np.random.choice(np.log(x[k][ii[0],ii[1],0]), min([len(ii[0]), 10]), replace = False))
             
                    muts = x[k][np.where(masks[k] != 0.)[0],:,-1].flatten()
                    n_muts.extend(np.random.choice(muts, min([len(muts), 10]), replace = False))
                    yl.extend(y[k])
                
                if classification:
                    c = y[k]
                    
                    data[c]['x'].append(x[k])
                    data[c]['x1'].append(x1[k])
                    data[c]['edge_index'].append(edge_index[k])
                    data[c]['mask'].append(masks[k])
                    data[c]['global_vec'].append(global_vecs[k])
                else:
                    data['x'].append(x[k])
                    data['x1'].append(x1[k])
                    data['edge_index'].append(edge_index[k])
                    data['mask'].append(masks[k])
                    data['global_vec'].append(global_vecs[k])
                    data['y'].append(y)
                
        # append sequence lengths for histogram
        lengths.extend(generator.lengths)
        
        if classification:
            cond = all([len(data[u]['x']) > 0 for u in classes])
        else:
            cond = (len(data['x']) >= chunk_size)
        
        while cond:
            X = []
            Ds = []
            edge_index = []
            masks = []
            X1 = []
            global_vec = []
            y = []
            
            if classification:
                for c in classes:
                    X.append(data[c]['x'].pop())
                    edge_index.append(data[c]['edge_index'].pop())
                    X1.append(data[c]['x1'].pop())
                    y.append(classes.index(c))
                    global_vec.append(data[c]['global_vec'].pop())
                    masks.append(data[c]['mask'].pop())
            else:
                for c in range(chunk_size):
                    X.append(data['x'].pop())
                    edge_index.append(data['edge_index'].pop())
                    X1.append(data['x1'].pop())
                    y.append(data['y'].pop())
                    global_vec.append(data['global_vec'].pop())
                    masks.append(data['mask'].pop())
                

            X = np.array(X, dtype = np.float32)
            edge_index = np.array(edge_index, dtype = np.int32)
            X1 = np.array(X1)
            print([u.shape for u in y])
            
            if classification:
                y = np.array(y, dtype = np.uint8)
            else:
                y = np.array(y, dtype = np.float32)
            global_vec = np.array(global_vec, dtype = np.float32)
            masks = np.array(masks, dtype = np.uint8)
            
            val = np.random.uniform() < val_prop
            
            if not val:
                ofile.create_dataset('{0:06d}/x'.format(counter), data = X, compression = 'lzf')
                ofile.create_dataset('{0:06d}/x1'.format(counter), data = X1, compression = 'lzf')
                ofile.create_dataset('{0:06d}/edge_index'.format(counter), data = edge_index, compression = 'lzf')
                ofile.create_dataset('{0:06d}/mask'.format(counter), data = np.array(masks), compression = 'lzf')
                ofile.create_dataset('{0:06d}/global_vec'.format(counter), data = global_vec, compression = 'lzf')
                ofile.create_dataset('{0:06d}/y'.format(counter), data = y, compression = 'lzf')
                ofile.flush()
            
                counter += 1
            else:
                ofile_val.create_dataset('{0:06d}/x'.format(val_counter), data = X, compression = 'lzf')
                ofile_val.create_dataset('{0:06d}/x1'.format(val_counter), data = X1, compression = 'lzf')
                ofile_val.create_dataset('{0:06d}/edge_index'.format(val_counter), data = edge_index, compression = 'lzf')
                ofile_val.create_dataset('{0:06d}/mask'.format(val_counter), data = np.array(masks), compression = 'lzf')
                ofile_val.create_dataset('{0:06d}/global_vec'.format(val_counter), data = global_vec, compression = 'lzf')
                ofile_val.create_dataset('{0:06d}/y'.format(val_counter), data = y, compression = 'lzf')
                ofile_val.flush()
            
                val_counter += 1
                
            if classification:
                cond = all([len(data[u]['x']) > 0 for u in classes])
            else:
                cond = (len(data['x']) >= chunk_size)
        
        logging.info('have {} samples...'.format(len(x1_means)))
        logging.info('have {} training, {} validation chunks...'.format(counter, val_counter))
            
        
    mean_bl = np.mean(bls)
    std_bl = np.std(bls)
    m_x1 = np.mean(np.array(x1_means), axis = 0)
    s_x1 = np.std(np.array(x1_means), axis = 0)
    
    if not classification:
        m_y = np.mean(np.array(yl), axis = 0)
        s_y = np.std(np.array(yl), axis = 0)
    
        np.savez(args.ofile.replace('hdf5', 'npz'), bl = np.array([mean_bl, std_bl]), m_x1 = m_x1, s_x1 = s_x1, m_y = m_y, s_y = s_y)
    else:
        np.savez(args.ofile.replace('hdf5', 'npz'), bl = np.array([mean_bl, std_bl]), m_x1 = m_x1, s_x1 = s_x1)
        
    
    logging.info('closing files and plotting hist...')
    ofile.close()
    ofile_val.close()
        
    plt.hist(lengths, bins = 35)
    plt.savefig('seql_hist.png', dpi = 100)
    plt.close()
        
    plt.hist(n_muts, bins = 35)
    plt.savefig('nmut_hist.png', dpi = 100)
    plt.close()
    
    
        
        
    # ${code_blocks}

if __name__ == '__main__':
    main()


