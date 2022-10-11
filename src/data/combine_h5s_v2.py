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

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()

    ifiles = sorted([os.path.join(args.idir, u) for u in os.listdir(args.idir) if u.split('.')[-1] == 'hdf5'])
    counts = dict()
    
    classes = args.classes.split(',')
    
    data = dict()
    for c in classes:
        data[c] = dict()
        data[c]['x'] = []
        data[c]['x1'] = []
        data[c]['edge_index'] = []
        data[c]['mask'] = []
        data[c]['global_vec'] = []
        data[c]['D'] = []
    
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
    
    val_prop = 0.1
    
    for ifile in ifiles:
        logging.info('working on {}...'.format(ifile))
        
        generator = TreeSeqGenerator(h5py.File(ifile, 'r'), n_samples_per = 1, sequence_length = L, pad = True)
        
        for j in range(len(generator)):
            x, x1, edge_index, masks, global_vecs, y, D = generator.get_single_model_batch(sample_mode = args.sampling_mode)
            
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
                
                    
                c = y[k]
                
                data[c]['x'].append(x[k])
                data[c]['x1'].append(x1[k])
                data[c]['edge_index'].append(edge_index[k])
                data[c]['mask'].append(masks[k])
                data[c]['global_vec'].append(global_vecs[k])
                data[c]['D'].append(D[k])
                
        # append sequence lengths for histogram
        lengths.extend(generator.lengths)
        
        while all([len(data[u]['x']) > 0 for u in classes]):
            X = []
            Ds = []
            edge_index = []
            masks = []
            X1 = []
            global_vec = []
            y = []
            
            for c in classes:
                X.append(data[c]['x'][-1])
                edge_index.append(data[c]['edge_index'][-1])
                X1.append(data[c]['x1'][-1])
                y.append(classes.index(c))
                global_vec.append(data[c]['global_vec'][-1])
                masks.append(data[c]['mask'][-1])
                Ds.append(data[c]['D'][-1])
                
                del data[c]['x'][-1]
                del data[c]['edge_index'][-1]
                del data[c]['x1'][-1]
                del data[c]['mask'][-1]
                del data[c]['global_vec'][-1]
                del data[c]['D'][-1]

            X = np.array(X, dtype = np.float32)
            edge_index = np.array(edge_index, dtype = np.int32)
            X1 = np.array(X1)
            y = np.array(y, dtype = np.uint8)
            global_vec = np.array(global_vec, dtype = np.float32)
            masks = np.array(masks, dtype = np.uint8)
            Ds = np.array(Ds, dtype = np.float32)
            
            val = np.random.uniform() < val_prop
            
            if not val:
                ofile.create_dataset('{0:06d}/x'.format(counter), data = X, compression = 'lzf')
                ofile.create_dataset('{0:06d}/x1'.format(counter), data = X1, compression = 'lzf')
                ofile.create_dataset('{0:06d}/edge_index'.format(counter), data = edge_index, compression = 'lzf')
                ofile.create_dataset('{0:06d}/mask'.format(counter), data = np.array(masks), compression = 'lzf')
                ofile.create_dataset('{0:06d}/global_vec'.format(counter), data = global_vec, compression = 'lzf')
                ofile.create_dataset('{0:06d}/y'.format(counter), data = y, compression = 'lzf')
                ofile.create_dataset('{0:06d}/D'.format(counter), data = Ds, compression = 'lzf')
                ofile.flush()
            
                counter += 1
            else:
                ofile_val.create_dataset('{0:06d}/x'.format(val_counter), data = X, compression = 'lzf')
                ofile_val.create_dataset('{0:06d}/x1'.format(val_counter), data = X1, compression = 'lzf')
                ofile_val.create_dataset('{0:06d}/edge_index'.format(val_counter), data = edge_index, compression = 'lzf')
                ofile_val.create_dataset('{0:06d}/mask'.format(val_counter), data = np.array(masks), compression = 'lzf')
                ofile_val.create_dataset('{0:06d}/global_vec'.format(val_counter), data = global_vec, compression = 'lzf')
                ofile_val.create_dataset('{0:06d}/y'.format(val_counter), data = y, compression = 'lzf')
                ofile_val.create_dataset('{0:06d}/D'.format(val_counter), data = Ds, compression = 'lzf')
                ofile_val.flush()
            
                val_counter += 1
        
        logging.info('have {} samples...'.format(len(x1_means)))
        logging.info('have {} training, {} validation chunks...'.format(counter, val_counter))
            
        
    mean_bl = np.mean(bls)
    std_bl = np.std(bls)
    m_x1 = np.mean(np.array(x1_means), axis = 0)
    s_x1 = np.std(np.array(x1_means), axis = 0)
    
    np.savez(args.ofile, bl = np.array([mean_bl, std_bl]), m_x1 = m_x1, s_x1 = s_x1)
    
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


