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
    parser.add_argument("--ofile", default = "None")
    parser.add_argument("--classes", default = "hard,hard-near,neutral,soft,soft-near")
    
    parser.add_argument("--L", default = "128")
    parser.add_argument("--n_sample_iter", default = "3") # number of times to sample sequences > L
    parser.add_argument("--chunk_size", default = "5")
    
    parser.add_argument("--n_train", default = "100000")

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
        
    
    ofile = h5py.File(args.ofile, 'w')
    ofile_val = h5py.File('/'.join(args.ofile.split('/')[:-1]) + '/' + args.ofile.split('/')[-1].split('.')[0] + '_val.hdf5', 'w')
    
    L = int(args.L)
    
    counter = 0

    for ifile in ifiles:
        generator = TreeSeqGenerator(h5py.File(ifile, 'r'), n_samples_per = 1, sequence_length = L, pad = True)
        val = '_val' in ifile        
        
        for j in range(len(generator)):
            batch, y, x1, bl = generator[j]
            
            if batch is None:
                break
            
            x1 = x1.detach().cpu().numpy()
            x = batch.x
            edge_index = batch.edge_index
            y = list(y.detach().cpu().numpy().flatten())
            
            classes_ = [classes[u] for u in y]
            
            x = to_dense_batch(x, bl)
            edge_index = unbatch_edge_index(edge_index, bl)
            
            print(x.shape)
            print(edge_index.shape)
            sys.exit()
            
            
            
            
            
    ofile.close()
    ofile_val.close()
        
        
        
        
    # ${code_blocks}

if __name__ == '__main__':
    main()


