# -*- coding: utf-8 -*-
import torch.nn as nn
import numpy as np
from collections import defaultdict

from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout
from networkx.algorithms.bipartite.matrix import from_biadjacency_matrix
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import scipy.signal
import scipy.optimize
import torch


from torch import autograd
from typing import Callable, Any, Optional, Tuple, List

import warnings
from torch import Tensor

from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_max, scatter, scatter_mean, scatter_std

from sparsenn.models.gcn.layers import DynamicGraphResBlock, GraphCyclicGRUBlock, GraphInceptionBlock
from torch_geometric.nn import global_mean_pool, MessageNorm
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax, remove_self_loops

from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import LayerNorm

from torch.nn import Parameter

from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch.nn.functional as F

from torch_geometric.nn import inits
import math

from torch_sparse import SparseTensor, set_diag
from gcn_layers import Res1dBlock, MLP

class RNNSegmenter(torch.nn.Module):
    def __init__(self, window_size = 128):
        return
    
    
class TransformerClassifier(nn.Module):
    def __init__(self, in_dim = 128, n_heads = 8, 
                         n_transformer_layers = 4, n_convs = 4, L = 351, 
                         info_dim = 12, global_dim = 37):
        super(TransformerClassifier, self).__init__()
        
        self.info_embedding = nn.Sequential(nn.Linear(info_dim, 16), nn.LayerNorm((16, ))) 
        encoder_layer = nn.TransformerEncoderLayer(d_model = in_dim + 16, nhead = n_heads, 
                                                   dim_feedforward = 1024, batch_first = True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers = n_transformer_layers)
           
        self.convs = nn.ModuleList()
        for ix in range(n_convs):
            self.convs.append(Res1dBlock(in_dim + 16, in_dim + 16, 2, pooling = 'max'))
            
            L = L // 2
            
        self.global_embedding = nn.Sequential(nn.Linear(global_dim, 32), nn.LayerNorm((32,)))
        self.mlp = nn.Sequential(MLP(in_dim * L + 32, 1024, 2048, dropout = 0.05, norm = nn.LayerNorm))
        
        self.final = nn.Sequential(nn.Linear(1024, 5, bias = False), nn.LogSoftmax())
        
    def forward(self, x, x1, x2):
        bs, l, c = x1.shape
        
        x1 = self.info_embedding(x1.flatten(0, 1)).reshape(bs, l, -1)
        bs, l, c = x.shape
        
        x = torch.cat([x, x1], dim = -1)
        
        x = self.transformer(x).transpose(1, 2)
        
        for ix in range(len(self.convs)):
            x = self.convs[ix](x)
            
        x = self.down_conv(x).flatten(1, 2)
        x2 = self.global_embedding(x2)
        
        x = torch.cat([x, x2], dim = -1)
        f = self.mlp(x)
        
        x = self.final(f)

        return x, f        

class TransformerRNNClassifier(nn.Module):
    def __init__(self, in_dim = 128, n_heads = 8, 
                         n_transformer_layers = 4, n_convs = 4, L = 351, 
                         info_dim = 12, global_dim = 37):
        super(TransformerRNNClassifier, self).__init__()
        
        self.embedding = MLP(in_dim, in_dim, in_dim, norm = nn.LayerNorm)
        
        self.info_embedding = nn.Sequential(nn.Linear(info_dim, 16), nn.LayerNorm((16, ))) 
        encoder_layer = nn.TransformerEncoderLayer(d_model = in_dim + 16, nhead = n_heads, 
                                                   dim_feedforward = 1024, batch_first = True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers = n_transformer_layers)
        self.gru = nn.GRU(in_dim + 16, 1024, batch_first = True)
            
        self.global_embedding = nn.Sequential(nn.Linear(global_dim, 32), nn.LayerNorm((32,)))
            
        self.mlp = nn.Sequential(MLP(1024 + 32, 1024, 1024, norm = nn.LayerNorm))
        
        self.final = nn.Sequential(nn.Linear(1024, 512), nn.LayerNorm((512,)), nn.ReLU(), 
                                   nn.Linear(512, 5), nn.LogSoftmax())
        
    def forward(self, x, x1, x2):
        bs, l, c = x1.shape
        
        x1 = self.info_embedding(x1.flatten(0, 1)).reshape(bs, l, -1)
        bs, l, c = x.shape
        
        x = x.flatten(0, 1)
        x = self.embedding(x).reshape(bs, l, c)
        
        x = torch.cat([x, x1], dim = -1)
        
        x = self.transformer(x)
        _, x = self.gru(x)
        x = torch.squeeze(x)
    
        x2 = self.global_embedding(x2)
        
        x = torch.cat([x, x2], dim = -1)
        f = self.mlp(x)
        
        x = self.final(f)

        return x, f        
    
#updated LexStyleNet with model from paper
class LexStyleNet(nn.Module):
    def __init__(self, h = 34, w = 508):
        super(LexStyleNet, self).__init__()
        
        self.firstconv = nn.Conv1d(h,256,2)
        self.convs = nn.ModuleList()
        
        self.down = nn.AvgPool1d(2)
        
        in_channels = 256 
        out_channels = [128, 128] 
        for ix in range(2):   
            self.convs.append(nn.Sequential(nn.Conv1d(in_channels, out_channels[ix], 2), 
                                            nn.InstanceNorm1d(out_channels[ix]), 
                                            nn.ReLU(), 
                                            #nn.Dropout(0.25)
            )) 
            
            in_channels = copy.copy(out_channels[ix])
            
            w = w // 2
        
        features = 3
        
        self.out_size = 8064
        self.out = nn.Sequential(nn.Linear(63872, 128), nn.LayerNorm((128,)), nn.ReLU(),
                                 nn.Linear(128, 3), nn.LogSoftmax(dim = -1))#<- use if not LabelSmoothing  #31872 if original, 63872 if lex's
    def forward(self, x):
        x = self.firstconv(x)  
        for ix in range(len(self.convs)):
            x = self.convs[ix](x)
            x = self.down(x)
        
        
        x = x.flatten(1,2)
        
        
        return self.out(x)

if __name__ == '__main__':
    model = TransformerClassifier()
    
    x = torch.zeros((16, 351, 128))
    x1 = torch.zeros((16, 351, 12))
    x2 = torch.zeros((16, 37))

    y = model(x, x1, x2)
    print(y.shape)