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
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import filtered_lrelu
from torch_utils.ops import bias_act

from torch import autograd
from typing import Callable, Any, Optional, Tuple, List

import warnings
from torch import Tensor

from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_max, scatter, scatter_mean, scatter_std

from sparsenn.models.gcn.layers import DynamicGraphResBlock, GraphCyclicGRUBlock, GraphInceptionBlock
from torch_geometric.nn import global_mean_pool, MessageNorm
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax

from torch_geometric.nn.inits import glorot
from torch_utils.ops import filtered_lrelu
from torch_geometric.nn import LayerNorm

from torch.nn import Parameter

from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch.nn.functional as F

from torch_geometric.nn import inits
import math

from torch_sparse import SparseTensor, set_diag

# not sure what I patched here...
# I'm assuming it was something...
class Linear(torch.nn.Module):
    r"""Applies a linear tranformation to the incoming data
    .. math::
        \mathbf{x}^{\prime} = \mathbf{x} \mathbf{W}^{\top} + \mathbf{b}
    similar to :class:`torch.nn.Linear`.
    It supports lazy initialization and customizable weight and bias
    initialization.
    Args:
        in_channels (int): Size of each input sample. Will be initialized
            lazily in case it is given as :obj:`-1`.
        out_channels (int): Size of each output sample.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        weight_initializer (str, optional): The initializer for the weight
            matrix (:obj:`"glorot"`, :obj:`"uniform"`, :obj:`"kaiming_uniform"`
            or :obj:`None`).
            If set to :obj:`None`, will match default weight initialization of
            :class:`torch.nn.Linear`. (default: :obj:`None`)
        bias_initializer (str, optional): The initializer for the bias vector
            (:obj:`"zeros"` or :obj:`None`).
            If set to :obj:`None`, will match default bias initialization of
            :class:`torch.nn.Linear`. (default: :obj:`None`)
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True,
                 weight_initializer: Optional[str] = None,
                 bias_initializer: Optional[str] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        if in_channels > 0:
            self.weight = Parameter(torch.Tensor(out_channels, in_channels))
        else:
            self.weight = nn.parameter.UninitializedParameter()
            self._hook = self.register_forward_pre_hook(
                self.initialize_parameters)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._load_hook = self._register_load_state_dict_pre_hook(
            self._lazy_load_hook)

        self.reset_parameters()

    def __deepcopy__(self, memo):
        out = Linear(self.in_channels, self.out_channels, self.bias
                     is not None, self.weight_initializer,
                     self.bias_initializer)
        if self.in_channels > 0:
            out.weight = copy.deepcopy(self.weight, memo)
        if self.bias is not None:
            out.bias = copy.deepcopy(self.bias, memo)
        return out

    def reset_parameters(self):
        if isinstance(self.weight, nn.parameter.UninitializedParameter):
            pass
        elif self.weight_initializer == 'glorot':
            inits.glorot(self.weight)
        elif self.weight_initializer == 'uniform':
            bound = 1.0 / math.sqrt(self.weight.size(-1))
            torch.nn.init.uniform_(self.weight.data, -bound, bound)
        elif self.weight_initializer == 'kaiming_uniform':
            inits.kaiming_uniform(self.weight, fan=self.in_channels,
                                  a=math.sqrt(5))
        elif self.weight_initializer is None:
            inits.kaiming_uniform(self.weight, fan=self.in_channels,
                                  a=math.sqrt(5))
        else:
            raise RuntimeError(f"Linear layer weight initializer "
                               f"'{self.weight_initializer}' is not supported")

        if isinstance(self.weight, nn.parameter.UninitializedParameter):
            pass
        elif self.bias is None:
            pass
        elif self.bias_initializer == 'zeros':
            inits.zeros(self.bias)
        elif self.bias_initializer is None:
            inits.uniform(self.in_channels, self.bias)
        else:
            raise RuntimeError(f"Linear layer bias initializer "
                               f"'{self.bias_initializer}' is not supported")

    def forward(self, x: Tensor) -> Tensor:
        """"""
        return F.linear(x, self.weight, self.bias)

    @torch.no_grad()
    def initialize_parameters(self, module, input):
        if isinstance(self.weight, nn.parameter.UninitializedParameter):
            self.in_channels = input[0].size(-1)
            self.weight.materialize((self.out_channels, self.in_channels))
            self.reset_parameters()
        self._hook.remove()
        delattr(self, '_hook')

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if isinstance(self.weight, nn.parameter.UninitializedParameter):
            destination[prefix + 'weight'] = self.weight
        else:
            destination[prefix + 'weight'] = self.weight.detach()
        if self.bias is not None:
            destination[prefix + 'bias'] = self.bias.detach()

    def _lazy_load_hook(self, state_dict, prefix, local_metadata, strict,
                        missing_keys, unexpected_keys, error_msgs):

        weight = state_dict[prefix + 'weight']
        if isinstance(weight, nn.parameter.UninitializedParameter):
            self.in_channels = -1
            self.weight = nn.parameter.UninitializedParameter()
            if not hasattr(self, '_hook'):
                self._hook = self.register_forward_pre_hook(
                    self.initialize_parameters)

        elif isinstance(self.weight, nn.parameter.UninitializedParameter):
            self.in_channels = weight.size(-1)
            self.weight.materialize((self.out_channels, self.in_channels))
            if hasattr(self, '_hook'):
                self._hook.remove()
                delattr(self, '_hook')

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, bias={self.bias is not None})')

# patched to have message norm.
class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
    the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j
        \, \Vert \, \mathbf{\Theta}_{e} \mathbf{e}_{i,j}]\right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k
        \, \Vert \, \mathbf{\Theta}_{e} \mathbf{e}_{i,k}]\right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        fill_value (float or Tensor or str, optional): The way to generate
            edge features of self-loops (in case :obj:`edge_dim != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
          :math:`((|\mathcal{V}_t|, H * F_{out})` if bipartite.
          If :obj:`return_attention_weights=True`, then
          :math:`((|\mathcal{V}|, H * F_{out}),
          ((2, |\mathcal{E}|), (|\mathcal{E}|, H)))`
          or :math:`((|\mathcal{V_t}|, H * F_{out}), ((2, |\mathcal{E}|),
          (|\mathcal{E}|, H)))` if bipartite
    """
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        activation: str = 'softmax',
        negative_slope: float = 0.2,
        dropout: float = 0.,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = False,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        self.norm = MessageNorm(True)

        if edge_dim is not None:
            self.lin_edge = nn.Sequential(Linear(edge_dim, heads * 2, weight_initializer='glorot'), nn.LayerNorm(heads * 2), nn.ReLU(),
                                          Linear(heads * 2, heads, bias = False, weight_initializer='glorot'))
            self.att_edge = Parameter(torch.Tensor(1, heads, 1))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None
        self.activation = activation

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_edge)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None):
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        # NOTE: attention weights will be returned whenever
        # `return_attention_weights` is set to a value, regardless of its
        # actual value (might be `True` or `False`). This is a current somewhat
        # hacky workaround to allow for TorchScript support via the
        # `torch.jit._overload` decorator, as we can only change the output
        # arguments conditioned on type (`None` or `bool`), not based on its
        # actual value.

        H, C = self.heads, self.out_channels

        x_src = x_dst = x.view(-1, H, C)
        x = (x_src, x_dst)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor, edge_attr: OptTensor)  # noqa
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr,
                             size=size)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out
        
    def update(self, inputs, x):
        return self.norm(x[0], inputs)

    def message(self, x_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)
        assert self.lin_edge is not None
        edge_attr = self.lin_edge(edge_attr)
        edge_attr = edge_attr.view(-1, self.heads, 1)
        alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
        alpha = alpha_edge

        if self.activation == 'sigmoid':
            alpha = torch.sigmoid(F.leaky_relu(alpha, self.negative_slope))
        elif self.activation == 'softmax':
            alpha = softmax(alpha, index, ptr, size_i)
        
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        return x_j * alpha.unsqueeze(-1)
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

class Res1dGraphBlock(nn.Module):
    def __init__(self, in_shape, out_channels, n_layers, gcn_channels = 4,
                             k = 3, pooling = 'max', up = False, att_activation = 'sigmoid'):
        super(Res1dGraphBlock, self).__init__()
        
        in_shape = list(in_shape)
        
        # pass each pop through their own convolutions
        self.convs_l = nn.ModuleList()
        self.convs_r = nn.ModuleList()
        
        self.gcn_convs = nn.ModuleList()
        
        self.norms = nn.ModuleList()
        self.gcns = nn.ModuleList()
        
        for ix in range(n_layers):
            self.convs_l.append(nn.Conv2d(in_shape[0], out_channels, (1, 3), 
                                        stride = (1, 1), padding = (0, (k + 1) // 2 - 1), bias = False))
            self.convs_r.append(nn.Conv2d(in_shape[0], out_channels, (1, 3), 
                                        stride = (1, 1), padding = (0, (k + 1) // 2 - 1), bias = False))
            
            # for down sampling the dimensionality of the features for the gcn part
            self.gcn_convs.append(nn.Conv2d(out_channels, gcn_channels, 1, 1))
            
            self.gcns.append(VanillaAttConv(activation = att_activation))
            self.norms.append(nn.Sequential(nn.InstanceNorm2d(out_channels), nn.Dropout2d(0.1)))
            
            in_shape[0] = out_channels + gcn_channels
        
        if pooling == 'max':
            self.pool = nn.MaxPool2d((1, 2), stride = (1, 2))
        else:
            self.pool = None
            
        self.activation = nn.ReLU()
        
    def forward(self, x, edge_index, edge_attr, batch):
        batch_size, n_channels, n_ind, n_sites = x.shape
        
        xs = []
        xgs = []
        
        xl = self.convs_l[0](x[:,:,:n_ind // 2,:])
        xr = self.convs_r[0](x[:,:,n_ind // 2:,:])
        
        xs.append(torch.cat([xl, xr], dim = 2))
        
        # the graph features at this point in the network
        xg = self.gcn_convs[0](xs[-1])
        n_channels = xg.shape[1]

        xg = torch.flatten(xg.transpose(1, 2), 2, 3).flatten(0, 1)   

        # insert graph convolution here...
        xg = self.gcns[0](xg, edge_index, edge_attr)

        ##################
        
        xg = to_dense_batch(xg, batch)[0]
        xg = xg.reshape(batch_size, n_ind, n_channels, n_sites).transpose(1, 2)
        
        # this will have out_channels + graph channels
        xs[-1] = self.norms[0](torch.cat([xg, xs[-1]], dim = 1))
        
        for ix in range(1, len(self.norms)):
            xl = self.convs_l[ix](xs[-1][:,:,:n_ind // 2,:])
            xr = self.convs_r[ix](xs[-1][:,:,n_ind // 2:,:])
            
            xs.append(torch.cat([xl, xr], dim = 2))

            xg = self.gcn_convs[ix](xs[-1])
            n_channels = xg.shape[1]
    
            xg = torch.flatten(xg.transpose(1, 2), 2, 3).flatten(0, 1)   
            
            # insert graph convolution here...
            xg = self.gcns[ix](xg, edge_index, edge_attr)     
            ##################
            
            xg = to_dense_batch(xg, batch)[0]
            xg = xg.reshape(batch_size, n_ind, n_channels, n_sites).transpose(1, 2)
            
            # this will have out_channels + graph channels
            # concatenate the graph features and add the previous for a residual connection
            xs[-1] = self.norms[ix](torch.cat([xg, xs[-1]], dim = 1) + xs[-2])
                
        x = self.activation(torch.cat(xs, dim = 1))
        
        if self.pool is not None:
            return self.pool(x)
        else:
            return x

class VanillaAttConv(MessagePassing):
    def __init__(self, negative_slope = 0.2, activation = 'sigmoid'):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.norm = MessageNorm(True)
        self.negative_slope = negative_slope
        
        _ = [nn.BatchNorm1d(8), nn.Linear(8, 16), nn.LayerNorm((16,)), nn.ReLU(), 
                                     nn.Linear(16, 32), nn.LayerNorm((32,)), nn.ReLU(), nn.Linear(32, 1)]
        
        if activation == 'sigmoid':
            _.append(nn.LeakyReLU(negative_slope = negative_slope))
            _.append(nn.Sigmoid())
        elif activation == 'tanh':
            _.append(nn.LeakyReLU(negative_slope = negative_slope))
            _.append(nn.Tanh())
        
        self.att_mlp = nn.Sequential(*_)
        
    
    def forward(self, x, edge_index, edge_attr):
        att = self.att_mlp(edge_attr)
        
        return self.propagate(edge_index, x = x, att = att)

    def message(self, x_j, att):
        return x_j * att
    
    def update(self, inputs, x):
        return self.norm(x, inputs)
    
class VanillaConv(MessagePassing):
    def __init__(self, negative_slope = 0.2):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.norm = MessageNorm(True)
        self.negative_slope = negative_slope

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j
    
    def update(self, inputs, x):
        return self.norm(x, inputs)
    
class Res1dBlock(nn.Module):
    def __init__(self, in_shape, out_channels, n_layers, 
                             k = 3, pooling = 'max'):
        super(Res1dBlock, self).__init__()
        
        in_shape = list(in_shape)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for ix in range(n_layers):
            self.convs.append(nn.Conv2d(in_shape[0], out_channels, (1, 3), 
                                        stride = (1, 1), padding = (0, (k + 1) // 2 - 1)))
            self.norms.append(nn.Sequential(nn.InstanceNorm2d(out_channels), nn.Dropout2d(0.1)))
            
            in_shape[0] = out_channels
        
        if pooling == 'max':
            self.pool = nn.MaxPool2d((1, 2), stride = (1, 2))
        else:
            self.pool = None
            
        self.activation = nn.ELU()
        
    def forward(self, x, return_unpooled = False):
        xs = [self.norms[0](self.convs[0](x))]
        
        for ix in range(1, len(self.norms)):
            xs.append(self.norms[ix](self.convs[ix](xs[-1])) + xs[-1])
            
        x = self.activation(torch.cat(xs, dim = 1))
        
        if self.pool is not None:
            xp = self.pool(x)
        else:
            xp = x
            
        if return_unpooled:
            return xp, x
        else:
            return xp
        
        if self.pool is not None:
            return self.pool(x)
        else:
            return x
