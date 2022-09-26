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
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
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

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels,
                                  bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

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

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

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

    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha


    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j
    
    def update(self, inputs, x):
        return self.norm(x[0], inputs)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
    
from torch_geometric.typing import Adj, OptTensor, PairTensor
    
class GATv2Conv(MessagePassing):
    r"""The GATv2 operator from the `"How Attentive are Graph Attention Networks?"
    <https://arxiv.org/abs/2105.14491>`_ paper, which fixes the static
    attention problem of the standard :class:`~torch_geometric.conv.GATConv`
    layer: since the linear layers in the standard GAT are applied right after
    each other, the ranking of attended nodes is unconditioned on the query
    node. In contrast, in GATv2, every node can attend to any other node.

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_k]
        \right)\right)}.

    If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
    the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_j \, \Vert \, \mathbf{e}_{i,j}]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_k \, \Vert \, \mathbf{e}_{i,k}]
        \right)\right)}.

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
        share_weights (bool, optional): If set to :obj:`True`, the same matrix
            will be applied to the source and the target node of every edge.
            (default: :obj:`False`)
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
        name = 'gcn',
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        share_weights: bool = False,
        **kwargs,
    ):
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
        self.share_weights = share_weights
        
        self.name = name

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * out_channels,
                                    bias=bias, weight_initializer='glorot')
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels,
                                bias=bias, weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels[1], heads * out_channels,
                                    bias=bias, weight_initializer='glorot')

        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
        else:
            self.lin_edge = None

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None
        self.norm = MessageNorm(True)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att)
        zeros(self.bias)


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None,
                return_attention_weights: bool = None):
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr,
                             size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x += edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
    
    def update(self, inputs, x):
        return self.norm(x[0], inputs)


    
# a basic MLP module
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim = 256, n_blk = 3, norm = nn.BatchNorm1d,
                 activation = nn.ReLU):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, dim), norm(dim), activation(inplace=True)]
        for _ in range(n_blk - 2):
            layers += [nn.Linear(dim, dim), norm(dim), activation(inplace=True)]
        layers += [nn.Linear(dim, output_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
   
import logging
    
class GATSeqClassifier(nn.Module):
    def __init__(self, batch_size, n_classes = 3, in_dim = 6, info_dim = 12, global_dim = 37, global_embedding_dim = 128, gcn_dim = 26, n_gcn_layers = 4, gcn_dropout = 0.,
                             num_gru_layers = 1, hidden_size = 128, L = 32, n_heads = 1, n_gcn_iter = 6,
                             use_conv = False, conv_k = 5, conv_dim = 4, momenta_gamma = 0.9): 
        super(GATSeqClassifier, self).__init__()

        self.gcns = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.act = nn.ReLU()
        
        self.batch_size = batch_size
        
        self.n_gcn_iter = n_gcn_iter
        
        self.embedding = nn.Linear(in_dim, gcn_dim, bias = True)
        self.global_embedding = nn.Linear(global_dim, global_embedding_dim, bias = True)
        self.global_embedding.register_backward_hook(self.update_momenta)
        self.global_embedding.name = 'global_embedding'
        
        self.embedding.name = 'node_embedding'
        self.embedding.register_backward_hook(self.update_momenta)
        
        gcn_dim += in_dim
        
        self.embedding_norm = nn.LayerNorm((gcn_dim, ))
        self.global_embedding_norm = nn.LayerNorm((global_embedding_dim, ))
        self.dropout = nn.Dropout(0.15)
        
        self.L = L
        self.gcns = nn.ModuleList()
        
        for ix in range(n_gcn_iter):    
            self.norms.append(nn.LayerNorm((gcn_dim, )))
            self.gcns.append(GATv2Conv(gcn_dim, gcn_dim // n_heads, heads = n_heads, dropout = gcn_dropout, name = 'gcn_layer_{}'.format(ix)))
            self.gcns[-1].register_backward_hook(self.update_momenta)
        
        self.use_conv = use_conv
        if self.use_conv:
            # 1d convolution over graph features to cat to MLP layer
            self.conv = Res1dBlock(hidden_size * num_gru_layers * 2 + info_dim, conv_dim, conv_k)
            self.conv.register_backward_hook(self.update_momenta)
            
        """
        for ix in range(1, n_gcn_layers):
            self.gcns.extend([GATv2Conv(gcn_dim, gcn_dim // n_heads, heads = n_heads)])
            self.norms.append(nn.LayerNorm((gcn_dim, )))
            
        self.final_gat = GATv2Conv(gcn_dim, gcn_dim)
        """  
        # we'll give it mean, max, min, std of GCN features per graph
        self.gru = nn.GRU(hidden_size * num_gru_layers * 2 + info_dim, hidden_size * num_gru_layers * 2, num_layers = num_gru_layers, batch_first = True, bidirectional = True)
        self.gru.name = 'gru'
        self.gru.register_backward_hook(self.update_momenta)
        
        self.graph_gru = nn.GRU(gcn_dim + in_dim, hidden_size, num_layers = num_gru_layers, batch_first = True, bidirectional = True)
        self.graph_gru.name = 'graph_gru'
        self.graph_gru.register_backward_hook(self.update_momenta)
        
        if not self.use_conv:
            self.out = MLP(hidden_size * num_gru_layers * 4 + global_embedding_dim, n_classes, dim = hidden_size * num_gru_layers * 2)
        else:
            self.out = MLP(hidden_size * num_gru_layers * 4 + L * conv_dim + global_embedding_dim, n_classes, dim = hidden_size * num_gru_layers * 4)
        self.out.name = 'out_mlp'
        self.out.register_backward_hook(self.update_momenta)
            
        self.soft = nn.LogSoftmax(dim = -1)
        self.relu = nn.ReLU(inplace = False)
        
        self.momenta_gamma = momenta_gamma
        self.init_momenta()
        
    def init_momenta(self):
        self.momenta = dict()
        
    def update_momenta(self, module, grad_input, grad_output):
        
        if not (module.name + '_input.0') in self.momenta.keys():
            if hasattr(module, 'weight'):
                if module.weight.requires_grad:
                    self.momenta[module.name + '_weight'] = np.abs(module.weight.grad)
            
            for i, grad in enumerate(grad_input):
                if grad is not None:
                    self.momenta[module.name + '_input.{}'.format(i)] = np.abs(grad.mean(dim = 0).detach().cpu().numpy())
                    
            for i, grad in enumerate(grad_output):
                if grad is not None:
                    self.momenta[module.name + '_output.{}'.format(i)] = np.abs(grad.mean(dim = 0).detach().cpu().numpy()) 
        else:
            if hasattr(module, 'weight'):
                if module.weight.requires_grad:
                    self.momenta[module.name + '_weight'] = (1 - self.momenta_gamma) * self.momenta[module.name + 'weight'] \
                                                                    + self.momenta_gamma * np.abs(module.weight.grad)
            
            for i, grad in enumerate(grad_input):
                if grad is not None:
                    self.momenta[module.name + '_input.{}'.format(i)] = (1 - self.momenta_gamma) * self.momenta[module.name + '_input.{}'.format(i)] \
                                                            + self.momenta_gamma * np.abs(grad.mean(dim = 0).detach().cpu().numpy())          
            for i, grad in enumerate(grad_output):
                if grad is not None:
                    self.momenta[module.name + '_output.{}'.format(i)] = (1 - self.momenta_gamma) * self.momenta[module.name + '_output.{}'.format(i)] \
                                                            + self.momenta_gamma * np.abs(grad.mean(dim = 0).detach().cpu().numpy())  
   

        
    def forward(self, x0, edge_index, batch, x1, x2):
        x = torch.cat([self.embedding(x0), x0], dim = -1)
        x2 = self.relu(self.global_embedding_norm(self.global_embedding(x2)))
        
        for ix in range(self.n_gcn_iter):
            x = self.norms[ix](self.gcns[ix](x, edge_index) + x)    
            x = self.act(x)
            
        x = torch.cat([x0, x], dim = -1)
       
        x = to_dense_batch(x, batch)[0]
        _, h = self.graph_gru(x)
        x = torch.flatten(h.transpose(0, 1), 1, 2)
        
        n_batch = self.batch_size
        
        x = x.view((n_batch, self.L, x.shape[-1]))
        x = torch.cat([x, x1], dim = -1)
        
        _, h = self.gru(x)
        h = torch.flatten(h.transpose(0, 1), 1, 2)
        
        if self.use_conv:
            xc = self.conv(x.transpose(1, 2)).flatten(1, 2)
            h = torch.cat([h, xc], dim = 1)
        
        h = torch.cat([h, x2], dim = 1)
        
        return self.out(h)
        

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
                             k = 5, pooling = None, name = 'res_conv'):
        super(Res1dBlock, self).__init__()
  
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        self.name = name
        
        for ix in range(n_layers):
            
            self.convs.append(nn.Conv1d(in_shape, out_channels, k, 
                                        stride = 1, padding = (k - 1) // 2))
            self.norms.append(nn.Sequential(nn.InstanceNorm1d(out_channels)))
            
            in_shape = out_channels
        
        if pooling == 'max':
            self.pool = nn.MaxPool2d((1, 2), stride = (1, 2))
        else:
            self.pool = None
            
        self.activation = nn.ReLU()
        
    def forward(self, x, return_unpooled = False):
        x = self.norms[0](self.convs[0](x))
        
        for ix in range(1, len(self.norms)):
            x = self.norms[ix](self.convs[ix](x)) + x
            
        x = self.activation(x)
        
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
