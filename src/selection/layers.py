# -*- coding: utf-8 -*-
import torch.nn as nn
import numpy as np
from collections import defaultdict
from typing import List, Type, Callable, Union, Optional, List, Any


from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout
#from networkx.algorithms.bipartite.matrix import from_biadjacency_matrix
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import scipy.signal
import scipy.optimize
import torch


from torch import autograd


import warnings
from torch import Tensor

from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_max, scatter, scatter_mean, scatter_std

#from sparsenn.models.gcn.layers import DynamicGraphResBlock, GraphCyclicGRUBlock, GraphInceptionBlock
from torch_geometric.nn import global_mean_pool, MessageNorm
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax, remove_self_loops

from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import LayerNorm

from torch.nn import Parameter

from typing import Union, Tuple, Optional, Callable
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch.nn.functional as F

from torch_geometric.nn import inits
import math

from torch_sparse import SparseTensor, set_diag

class RNNSegmenter(torch.nn.Module):
    def __init__(self, window_size = 128):
        return


#updated LexStyleNet with model from paper
class LexSelectionNet(nn.Module):
    def __init__(self,h = 208, w = 508):
        super(LexSelectionNet, self).__init__()
        
        self.firstconv = nn.Conv1d(h,256,2)
        self.convs = nn.ModuleList()
        self.down = nn.Dropout(.2)
        
        in_channels = 256 #lex's 256  #original 34
        out_channels = [256, 256, 256,256] #original add 256 #lex's remove 256
        for ix in range(4):   
            self.convs.append(nn.Sequential(nn.Conv1d(in_channels, out_channels[ix], 2), 
                                            nn.InstanceNorm1d(out_channels[ix]), 
                                            nn.ReLU(), 
                                            nn.AvgPool1d(2)
            ))  #original was .25
            
            in_channels = copy.copy(out_channels[ix])
            
            w = w // 2
        
        features = 3
        
        self.out_size = 8064

        self.convs2 = nn.Sequential(nn.Linear(4096,64),nn.LayerNorm((64,)), nn.ReLU())#,nn.Dropout(.1))

        self.out = nn.Sequential(nn.Linear(65344, 256), nn.LayerNorm((256,)), nn.ReLU(),nn.Dropout(.25),
                                 nn.Linear(256, 5))#<- May need to implement LogSoftmax!!! 79680 #nn.Dropout(.25) after relu maybe
        
    def forward(self, x):
        x1 = self.firstconv(x[0])  #add this for lex's
        for ix in range(len(self.convs)):
            x1 = self.convs[ix](x1)
            x1 = self.down(x1)
        
        x1 = x1.flatten(1,2)
        x2 = self.convs2(x[1])
        x_out = torch.cat((x1,x2),dim=1)
        
        return self.out(x_out)

class LexNet_EXACT(nn.Module):
    def __init__(self,h = 208, w = 508):
        super(LexNet_EXACT, self).__init__()
        
        self.firstconv = nn.Conv1d(h,256,2)
        self.convs = nn.ModuleList()
        self.down = nn.Dropout(.2)
        
        
        self.convs.append(
            nn.Sequential(
                nn.Conv1d(256, 256, 2), 
                nn.InstanceNorm1d(256), 
                nn.ReLU(), 
                nn.MaxPool1d(2),
                nn.Dropout(.2),

                nn.Conv1d(256, 256, 2), 
                nn.InstanceNorm1d(256), 
                nn.ReLU(), 
                nn.MaxPool1d(2),
                nn.Dropout(.2),

                nn.Conv1d(256, 256, 2), 
                nn.InstanceNorm1d(256), 
                nn.ReLU(), 
                nn.AvgPool1d(2),
                nn.Dropout(.2),

                nn.Conv1d(256, 256, 2), 
                nn.InstanceNorm1d(256), 
                nn.ReLU(), 
                nn.AvgPool1d(2),
                nn.Dropout(.2)
            ))  
            
        
        features = 3
        
        self.out_size = 8064

        self.convs2 = nn.Sequential(nn.Linear(5000,64),nn.Dropout(.1)) #nn.LayerNorm((64,)), nn.ReLU(),

        self.out = nn.Sequential(nn.Linear(79680, 256),nn.Dropout(.25), # nn.LayerNorm((256,)), nn.ReLU(),
                                 nn.Linear(256, 5))#,nn.Softmax(dim=1))#<- May need to implement LogSoftmax!!! 79680 with 5000,65344 with 4096 #nn.Dropout(.25) after relu maybe
        
    def forward(self, x):
        x1 = self.firstconv(x[0])  #add this for lex's
        x1 = self.convs[0](x1)
        #x1 = self.down(x1)
        
        x1 = x1.flatten(1,2)
        x2 = self.convs2(x[1])
        x_out = torch.cat((x1,x2),dim=1)
        
        return self.out(x_out)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 5,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(512 * block.expansion, num_classes), nn.LogSoftmax(dim = -1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
        

def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model

def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)

def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
​
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


# class LexStyleNet(nn.Module):
#     def __init__(self, h = 34, w = 508, n_layers = 3):
#         super(LexStyleNet, self).__init__()

#         self.convs = nn.ModuleList()
        
#         self.down = nn.AvgPool1d(2)
        
#         in_channels = h
#         out_channels = [256, 128, 128]
#         for ix in range(n_layers):
#             self.convs.append(nn.Sequential(nn.Conv1d(in_channels, out_channels[ix], 2), nn.InstanceNorm1d(out_channels[ix]), nn.ReLU(), nn.Dropout(0.25)))
            
#             in_channels = copy.copy(out_channels[ix])
            
#         self.out = nn.Sequential(nn.Linear(3840, 128), nn.LayerNorm((128,)), nn.ReLU(),
#                                  nn.Linear(128, 3), nn.LogSoftmax(dim = -1))   #when kernel=2, 31872
#     def forward(self, x):
#         for ix in range(len(self.convs)):
#             x = self.convs[ix](x)
#             x = self.down(x)
        
#         x = x.flatten(1, 2)
        
#         return self.out(x)
