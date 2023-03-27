import math
import warnings

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules import Module
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple
import torch.multiprocessing as mp

from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union

# import my_sparse_conv_cpu
import sparse_conv
import os
import time
import pdb



def sparse_gn(x, gn, pw_x):
    N, C, H, W = x.size()

    G = gn.num_groups

    x = x.view(N, G, -1)
    pw_x = pw_x.view(N, G, -1)
    mean_part = pw_x.mean(-1, keepdim=True)
    var_part = pw_x.var(-1, keepdim=True)

    x_part = (x - mean_part) / (var_part + gn.eps).sqrt()
    x_part = x_part.view(N, C, H, W)
    x_part = x_part * gn.weight.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3) + gn.bias.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3)

    return x_part


class Sparse_conv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, hard, weights, bias, stride, padding, isbias, base, gn=None, pw=None, nonzero_hard=None):
        groups = -999
        gnweight = bias
        gnbias = bias
        eps = 1e-5
        if pw == None:
            pw_mean = bias
            pw_rstd = bias
        else:
            pw_mean = pw[0].type_as(input)
            pw_rstd = pw[1].type_as(input)
        if gn != None:
            groups = gn.num_groups
            gnweight = gn.weight.type_as(input)
            gnbias = gn.bias.type_as(input)
            eps = gn.eps
        if str(input.device) == 'cpu':
            assert not str(input.device) == 'cpu', 'we do not support CPU inference, you can try codes in sparse_conv_cpu folder, but we cannot ensure the correctness'
            # output = my_sparse_conv_cpu.forward(input, hard.type_as(input), weights, bias, stride[0], padding[0], isbias, base, groups, gnweight, gnbias, pw_mean, pw_rstd, eps, nonzero_hard[0], nonzero_hard[1])[0]
        else:
            output = sparse_conv.forward(input, hard, weights, bias, stride[0], padding[0], isbias, base, groups, gnweight, gnbias, pw_mean, pw_rstd, eps, nonzero_hard[0], nonzero_hard[1])[0]
        variables = [input, hard, weights, bias, None, None, None, None, None, None, None]
        ctx.save_for_backward(*variables)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert False, "Warning: using sparse conv2d's backward, it should not happen as we do not provide its backward function"
        return None, None, None, None, None, None, None, None, None, None, None


class SparseConv2d(torch.nn.modules.conv._ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        base=0,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:
        self.isbias = bias
        self.base = base
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        
        
        super(SparseConv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode)

    def _slow_forward(self, input, hard, pw, weight, bias, gn):
        hard = hard.hard
        x = F.conv2d(input, weight, bias, self.stride,
                    self.padding, self.dilation, self.groups)   
        if gn != None:
            with torch.no_grad():
                x_total = gn(x)
                x_total = hard * x_total

            assert self.base == 0
            x_part = sparse_gn(x, gn, pw)
            x_part = hard * x_part
            if not self.training:
                return x_part
            return x_part, F.mse_loss(x_part, x_total.detach())

        return hard * x + self.base * (1.0 - hard)

    def _fast_forward(self, input, hard, pw, weight, bias, isbias, base, gn):
        if hard.n_keep == 0:
            one_ = torch.ones((input.shape[0], self.out_channels, input.shape[2], input.shape[3]), dtype=input.dtype, device=input.device)
            return one_ * self.base
        
        nonzero_hard = hard.nonzero_hard
        hard = hard.hard
        
        if not isbias:
            bias_ = torch.ones((0)).to(input.device)
            x = Sparse_conv2d.apply(input, hard, weight, bias_, self.stride,
                        self.padding, isbias, base, gn, pw, nonzero_hard)
        else:
            x = Sparse_conv2d.apply(input, hard, weight, bias, self.stride,
                        self.padding, isbias, base, gn, pw, nonzero_hard) 

        if gn != None:
            x_part = hard * x
            return x_part.type_as(input)
        return (hard * x + self.base * (1.0 - hard)).type_as(input)

    def forward(self, input, hard, pw=None, gn=None):
        if self.training:
            return self._slow_forward(input, hard, pw, self.weight, self.bias, gn)

        else:
            fast_ans = self._fast_forward(input, hard, pw, self.weight, self.bias, self.isbias, self.base, gn)
            return fast_ans


            
