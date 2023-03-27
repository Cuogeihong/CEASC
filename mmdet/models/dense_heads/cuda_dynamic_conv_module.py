import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmcv.cnn import build_norm_layer

from .sparse_conv_net import *

class DyConv2D(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True,
                 dense=False,
                 base=0,
                 gn_inside=True):
        super(DyConv2D, self).__init__()
        self.dense = dense
        self.base = base,
        self.gn_inside = gn_inside
        if dense:
            self.conv = nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size,
                                stride=1,
                                padding=1,
                                bias=bias)
        else:
            self.conv = SparseConv2d(
                            in_channels,
                            out_channels,
                            kernel_size,
                            stride=stride,
                            padding=padding,
                            bias=bias,
                            base=base)
        if self.gn_inside:
            self.gn = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-05)
    
    def forward(self, inputs_meta):
        # TODO: pw has different values while inference, which should be avoided
        inputs, mask, pw = inputs_meta
        if self.training:
            pw_back = pw
        elif self.gn_inside:
            pw_back = pw[0]
        if self.dense:
            out = self.conv(inputs)
        else:    
            if self.gn_inside:
                if self.training:
                    out, mse = self.conv(inputs, mask, pw, gn=self.gn)
                    out = out + pw_back
                    return out, mse
                else:
                    return self.conv(inputs, mask, (pw[1], pw[2]), gn=self.gn) + pw_back, None
            else:
                return self.conv(inputs, mask)

