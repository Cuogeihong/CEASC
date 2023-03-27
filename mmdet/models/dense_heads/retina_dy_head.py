# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, normal_init, bias_init_with_prob

from ..builder import HEADS
from .anchor_head import AnchorHead

from .cuda_dynamic_conv_module import DyConv2D
from .sparseconv_utils import *
from .anchor_dy_head import AnchorDYHead


@HEADS.register_module()
class RetinaDYHead(AnchorDYHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=None,
                 mask_kernel_size=3,
                 return_hards=False,
                 beta=10,
                 reg_base=0,
                 cls_base=-5,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.return_hards = return_hards
        self.mask_kernel_size = mask_kernel_size
        self.num_groups = 32
        self.eps = 1e-5
        self.reg_base = reg_base
        self.cls_base = cls_base
        super(RetinaDYHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            beta=beta,
            **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        self.cls_pw_convs = torch.nn.Sequential(
                torch.nn.Conv2d(self.in_channels, self.feat_channels, kernel_size=1),
            )
        self.reg_pw_convs = torch.nn.Sequential(
                torch.nn.Conv2d(self.in_channels, self.feat_channels, kernel_size=1),
            )

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                nn.Sequential(
                    DyConv2D(
                        chn,
                        self.feat_channels,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                    ),
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    DyConv2D(
                        chn,
                        self.feat_channels,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                    ),
                )
            )
        self.retina_cls = DyConv2D(
                        self.feat_channels,
                        self.num_base_priors * self.cls_out_channels,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        base=self.cls_base,
                        gn_inside=False,
                    )

        self.retina_reg = DyConv2D(
                        self.feat_channels,
                        self.num_base_priors * 4,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        base=self.reg_base,
                        gn_inside=False,
                    )
        self.retina_cls_mask= nn.Sequential(
                nn.Conv2d(self.feat_channels,
                            1,
                            self.mask_kernel_size,
                            padding=self.mask_kernel_size//2,
                            stride=1),
                Gumbel()
            )
        self.retina_reg_mask= nn.Sequential(
                nn.Conv2d(self.feat_channels,
                            1,
                            self.mask_kernel_size,
                            padding=self.mask_kernel_size//2,
                            stride=1),
                Gumbel()
            )
        self.a_relu = nn.ReLU(inplace=True)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m[0].conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m[0].conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls.conv, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg.conv, std=0.01)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat = x
        reg_feat = x

        cls_hards = []
        reg_hards = []
        active_positions = [] 
        total_positions = [] 
        mse_losses = []


        cls_mask = self.retina_cls_mask(cls_feat)
        reg_mask = self.retina_reg_mask(reg_feat)
        cls_pws = self.cls_pw_convs(x)
        reg_pws = self.reg_pw_convs(x)

        if not self.training:
            cls_pws = self.cal_gn_distribution(cls_pws)
            reg_pws = self.cal_gn_distribution(reg_pws)

        for cls_conv in self.cls_convs:
            cls_feat, loss = cls_conv((cls_feat, cls_mask, cls_pws))
            cls_feat = self.a_relu(cls_feat)
            mask = cls_mask.hard
            if self.return_hards:
                cls_hards.append(mask)
            if self.training:
                active_positions.append(torch.sum(mask))
                total_positions.append(mask.numel())
                mse_losses.append(loss)

        for reg_conv in self.reg_convs:
            reg_feat, loss = reg_conv((reg_feat, reg_mask, reg_pws))
            reg_feat = self.a_relu(reg_feat)
            mask = reg_mask.hard
            if self.return_hards:
                reg_hards.append(mask)
            if self.training:
                active_positions.append(torch.sum(mask))
                total_positions.append(mask.numel())
                mse_losses.append(loss)

        cls_score = self.retina_cls((cls_feat, cls_mask, None))
        cls_mask = cls_mask.hard
        bbox_pred = self.retina_reg((reg_feat, reg_mask, None))
        reg_mask = reg_mask.hard

        if self.training:
            active_positions.append(torch.sum(cls_mask))
            total_positions.append(cls_mask.numel())
            active_positions.append(torch.sum(reg_mask))
            total_positions.append(reg_mask.numel())
            return cls_score, bbox_pred, active_positions, total_positions, mse_losses
        else:
            if self.return_hards:
                return cls_score, bbox_pred, cls_hards, reg_hards
            return cls_score, bbox_pred

    def cal_gn_distribution(self, pw):
        G = self.num_groups
        N, C, H, W = pw.size()
        pw = pw.view(N, G, -1)
        mean_part = pw.mean(-1, keepdim=True)
        var_part = pw.var(-1, keepdim=True)
        rstd_part = 1 / torch.sqrt(var_part + self.eps)
        pw = pw.view(N, C, H, W)
        return (pw, mean_part, rstd_part)

