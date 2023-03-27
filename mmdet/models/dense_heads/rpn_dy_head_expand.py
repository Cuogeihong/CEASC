# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, normal_init, bias_init_with_prob
from mmcv.ops import batched_nms

from ..builder import HEADS
from .anchor_dy_head import AnchorDYHead
from .cuda_dynamic_conv_module import DyConv2D
from .sparseconv_utils import *


@HEADS.register_module()
class RPNDYHeadExpand(AnchorDYHead):
    """RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict or list[dict], optional): Initialization config dict.
        num_convs (int): Number of convolution layers in the head. Default 1.
    """  # noqa: W605

    def __init__(self,
                 in_channels,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 init_cfg=None,
                 num_convs=4,
                 mask_kernel_size=3,
                 return_hards=False,
                 reg_base=0,
                 cls_base=-5,
                 **kwargs):
        self.num_convs = num_convs
        self.norm_cfg = norm_cfg

        self.return_hards = return_hards
        self.mask_kernel_size = mask_kernel_size
        self.num_groups = 32
        self.eps = 1e-5
        self.reg_base = reg_base
        self.cls_base = cls_base

        super(RPNDYHeadExpand, self).__init__(
            1, in_channels, init_cfg=init_cfg, **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        if self.num_convs > 1:
            self.rpn_convs = nn.ModuleList()
            self.pw_convs = torch.nn.Sequential(
                torch.nn.Conv2d(self.in_channels, self.feat_channels, kernel_size=1),
            )
            for i in range(self.num_convs):
                if i == 0:
                    in_channels = self.in_channels
                else:
                    in_channels = self.feat_channels

                self.rpn_convs.append(
                    nn.Sequential(
                        DyConv2D(
                            in_channels,
                            self.feat_channels,
                            kernel_size=3,
                            padding=1,
                            stride=1,
                        ),
                    )
                )
        else:
            assert False, ('error, rpn head expand need more than 1 convs!')
        self.rpn_cls = DyConv2D(
                        self.feat_channels,
                        self.num_base_priors * self.cls_out_channels,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        base=self.cls_base,
                        gn_inside=False,
                    )

        self.rpn_reg = DyConv2D(
                        self.feat_channels,
                        self.num_base_priors * 4,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        base=self.reg_base,
                        gn_inside=False,
                    )
        self.rpn_mask= nn.Sequential(
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
        for m in self.rpn_convs:
            normal_init(m[0].conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.rpn_cls.conv, std=0.01, bias=bias_cls)
        normal_init(self.rpn_reg.conv, std=0.01)

    def forward_single(self, x):
        """Forward feature map of a single scale level."""
        hards = []
        active_positions = []
        total_positions = [] 
        mse_losses = []

        mask = self.rpn_mask(x)
        pws = self.pw_convs(x)

        if not self.training:
            pws = self.cal_gn_distribution(pws)

        for conv in self.rpn_convs:
            x, loss = conv((x, mask, pws))
            x = self.a_relu(x)
            tmask = mask.hard
            if self.return_hards:
                hards.append(tmask)
            if self.training:
                active_positions.append(torch.sum(tmask))
                total_positions.append(tmask.numel())
                mse_losses.append(loss)
        rpn_cls_score = self.rpn_cls((x, mask, None))
        rpn_bbox_pred = self.rpn_reg((x, mask, None))
        mask = mask.hard
        if self.training:
            active_positions.append(torch.sum(mask))
            total_positions.append(mask.numel())
            active_positions.append(torch.sum(mask))
            total_positions.append(mask.numel())
            return rpn_cls_score, rpn_bbox_pred, active_positions, total_positions, mse_losses
        else:
            if self.return_hards:
                return rpn_cls_score, rpn_bbox_pred, hards
            return rpn_cls_score, rpn_bbox_pred

    def cal_gn_distribution(self, pw):
        G = self.num_groups
        N, C, H, W = pw.size()
        pw = pw.view(N, G, -1)
        mean_part = pw.mean(-1, keepdim=True)
        var_part = pw.var(-1, keepdim=True)
        rstd_part = 1 / torch.sqrt(var_part + self.eps)
        pw = pw.view(N, C, H, W)
        return (pw, mean_part, rstd_part)


    def loss(self,
             cls_scores,
             bbox_preds,
             active_positions,
             total_positions,
             mse_losses,
             gt_bboxes,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        losses = super(RPNDYHeadExpand, self).loss(
            cls_scores,
            bbox_preds,
            active_positions,
            total_positions,
            mse_losses,
            gt_bboxes,
            None,
            img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'], 
            loss_cost=losses['loss_cost'], loss_gn=losses['loss_gn'], cost=losses['cost'])

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_anchors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_anchors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has
                shape (num_anchors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image. RPN head does not need this value.
            mlvl_anchors (list[Tensor]): Anchors of all scale level
                each item has shape (num_anchors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']

        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        nms_pre = cfg.get('nms_pre', -1)
        for level_idx in range(len(cls_score_list)):
            rpn_cls_score = cls_score_list[level_idx]
            rpn_bbox_pred = bbox_pred_list[level_idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # We set FG labels to [0, num_class-1] and BG label to
                # num_class in RPN head since mmdet v2.5, which is unified to
                # be consistent with other head since mmdet v2.0. In mmdet v2.0
                # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
                scores = rpn_cls_score.softmax(dim=1)[:, 0]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            anchors = mlvl_anchors[level_idx]
            if 0 < nms_pre < scores.shape[0]:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:nms_pre]
                scores = ranked_scores[:nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]

            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ),
                                level_idx,
                                dtype=torch.long))

        return self._bbox_post_process(mlvl_scores, mlvl_bbox_preds,
                                       mlvl_valid_anchors, level_ids, cfg,
                                       img_shape)

    def _bbox_post_process(self, mlvl_scores, mlvl_bboxes, mlvl_valid_anchors,
                           level_ids, cfg, img_shape, **kwargs):
        """bbox post-processing method.

        Do the nms operation for bboxes in same level.

        Args:
            mlvl_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_bboxes (list[Tensor]): Decoded bboxes from all scale
                levels of a single image, each item has shape (num_bboxes, 4).
            mlvl_valid_anchors (list[Tensor]): Anchors of all scale level
                each item has shape (num_bboxes, 4).
            level_ids (list[Tensor]): Indexes from all scale levels of a
                single image, each item has shape (num_bboxes, ).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, `self.test_cfg` would be used.
            img_shape (tuple(int)): The shape of model's input image.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bboxes)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        if cfg.min_bbox_size >= 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                proposals = proposals[valid_mask]
                scores = scores[valid_mask]
                ids = ids[valid_mask]

        if proposals.numel() > 0:
            dets, _ = batched_nms(proposals, scores, ids, cfg.nms)
        else:
            return proposals.new_zeros(0, 5)

        return dets[:cfg.max_per_img]

    def onnx_export(self, x, img_metas):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): Meta info of each image.
        Returns:
            Tensor: dets of shape [N, num_det, 5].
        """
        cls_scores, bbox_preds = self(x)

        assert len(cls_scores) == len(bbox_preds)

        batch_bboxes, batch_scores = super(RPNDYHeadExpand, self).onnx_export(
            cls_scores, bbox_preds, img_metas=img_metas, with_nms=False)
        # Use ONNX::NonMaxSuppression in deployment
        from mmdet.core.export import add_dummy_nms_for_onnx
        cfg = copy.deepcopy(self.test_cfg)
        score_threshold = cfg.nms.get('score_thr', 0.0)
        nms_pre = cfg.get('deploy_nms_pre', -1)
        # Different from the normal forward doing NMS level by level,
        # we do NMS across all levels when exporting ONNX.
        dets, _ = add_dummy_nms_for_onnx(batch_bboxes, batch_scores,
                                         cfg.max_per_img,
                                         cfg.nms.iou_threshold,
                                         score_threshold, nms_pre,
                                         cfg.max_per_img)
        return dets
