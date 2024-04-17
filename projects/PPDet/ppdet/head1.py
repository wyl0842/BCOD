# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import Scale, ConvModule
from mmengine.structures import InstanceData
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.models.layers import NormedConv2d
from mmdet.registry import MODELS
from mmdet.utils import (ConfigType, InstanceList, MultiConfig, OptMultiConfig,
                         OptInstanceList, RangeType, reduce_mean)
from mmdet.models.utils import multi_apply
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead

from mmcv.ops.deform_conv import DeformConv2d
from .bbox_nms import multiclass_nms

INF = 1e8
StrideType = Union[Sequence[int], Sequence[Tuple[int, int]]]

class FeatureAlign(BaseModule):
    """Feature Alignment Module.

    Feature Alignment Module is implemented based on DCN v1.
    It uses anchor shape prediction rather than feature map to
    predict offsets of deformable conv layer.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (int): Deformable conv kernel size.
        deformable_groups (int): Deformable conv group size.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 deformable_groups: int = 4,
                 init_cfg: OptMultiConfig = dict(
            type='Normal', layer='Conv2d', std=0.01, override=dict(type='Normal', name='conv_adaption', std=0.01))):
        super().__init__(init_cfg=init_cfg)
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(4,
                                     deformable_groups * offset_channels,
                                     1,
                                     bias=False)
        self.conv_adaption = DeformConv2d(in_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        padding=(kernel_size - 1) // 2,
                                        deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, shape):
        offset = self.conv_offset(shape)
        x = self.relu(self.conv_adaption(x, offset))
        return x

@MODELS.register_module()
class PPDetHead(AnchorFreeHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to suppress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (Sequence[int] or Sequence[Tuple[int, int]]): Strides of points
            in multiple feature levels. Defaults to (4, 8, 16, 32, 64).
        regress_ranges (Sequence[Tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling.
            Defaults to False.
        center_sample_radius (float): Radius of center sampling.
            Defaults to 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets with
            FPN strides. Defaults to False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Defaults to False.
        conv_bias (bool or str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias of conv will be set as True if `norm_cfg` is
            None, otherwise False. Defaults to "auto".
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_centerness (:obj:`ConfigDict`, or dict): Config of centerness
            loss.
        norm_cfg (:obj:`ConfigDict` or dict): dictionary to construct and
            config norm layer.  Defaults to
            ``norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)``.
        cls_predictor_cfg (:obj:`ConfigDict` or dict): dictionary to construct and
            config conv_cls. Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 feat_channels: int=256,
                 stacked_convs: int=4,
                 strides: StrideType = (4, 8, 16, 32, 64),
                 base_edge_list: StrideType = (16, 32, 64, 126, 256),
                 scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
                #  regress_ranges: RangeType = ((-1, 64), (64, 128), (128, 256),
                #                               (256, 512), (512, INF)),
                 sigma: float = 0.4,
                 with_deform: bool = False,
                 deformable_groups: int = 4,
                #  center_sampling: bool = False,
                #  center_sample_radius: float = 1.5,
                #  norm_on_bbox: bool = False,
                #  centerness_on_reg: bool = False,
                 loss_cls: ConfigType = dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=1.50,
                     alpha=0.4,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.75),
                #  loss_centerness=None,
                 norm_cfg=None,
                 cls_predictor_cfg=None,
                 init_cfg: MultiConfig = dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=[
                        dict(
                         type='Normal',
                         name='convs_cls',
                         std=0.01,
                         bias_prob=0.01),
                        dict(
                         type='Normal',
                         name='convs_reg',
                         std=0.01,
                         bias_prob=0.01)]),
                 **kwargs) -> None:
        # self.regress_ranges = regress_ranges
        # self.center_sampling = center_sampling
        # self.center_sample_radius = center_sample_radius
        # self.norm_on_bbox = norm_on_bbox
        # self.centerness_on_reg = centerness_on_reg
        self.cls_predictor_cfg = cls_predictor_cfg
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges
        self.sigma = sigma
        self.with_deform = with_deform
        self.deformable_groups = deformable_groups
        self.norm_cfg = norm_cfg
        self.init_layers()
        # self.loss_centerness = MODELS.build(loss_centerness)

    def init_layers(self) -> None:
        """Initialize layers of the head."""
        # super()._init_layers()
        # self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        # if self.cls_predictor_cfg is not None:
        #     self.cls_predictor_cfg.pop('type')
        #     self.conv_cls = NormedConv2d(
        #         self.feat_channels,
        #         self.cls_out_channels,
        #         1,
        #         padding=0,
        #         **self.cls_predictor_cfg)
        self.convs_cls = nn.ModuleList()
        self.convs_reg = nn.ModuleList()
        # box branch
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.convs_reg.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.ppdet_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        # cls branch
        if not self.with_deform:
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                self.convs_cls.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        bias=self.norm_cfg is None))
            self.ppdet_cls = nn.Conv2d(
                self.feat_channels, self.cls_out_channels, 3, padding=1)
        else:
            self.convs_cls.append(
                ConvModule(
                    self.feat_channels,
                    (self.feat_channels * 4),
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))

            self.convs_cls.append(
                ConvModule(
                    (self.feat_channels * 4),
                    (self.feat_channels * 4),
                    1,
                    stride=1,
                    padding=0,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))

            self.feature_adaption = FeatureAlign(
                self.feat_channels,
                self.feat_channels,
                kernel_size=3,
                deformable_groups=self.deformable_groups)

            self.ppdet_cls = nn.Conv2d(
                int(self.feat_channels * 4), self.cls_out_channels, 3, padding=1)

    def forward(
            self, x: Tuple[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of each level outputs.

            - cls_scores (list[Tensor]): Box scores for each scale level, \
            each is a 4D-tensor, the channel number is \
            num_points * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for each \
            scale level, each is a 4D-tensor, the channel number is \
            num_points * 4.
            - centernesses (list[Tensor]): centerness for each scale level, \
            each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, x)

    def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj:`mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness
            predictions of input feature maps.
        """
        # cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
        # if self.centerness_on_reg:
        #     centerness = self.conv_centerness(reg_feat)
        # else:
        #     centerness = self.conv_centerness(cls_feat)
        # # scale the bbox_pred of different level
        # # float to avoid overflow when enabling FP16
        # bbox_pred = scale(bbox_pred).float()
        # if self.norm_on_bbox:
        #     # bbox_pred needed for gradient computation has been modified
        #     # by F.relu(bbox_pred) when run with PyTorch 1.10. So replace
        #     # F.relu(bbox_pred) with bbox_pred.clamp(min=0)
        #     bbox_pred = bbox_pred.clamp(min=0)
        #     if not self.training:
        #         bbox_pred *= stride
        # else:
        #     bbox_pred = bbox_pred.exp()
        # return cls_score, bbox_pred, centerness
        cls_feat = x
        reg_feat = x
        for reg_layer in self.convs_reg:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.ppdet_reg(reg_feat)
        if self.with_deform:
            cls_feat = self.feature_adaption(cls_feat, bbox_pred.exp())
        for cls_layer in self.convs_cls:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.ppdet_cls(cls_feat)
        return cls_score, bbox_pred
    
    # 获取anchor点的位置
    def get_points(self, featmap_sizes, dtype, device, flatten=False):
        points = []
        for featmap_size in featmap_sizes:
            x_range = torch.arange(featmap_size[1], dtype=dtype, device=device) + 0.5
            y_range = torch.arange(featmap_size[0], dtype=dtype, device=device) + 0.5
            # 生成坐标
            y, x = torch.meshgrid(y_range, x_range)
            if flatten:
                points.append((y.flatten(), x.flatten()))
            else:
                points.append((y, x))

        return points

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        # centernesses: List[Tensor],
        # gt_bbox_list: List[Tensor],
        # gt_label_list: List[Tensor],
        batch_gt_instances: InstanceList,
        # batch_gt_instances: InstanceData,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        # all_level_points = self.prior_generator.grid_priors(
        #     featmap_sizes,
        #     dtype=bbox_preds[0].dtype,
        #     device=bbox_preds[0].device)
        # labels, bbox_targets = self.get_targets(all_level_points,
        #                                         batch_gt_instances)
        # 获取anchor点的位置
        points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                 bbox_preds[0].device)
        
        gt_bbox_list = []
        gt_label_list = []
        for each_gt_instances in batch_gt_instances:
            # [tensor([[756., 117., 835., 148.]], device='cuda:0'), 
            #  tensor([[165., 286., 274., 333.]], device='cuda:0')]
            gt_bbox_list.append(each_gt_instances.bboxes)
            # [tensor([5], device='cuda:0'), 
            # tensor([6], device='cuda:0')]
            gt_label_list.append(each_gt_instances.labels)
        
        # tensor([5, 6], device='cuda:0')
        all_gt_label_list = torch.cat([x for x in gt_label_list])
        temp = torch.tensor(0).cuda()
        label_size_list = []
        for x in gt_label_list:
            # [tensor([0], device='cuda:0'), 
            # tensor([1], device='cuda:0'),
            # tensor([2], device='cuda:0')]
            label_size_list.append(temp)
            temp = torch.tensor(x.size()).cuda() + label_size_list[-1]

        label_list, bbox_target_list, gt_ids_list = multi_apply(
            self.ppdet_target_single,
            gt_bbox_list,
            gt_label_list,
            label_size_list,
            featmap_size_list=featmap_sizes,
            point_list=points)

        flatten_labels = [
            torch.cat([labels_level_img.flatten()
                       for labels_level_img in labels_level])
            for labels_level in zip(*label_list)
        ]
        flatten_bbox_targets = [
            torch.cat([bbox_targets_level_img.reshape(-1, 4)
                       for bbox_targets_level_img in bbox_targets_level])
            for bbox_targets_level in zip(*bbox_target_list)
        ]
        flatten_ids = [
            torch.cat([gt_ids_level_img.flatten()
                       for gt_ids_level_img in gt_ids_level])
            for gt_ids_level in zip(*gt_ids_list)
        ]

        flatten_labels = torch.cat(flatten_labels)
        flatten_bbox_targets = torch.cat(flatten_bbox_targets)
        flatten_ids = torch.cat(flatten_ids)
        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        pos_inds = (flatten_labels > 0).nonzero().view(-1)
        num_pos = len(pos_inds)


        neg_inds = (flatten_labels <= 0).nonzero().view(-1)
        agg_labels = flatten_labels[neg_inds]
        agg_cls_scores = flatten_cls_scores[neg_inds]
        num_agg_pos = 0

        for i, class_id in enumerate(all_gt_label_list):

            aggregation_indices = (flatten_ids == i).nonzero()

            if flatten_labels[aggregation_indices].size()[0] != 0:
                agg_labels = torch.cat((all_gt_label_list[i:i+1], agg_labels))
                agg_cls_scores = torch.cat((flatten_cls_scores[aggregation_indices].mean(dim=0), agg_cls_scores))
                num_agg_pos +=1

        loss_cls = self.loss_cls(agg_cls_scores, agg_labels, avg_factor=num_agg_pos + num_imgs)

        if num_pos > 0:
            pos_bbox_preds = flatten_bbox_preds[pos_inds]
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_weights = pos_bbox_targets.new_zeros(pos_bbox_targets.size())+1.0
            loss_bbox = self.loss_bbox(pos_bbox_preds,
                pos_bbox_targets, pos_weights, avg_factor = num_pos)
        else:
            loss_bbox = torch.tensor([0], dtype=flatten_bbox_preds.dtype, device=flatten_bbox_preds.device)
        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox)
        
        # num_imgs = cls_scores[0].size(0)
        # # flatten cls_scores, bbox_preds and centerness
        # flatten_cls_scores = [
        #     cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        #     for cls_score in cls_scores
        # ]
        # flatten_bbox_preds = [
        #     bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        #     for bbox_pred in bbox_preds
        # ]
        # flatten_centerness = [
        #     centerness.permute(0, 2, 3, 1).reshape(-1)
        #     for centerness in centernesses
        # ]
        # flatten_cls_scores = torch.cat(flatten_cls_scores)
        # flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        # flatten_centerness = torch.cat(flatten_centerness)
        # flatten_labels = torch.cat(labels)
        # flatten_bbox_targets = torch.cat(bbox_targets)
        # # repeat points to align with bbox_preds
        # flatten_points = torch.cat(
        #     [points.repeat(num_imgs, 1) for points in all_level_points])

        # losses = dict()

        # # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        # bg_class_ind = self.num_classes
        # pos_inds = ((flatten_labels >= 0)
        #             & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        # num_pos = torch.tensor(
        #     len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        # num_pos = max(reduce_mean(num_pos), 1.0)
        # loss_cls = self.loss_cls(
        #     flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        # if getattr(self.loss_cls, 'custom_accuracy', False):
        #     acc = self.loss_cls.get_accuracy(flatten_cls_scores,
        #                                      flatten_labels)
        #     losses.update(acc)

        # pos_bbox_preds = flatten_bbox_preds[pos_inds]
        # pos_centerness = flatten_centerness[pos_inds]
        # pos_bbox_targets = flatten_bbox_targets[pos_inds]
        # pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # # centerness weighted iou loss
        # centerness_denorm = max(
        #     reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        # if len(pos_inds) > 0:
        #     pos_points = flatten_points[pos_inds]
        #     pos_decoded_bbox_preds = self.bbox_coder.decode(
        #         pos_points, pos_bbox_preds)
        #     pos_decoded_target_preds = self.bbox_coder.decode(
        #         pos_points, pos_bbox_targets)
        #     loss_bbox = self.loss_bbox(
        #         pos_decoded_bbox_preds,
        #         pos_decoded_target_preds,
        #         weight=pos_centerness_targets,
        #         avg_factor=centerness_denorm)
        #     loss_centerness = self.loss_centerness(
        #         pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        # else:
        #     loss_bbox = pos_bbox_preds.sum()
        #     loss_centerness = pos_centerness.sum()

        # losses['loss_cls'] = loss_cls
        # losses['loss_bbox'] = loss_bbox
        # losses['loss_centerness'] = loss_centerness

        # return losses

    def ppdet_target_single(self,
                            gt_bboxes_raw,
                            gt_labels_raw,
                            label_size_list_raw,
                            featmap_size_list=None,
                            point_list=None):

        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))
        label_list = []
        bbox_target_list = []
        ids_list = []
        for base_len, (lower_bound, upper_bound), stride, featmap_size, (y, x) \
                in zip(self.base_edge_list, self.scale_ranges, self.strides, featmap_size_list, point_list):
            labels = gt_labels_raw.new_zeros(featmap_size)
            bbox_targets = gt_bboxes_raw.new(featmap_size[0], featmap_size[1], 4) + 1
            gt_ids = gt_labels_raw.new_zeros(featmap_size) - 1
            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            if len(hit_indices) == 0:
                label_list.append(labels)
                bbox_target_list.append(torch.log(bbox_targets))
                ids_list.append(gt_ids)
                continue

            _, hit_index_order = torch.sort(-gt_areas[hit_indices])
            hit_indices = hit_indices[hit_index_order]
            gt_bboxes = gt_bboxes_raw[hit_indices, :] / stride
            gt_labels = gt_labels_raw[hit_indices]
            half_w = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0])
            half_h = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
            pos_left = torch.ceil(gt_bboxes[:, 0] + (1 - self.sigma) * half_w - 0.5).long().\
                clamp(0, featmap_size[1] - 1)
            pos_right = torch.floor(gt_bboxes[:, 0] + (1 + self.sigma) * half_w - 0.5).long().\
                clamp(0, featmap_size[1] - 1)
            pos_top = torch.ceil(gt_bboxes[:, 1] + (1 - self.sigma) * half_h - 0.5).long().\
                clamp(0, featmap_size[0] - 1)
            pos_down = torch.floor(gt_bboxes[:, 1] + (1 + self.sigma) * half_h - 0.5).long().\
                clamp(0, featmap_size[0] - 1)
            for px1, py1, px2, py2, label, gt_id, (gt_x1, gt_y1, gt_x2, gt_y2) in \
                    zip(pos_left, pos_top, pos_right, pos_down, gt_labels, hit_indices,
                        gt_bboxes_raw[hit_indices, :]):
                labels[py1:py2 + 1, px1:px2 + 1] = label
                gt_ids[py1:py2 + 1, px1:px2 + 1] = gt_id + label_size_list_raw
                bbox_targets[py1:py2 + 1, px1:px2 + 1, 0] = (stride * x[py1:py2 + 1, px1:px2 + 1] - gt_x1) / base_len
                bbox_targets[py1:py2 + 1, px1:px2 + 1, 1] = (stride * y[py1:py2 + 1, px1:px2 + 1] - gt_y1) / base_len
                bbox_targets[py1:py2 + 1, px1:px2 + 1, 2] = (gt_x2 - stride * x[py1:py2 + 1, px1:px2 + 1]) / base_len
                bbox_targets[py1:py2 + 1, px1:px2 + 1, 3] = (gt_y2 - stride * y[py1:py2 + 1, px1:px2 + 1]) / base_len
            bbox_targets = bbox_targets.clamp(min=1./16, max=16.)
            label_list.append(labels)
            bbox_target_list.append(torch.log(bbox_targets))
            ids_list.append(gt_ids)
        return label_list, bbox_target_list, ids_list
    
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg,
                   rescale=None):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                 bbox_preds[0].device, flatten=True)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self.get_bboxes_single(cls_score_list, bbox_pred_list, featmap_sizes, points,
                                                img_shape, scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def get_bboxes_aug(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg,
                   rescale=None):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                 bbox_preds[0].device, flatten=True)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self.get_bboxes_single_aug(cls_score_list, bbox_pred_list, featmap_sizes, points,
                                                img_shape, scale_factor, cfg, rescale)
            result_list.append(det_bboxes)

        return result_list

    def get_bboxes_single_aug(self,
                          cls_scores,
                          bbox_preds,
                          featmap_sizes,
                          point_list,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False, debug=False):
        assert len(cls_scores) == len(bbox_preds) == len(point_list)
        det_bboxes = []
        det_scores = []
        for cls_score, bbox_pred, featmap_size, stride, base_len, (y, x) in zip(
                cls_scores, bbox_preds, featmap_sizes, self.strides, self.base_edge_list, point_list):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4).exp()
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                y = y[topk_inds]
                x = x[topk_inds]
            x1 = (stride * x - base_len * bbox_pred[:, 0]).clamp(min=0, max=img_shape[1] - 1)
            y1 = (stride * y - base_len * bbox_pred[:, 1]).clamp(min=0, max=img_shape[0] - 1)
            x2 = (stride * x + base_len * bbox_pred[:, 2]).clamp(min=0, max=img_shape[1] - 1)
            y2 = (stride * y + base_len * bbox_pred[:, 3]).clamp(min=0, max=img_shape[0] - 1)
            bboxes = torch.stack([x1, y1, x2, y2], -1)
            det_bboxes.append(bboxes)
            det_scores.append(scores)

        det_bboxes = torch.cat(det_bboxes)

        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)

        det_scores = torch.cat(det_scores)
        padding = det_scores.new_zeros(det_scores.shape[0], 1)
        det_scores = torch.cat([padding, det_scores], dim=1)

        return det_bboxes, det_scores


    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          featmap_sizes,
                          point_list,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False, debug=False):
        assert len(cls_scores) == len(bbox_preds) == len(point_list)
        det_bboxes = []
        det_scores = []
        for cls_score, bbox_pred, featmap_size, stride, base_len, (y, x) in zip(
                cls_scores, bbox_preds, featmap_sizes, self.strides, self.base_edge_list, point_list):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4).exp()
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                y = y[topk_inds]
                x = x[topk_inds]
            x1 = (stride * x - base_len * bbox_pred[:, 0]).clamp(min=0, max=img_shape[1] - 1)
            y1 = (stride * y - base_len * bbox_pred[:, 1]).clamp(min=0, max=img_shape[0] - 1)
            x2 = (stride * x + base_len * bbox_pred[:, 2]).clamp(min=0, max=img_shape[1] - 1)
            y2 = (stride * y + base_len * bbox_pred[:, 3]).clamp(min=0, max=img_shape[0] - 1)
            bboxes = torch.stack([x1, y1, x2, y2], -1)
            det_bboxes.append(bboxes)
            det_scores.append(scores)
        det_bboxes = torch.cat(det_bboxes)
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_scores = torch.cat(det_scores)
        padding = det_scores.new_zeros(det_scores.shape[0], 1)
        det_scores = torch.cat([padding, det_scores], dim=1)
        if debug:
            det_bboxes, det_labels = multiclass_nms(
                det_bboxes,
                det_scores,
                cfg['k'],
                cfg['agg_thr'],
                cfg['score_thr'],
                cfg['nms'],
                cfg['max_per_img'])
        else:
            det_bboxes, det_labels = multiclass_nms(
                det_bboxes,
                det_scores,
                cfg.score_thr,
                cfg.k,
                cfg.agg_thr,
                cfg.nms,
                cfg.max_per_img)
        return det_bboxes, det_labels

    def get_targets(
            self, points: List[Tensor], batch_gt_instances: InstanceList
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.

        Returns:
            tuple: Targets of each level.

            - concat_lvl_labels (list[Tensor]): Labels of each level.
            - concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
            level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_targets_single,
            batch_gt_instances,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_targets_single(
            self, gt_instances: InstanceData, points: Tensor,
            regress_ranges: Tensor,
            num_points_per_lvl: List[int]) -> Tuple[Tensor, Tensor]:
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = len(gt_instances)
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels

        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets

    # def centerness_target(self, pos_bbox_targets: Tensor) -> Tensor:
    #     """Compute centerness targets.

    #     Args:
    #         pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
    #             (num_pos, 4)

    #     Returns:
    #         Tensor: Centerness target.
    #     """
    #     # only calculate pos centerness targets, otherwise there may be nan
    #     left_right = pos_bbox_targets[:, [0, 2]]
    #     top_bottom = pos_bbox_targets[:, [1, 3]]
    #     if len(left_right) == 0:
    #         centerness_targets = left_right[..., 0]
    #     else:
    #         centerness_targets = (
    #             left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
    #                 top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    #     return torch.sqrt(centerness_targets)
