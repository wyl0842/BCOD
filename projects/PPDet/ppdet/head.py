# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import Scale, ConvModule
from mmengine.structures import InstanceData
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import InstanceList, OptMultiConfig, OptInstanceList
from mmdet.models.utils import filter_scores_and_topk, multi_apply
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead

from mmcv.ops import DeformConv2d, batched_nms
from .bbox_nms import agg_scores
from mmdet.structures.bbox import (cat_boxes, get_box_tensor, get_box_wh, scale_boxes)

INF = 1e8
# StrideType = Union[Sequence[int], Sequence[Tuple[int, int]]]

class FeatureAlign(BaseModule):
    """Feature Align Module.

    Feature Align Module is implemented based on DCN v1.
    It uses anchor shape prediction rather than feature map to
    predict offsets of deform conv layer.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (int): Size of the convolution kernel.
            ``norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)``.
        deform_groups: (int): Group number of DCN in
            FeatureAdaption module.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        deform_groups: int = 4,
        init_cfg: OptMultiConfig = dict(
            type='Normal',
            layer='Conv2d',
            std=0.1,
            override=dict(type='Normal', name='conv_adaption', std=0.01))
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(
            4, deform_groups * offset_channels, 1, bias=False)
        self.conv_adaption = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deform_groups=deform_groups)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, shape: Tensor) -> Tensor:
        """Forward function of feature align module.

        Args:
            x (Tensor): Features from the upstream network.
            shape (Tensor): Exponential of bbox predictions.

        Returns:
            x (Tensor): The aligned features.
        """
        offset = self.conv_offset(shape)
        x = self.relu(self.conv_adaption(x, offset))
        return x

@MODELS.register_module()
class PPDetHead(AnchorFreeHead):
    """Detection Head of `FoveaBox: Beyond Anchor-based Object Detector.

    <https://arxiv.org/abs/1904.03797>`_.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        base_edge_list (list[int]): List of edges.
        scale_ranges (list[tuple]): Range of scales.
        sigma (float): Super parameter of ``FoveaHead``.
        with_deform (bool):  Whether use deform conv.
        deform_groups (int): Deformable conv group size.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 base_edge_list: List[int] = (16, 32, 64, 126, 256),
                 scale_ranges: List[tuple] = ((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
                 sigma: float = 0.4,
                 with_deform: bool = False,
                 deform_groups: int = 4,
                 init_cfg: OptMultiConfig = dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='ppdet_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs) -> None:
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges
        self.sigma = sigma
        self.with_deform = with_deform
        self.deform_groups = deform_groups
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            init_cfg=init_cfg,
            **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        # box branch
        super()._init_reg_convs()
        self.ppdet_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)

        # cls branch
        if not self.with_deform:
            super()._init_cls_convs()
            self.ppdet_cls = nn.Conv2d(
                self.feat_channels, self.cls_out_channels, 3, padding=1)
        else:
            self.cls_convs = nn.ModuleList()
            self.cls_convs.append(
                ConvModule(
                    self.feat_channels, 
                    (self.feat_channels * 4),
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.cls_convs.append(
                ConvModule((self.feat_channels * 4), 
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
                deform_groups=self.deform_groups)
            self.ppdet_cls = nn.Conv2d(
                int(self.feat_channels * 4), 
                self.cls_out_channels, 
                3, 
                padding=1)

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
        cls_feat = x
        reg_feat = x
        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.ppdet_reg(reg_feat)
        if self.with_deform:
            cls_feat = self.feature_adaption(cls_feat, bbox_pred.exp())
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.ppdet_cls(cls_feat)
        return cls_score, bbox_pred

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        batch_gt_instances: InstanceList,
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
        priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        
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
        
        gt_bbox_list = []
        gt_label_list = []
        for each_gt_instances in batch_gt_instances:
            gt_bbox_list.append(each_gt_instances.bboxes)
            gt_label_list.append(each_gt_instances.labels)
        
        # tensor([5, 6], device='cuda:0')
        all_gt_label_list = torch.cat([x for x in gt_label_list])
        temp = torch.tensor(0).cuda()
        label_size_list = []
        for x in gt_label_list:
            label_size_list.append(temp)
            temp = torch.tensor(x.size()).cuda() + label_size_list[-1]

        # flatten_ids找出all_gt_label_list里面哪些是符合要求的label，记录序号，将box位置的值设置为序号
        flatten_labels, flatten_bbox_targets, flatten_ids = self.get_targets(
            batch_gt_instances, 
            label_size_list, 
            featmap_sizes, 
            priors)

        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < self.num_classes)).nonzero().view(-1)
        num_pos = len(pos_inds)

        # 找出背景的位置
        neg_inds = (flatten_labels <= 0).nonzero().view(-1)
        # gt标签
        agg_labels = flatten_labels[neg_inds]
        # 预测分数
        agg_cls_scores = flatten_cls_scores[neg_inds]
        num_agg_pos = 0

        for i, class_id in enumerate(all_gt_label_list):

            # 找出符合要求的正区域
            aggregation_indices = (flatten_ids == i).nonzero()

            if flatten_labels[aggregation_indices].size()[0] != 0:
                # 原本的若干标签变为一个完整标签
                agg_labels = torch.cat((all_gt_label_list[i:i+1], agg_labels))
                # 原本的若干分数变为一个完整分数
                agg_cls_scores = torch.cat((flatten_cls_scores[aggregation_indices].mean(dim=0), agg_cls_scores))
                num_agg_pos +=1

        loss_cls = self.loss_cls(agg_cls_scores, agg_labels, avg_factor=num_agg_pos + num_imgs)

        if num_pos > 0:
            pos_bbox_preds = flatten_bbox_preds[pos_inds]
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_weights = pos_bbox_targets.new_ones(pos_bbox_targets.size())
            loss_bbox = self.loss_bbox(
                pos_bbox_preds,
                pos_bbox_targets,
                pos_weights,
                avg_factor=num_pos)
        else:
            loss_bbox = torch.tensor(
                0,
                dtype=flatten_bbox_preds.dtype,
                device=flatten_bbox_preds.device)
        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox)
    
    def get_targets(
            self, 
            batch_gt_instances: InstanceList, 
            label_size_list: List[Tensor], 
            featmap_sizes: List[tuple],
            priors_list: List[Tensor]) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Compute regression and classification for priors in multiple images.

        Args:
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            featmap_sizes (list[tuple]): Size tuple of feature maps.
            priors_list (list[Tensor]): Priors list of each fpn level, each has
                shape (num_priors, 2).

        Returns:
            tuple: Targets of each level.

            - flatten_labels (list[Tensor]): Labels of each level.
            - flatten_bbox_targets (list[Tensor]): BBox targets of each
              level.
        """
        label_list, bbox_target_list, gt_ids_list = multi_apply(
            self._get_targets_single,
            batch_gt_instances,
            label_size_list,
            featmap_size_list=featmap_sizes,
            priors_list=priors_list)
        flatten_labels = [
            torch.cat([
                labels_level_img.flatten() for labels_level_img in labels_level
            ]) for labels_level in zip(*label_list)
        ]
        flatten_bbox_targets = [
            torch.cat([
                bbox_targets_level_img.reshape(-1, 4)
                for bbox_targets_level_img in bbox_targets_level
            ]) for bbox_targets_level in zip(*bbox_target_list)
        ]
        flatten_ids = [
            torch.cat([gt_ids_level_img.flatten()
                       for gt_ids_level_img in gt_ids_level])
            for gt_ids_level in zip(*gt_ids_list)
        ]

        flatten_labels = torch.cat(flatten_labels)
        flatten_bbox_targets = torch.cat(flatten_bbox_targets)
        flatten_ids = torch.cat(flatten_ids)
        return flatten_labels, flatten_bbox_targets, flatten_ids

    def _get_targets_single(self,
                            gt_instances: InstanceData,
                            label_size_list_raw: List[Tensor] = None,
                            featmap_size_list: List[tuple] = None,
                            priors_list: List[Tensor] = None) -> tuple:
        """Compute regression and classification targets for a single image.

        Args:
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.
            featmap_size_list (list[tuple]): Size tuple of feature maps.
            priors_list (list[Tensor]): Priors of each fpn level, each has
                shape (num_priors, 2).

        Returns:
            tuple:

            - label_list (list[Tensor]): Labels of all anchors in the image.
            - box_target_list (list[Tensor]): BBox targets of all anchors in
              the image.
        """
        gt_bboxes_raw = gt_instances.bboxes
        gt_labels_raw = gt_instances.labels
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) *
                              (gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))
        label_list = []
        bbox_target_list = []
        ids_list = []
        # for each pyramid, find the cls and box target
        for base_len, (lower_bound, upper_bound), stride, featmap_size, \
            priors in zip(self.base_edge_list, self.scale_ranges,
                          self.strides, featmap_size_list, priors_list):
            # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
            priors = priors.view(*featmap_size, 2)
            x, y = priors[..., 0], priors[..., 1]
            labels = gt_labels_raw.new_full(featmap_size, self.num_classes)
            bbox_targets = gt_bboxes_raw.new_ones(featmap_size[0],
                                                  featmap_size[1], 4)
            gt_ids = gt_labels_raw.new_zeros(featmap_size) - 1
            # scale assignment
            hit_indices = ((gt_areas >= lower_bound) &
                           (gt_areas <= upper_bound)).nonzero().flatten()
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
            # valid fovea area: left, right, top, down
            pos_left = torch.ceil(
                gt_bboxes[:, 0] + (1 - self.sigma) * half_w - 0.5).long(). \
                clamp(0, featmap_size[1] - 1)
            pos_right = torch.floor(
                gt_bboxes[:, 0] + (1 + self.sigma) * half_w - 0.5).long(). \
                clamp(0, featmap_size[1] - 1)
            pos_top = torch.ceil(
                gt_bboxes[:, 1] + (1 - self.sigma) * half_h - 0.5).long(). \
                clamp(0, featmap_size[0] - 1)
            pos_down = torch.floor(
                gt_bboxes[:, 1] + (1 + self.sigma) * half_h - 0.5).long(). \
                clamp(0, featmap_size[0] - 1)
            for px1, py1, px2, py2, label, gt_id, (gt_x1, gt_y1, gt_x2, gt_y2) in \
                    zip(pos_left, pos_top, pos_right, pos_down, gt_labels, hit_indices,
                        gt_bboxes_raw[hit_indices, :]):
                labels[py1:py2 + 1, px1:px2 + 1] = label
                gt_ids[py1:py2 + 1, px1:px2 + 1] = gt_id + label_size_list_raw
                bbox_targets[py1:py2 + 1, px1:px2 + 1, 0] = \
                    (x[py1:py2 + 1, px1:px2 + 1] - gt_x1) / base_len
                bbox_targets[py1:py2 + 1, px1:px2 + 1, 1] = \
                    (y[py1:py2 + 1, px1:px2 + 1] - gt_y1) / base_len
                bbox_targets[py1:py2 + 1, px1:px2 + 1, 2] = \
                    (gt_x2 - x[py1:py2 + 1, px1:px2 + 1]) / base_len
                bbox_targets[py1:py2 + 1, px1:px2 + 1, 3] = \
                    (gt_y2 - y[py1:py2 + 1, px1:px2 + 1]) / base_len
            bbox_targets = bbox_targets.clamp(min=1. / 16, max=16.)
            label_list.append(labels)
            bbox_target_list.append(torch.log(bbox_targets))
            ids_list.append(gt_ids)
        return label_list, bbox_target_list, ids_list
    
    # Same as base_dense_head/_predict_by_feat_single except self._bbox_decode
    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: Optional[ConfigDict] = None,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid, has shape
                (num_priors, 2).
            img_meta (dict): Image meta info.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        for level_idx, (cls_score, bbox_pred, stride, base_len, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list, self.strides,
                              self.base_edge_list, mlvl_priors)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, _, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            bboxes = self._bbox_decode(priors, bbox_pred, base_len, img_shape)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        results = InstanceData()
        results.bboxes = torch.cat(mlvl_bboxes)
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)
    
    def _bbox_post_process(self,
                           results: InstanceData,
                           cfg: ConfigDict,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: Optional[dict] = None) -> InstanceData:
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (ConfigDict): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        if hasattr(results, 'score_factors'):
            # TODO: Add sqrt operation in order to be consistent with
            #  the paper.
            score_factors = results.pop('score_factors')
            results.scores = results.scores * score_factors

        # 筛出同类别的样本
        high_score_condition = results.scores > cfg.score_thr
        for i in range(0, self.num_classes):
            cls_inds = torch.where(high_score_condition & (results.labels == i))[0]
            if cls_inds.numel() > 0:
                # 聚合同类别样本
                subset_bboxes = results.bboxes[cls_inds]
                subset_scores = results.scores[cls_inds]
                results.bboxes[cls_inds], results.scores[cls_inds] = agg_scores(subset_bboxes, subset_scores, cfg.k, cfg.agg_thr)

        # filter small size bboxes
        if cfg.get('min_bbox_size', -1) >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        # TODO: deal with `with_nms` and `nms_cfg=None` in test_cfg
        if with_nms and results.bboxes.numel() > 0:
            bboxes = get_box_tensor(results.bboxes)
            det_bboxes, keep_idxs = batched_nms(bboxes, results.scores,
                                                results.labels, cfg.nms)
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[:cfg.max_per_img]

        return results

    def _bbox_decode(self, priors: Tensor, bbox_pred: Tensor, base_len: int,
                     max_shape: int) -> Tensor:
        """Function to decode bbox.

        Args:
            priors (Tensor): Center proiors of an image, has shape
                (num_instances, 2).
            bbox_preds (Tensor): Box energies / deltas for all instances,
                has shape (batch_size, num_instances, 4).
            base_len (int): The base length.
            max_shape (int): The max shape of bbox.

        Returns:
            Tensor: Decoded bboxes in (tl_x, tl_y, br_x, br_y) format. Has
            shape (batch_size, num_instances, 4).
        """
        bbox_pred = bbox_pred.exp()

        y = priors[:, 1]
        x = priors[:, 0]
        x1 = (x - base_len * bbox_pred[:, 0]). \
            clamp(min=0, max=max_shape[1] - 1)
        y1 = (y - base_len * bbox_pred[:, 1]). \
            clamp(min=0, max=max_shape[0] - 1)
        x2 = (x + base_len * bbox_pred[:, 2]). \
            clamp(min=0, max=max_shape[1] - 1)
        y2 = (y + base_len * bbox_pred[:, 3]). \
            clamp(min=0, max=max_shape[0] - 1)
        decoded_bboxes = torch.stack([x1, y1, x2, y2], -1)
        return decoded_bboxes