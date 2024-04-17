from typing import Optional

import torch
from mmengine.structures import InstanceData
import torch.nn.functional as F
from mmdet.models.task_modules.assigners import AssignResult, BaseAssigner
from mmdet.registry import TASK_UTILS
from mmdet.utils import ConfigType
from mmdet.structures.bbox import BaseBoxes

from torch import Tensor

INF = 100000000
EPS = 1.0e-7

@TASK_UTILS.register_module()
class DynamicAssigner(BaseAssigner):
    """Computes matching between predictions and ground truth with dynamic soft
    label assignment.

    Args:
        topk (int): Select top-k predictions to calculate dynamic k
            best matchs for each gt. Default 13.
        iou_calculator (dict): Config dict for iou calculator.
            Default: dict(type='BboxOverlaps2D')
        iou_factor (float): The scale factor of iou cost. Default 3.0.
    """
    def __init__(self,
                 topk: int = 13,
                 iou_calculator: ConfigType = dict(type='BboxOverlaps2D'),
                 iou_factor: float = 3.0):
        self.topk = topk
        self.iou_calculator = TASK_UTILS.build(iou_calculator)
        self.iou_factor = iou_factor

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               alpha: int = 1,
               beta: int = 6) -> AssignResult:
        """Assign gt to priors with dynamic soft label assignment.
        Args:
            pred_scores (Tensor): Classification scores of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Predicted bboxes, a 2D-Tensor with shape
                [num_priors, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        num_gt = gt_bboxes.size(0)

        decoded_bboxes = pred_instances.bboxes
        pred_scores = pred_instances.scores
        priors = pred_instances.priors
        num_bboxes = decoded_bboxes.size(0)

        priors = priors[:, :4]

        # assign 0 by default
        # 存储每个预测框分配的真实框的索引，初始化为0
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes,),
                                                   0,
                                                   dtype=torch.long)
        
        # 提取prior的中心坐标
        prior_center = priors[:, :2]
        # 检查真实框是否在prior内
        if isinstance(gt_bboxes, BaseBoxes):
            is_in_gts = gt_bboxes.find_inside_points(prior_center)
        else:
            # Tensor boxes will be treated as horizontal boxes by defaults
            lt_ = prior_center[:, None] - gt_bboxes[:, :2]
            rb_ = gt_bboxes[:, 2:] - prior_center[:, None]

            deltas = torch.cat([lt_, rb_], dim=-1)
            is_in_gts = deltas.min(dim=-1).values > 0

        # 至少在一个真实框内
        valid_mask = is_in_gts.sum(dim=1) > 0

        # 过滤出有效的decoded bounding boxes及其对应的分数
        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        num_valid = valid_decoded_bbox.size(0)

        # 如果没有真实框、预测框或有效框，则返回空的分配结果
        if num_gt == 0 or num_bboxes == 0 or num_valid == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            gt_cost = torch.zeros_like(gt_labels).float()
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = decoded_bboxes.new_full((num_bboxes,),
                                                          -1,
                                                          dtype=torch.long)
            assign_result = AssignResult(num_gt,
                                         assigned_gt_inds,
                                         max_overlaps,
                                         labels=assigned_labels)
            assign_result.assign_metrics = max_overlaps
            assign_result.gt_cost = gt_cost
            return assign_result

        # 计算基于距离、交并比（IoU）和分类的匹配成本
        gt_center = (gt_bboxes[:, :2] + gt_bboxes[:, 2:4]) / 2.0
        # 计算距离成本，使用soft center prior
        valid_prior = priors[valid_mask]
        strides = valid_prior[:, 2]
        distance = (valid_prior[:, None, :2] - gt_center[None, :, :]
                    ).pow(2).sum(-1).sqrt() / strides[:, None]
        dis_cost = 0.001 * torch.pow(10, distance)

        # 计算预测框与真实框的IoU成本
        pairwise_ious = self.iou_calculator(valid_decoded_bbox,
                                            gt_bboxes[:, :4]).detach()
        iou_cost = -torch.log(pairwise_ious + 1e-7)

        # 使用focal loss计算分类成本
        # 创建ground truth的one-hot标签
        gt_onehot_label = (F.one_hot(gt_labels.to(
            torch.int64), pred_scores.shape[-1]).float().unsqueeze(0).repeat(
            num_valid, 1, 1))
        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)

        target = gt_onehot_label
        pred = valid_pred_scores
        alpha = 0.25
        gamma = 2
        pt = (1 - pred) * target + pred * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) *
                        (1 - target)) * pt.pow(gamma)
        cls_cost = F.binary_cross_entropy(
            pred, target, reduction='none') * focal_weight

        cls_cost = cls_cost.sum(dim=-1)

        # 将所有成本组合成一个成本矩阵
        cost_matrix = cls_cost + iou_cost * self.iou_factor + dis_cost

        # 使用动态k匹配算法找到匹配并计算前景预测的成本
        matched_pred_ious, matched_gt_inds, fg_mask_inboxes = self.dynamic_k_matching(
            cost_matrix, pairwise_ious, num_gt, valid_mask)
        cost_fg = cost_matrix[fg_mask_inboxes]
        cost_fg = cost_fg[
            torch.arange(cost_fg.shape[0]).to(
                cls_cost.device), matched_gt_inds]
        # 基于匹配的真实框索引计算真实框的成本
        gt_cost = torch.zeros_like(gt_labels).float()
        for idx in range(num_gt):
            gt_cost[idx] = cost_fg[matched_gt_inds == idx].mean()
        # convert to AssignResult format
        # 更新分配的真实框索引、标签和最大交并比在AssignResult中
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full((num_bboxes,),
                                                 -INF,
                                                 dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious
        assign_result = AssignResult(num_gt,
                                     assigned_gt_inds,
                                     max_overlaps,
                                     labels=assigned_labels)
        if hasattr(assign_result, 'assign_metrics'):
            del assign_result.assign_metrics
        assign_result.assign_metrics = max_overlaps

        if hasattr(assign_result, 'gt_cost'):
            del assign_result.gt_cost
        assign_result.gt_cost = gt_cost

        return assign_result

    def dynamic_k_matching(self, cost, pairwise_ious, num_gt, valid_mask):
        """Use sum of topk pred iou as dynamic k. Refer from OTA and YOLOX.动态K匹配算法，使用topk预测IoU之和来确定K值。参考了OTA和YOLOX。

        Args:
            cost (Tensor): Cost matrix.
            pairwise_ious (Tensor): Pairwise iou matrix.
            num_gt (int): Number of gt.
            valid_mask (Tensor): Mask for valid bboxes.
        """
        # 初始化匹配矩阵
        matching_matrix = torch.zeros_like(cost)
        # select candidate topk ious for dynamic-k calculation
        # 选择用于动态K计算的候选topk IoU
        candidate_topk = min(self.topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        # calculate dynamic k for each gt
        # 计算每个Ground Truth的动态K值
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            # 对每个Ground Truth，在cost中选择动态K个最小的元素的索引
            _, pos_idx = torch.topk(cost[:, gt_idx],
                                    k=dynamic_ks[gt_idx].item(),
                                    largest=False)
            # 将匹配矩阵中对应位置设置为1
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        # 释放中间变量的内存
        del topk_ious, dynamic_ks, pos_idx

        # 处理那些多于一个匹配的先验框
        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            # 对于每个多于一个匹配的先验框，选择成本最小的那个作为匹配目标
            cost_min, cost_argmin = torch.min(cost[prior_match_gt_mask, :],
                                              dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0.0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1.0
        # get foreground mask inside box and center prior
            # 获取在框内和中心先验内的前景掩码
        fg_mask_inboxes = matching_matrix.sum(1) > 0.0
        # 更新有效框的掩码，将其与前景掩码对齐
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        # 获取匹配的Ground Truth的索引和匹配的预测IoU
        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix *
                             pairwise_ious).sum(1)[fg_mask_inboxes]
        return matched_pred_ious, matched_gt_inds, fg_mask_inboxes