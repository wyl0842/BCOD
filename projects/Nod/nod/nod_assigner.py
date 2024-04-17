# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmengine.structures import InstanceData

from mmdet.registry import TASK_UTILS
from mmdet.utils import ConfigType
from mmdet.models.task_modules.assigners import AssignResult, BaseAssigner

INF = 100000000


@TASK_UTILS.register_module()
class NODAssigner(BaseAssigner):
    """Task aligned assigner used in the paper:
    `TOOD: Task-aligned One-stage Object Detection.
    <https://arxiv.org/abs/2108.07755>`_.

    Assign a corresponding gt bbox or background to each predicted bbox.
    Each bbox will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (int): number of bbox selected in each level
        iou_calculator (:obj:`ConfigDict` or dict): Config dict for iou
            calculator. Defaults to ``dict(type='BboxOverlaps2D')``
    """

    def __init__(self,
                 topk: int,
                 iou_calculator: ConfigType = dict(type='BboxOverlaps2D')):
        assert topk >= 1
        self.topk = topk
        self.iou_calculator = TASK_UTILS.build(iou_calculator)

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               alpha: int = 1,
               beta: int = 6) -> AssignResult:
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute alignment metric between all bbox (bbox of all pyramid
           levels) and gt
        2. select top-k bbox as candidates for each gt
        3. limit the positive sample's center in gt (because the anchor-free
           detector only can predict positive distance)


        Args:
            pred_instances (:obj:`InstaceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors, points, or bboxes predicted by the model,
                shape(n, 4).
            gt_instances (:obj:`InstaceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.
            gt_instances_ignore (:obj:`InstaceData`, optional): Instances
                to be ignored during training. It includes ``bboxes``
                attribute data that is ignored during training and testing.
                Defaults to None.
            alpha (int): Hyper-parameters related to alignment_metrics.
                Defaults to 1.
            beta (int): Hyper-parameters related to alignment_metrics.
                Defaults to 6.

        Returns:
            :obj:`TaskAlignedAssignResult`: The assign result.
        """
        # anchors
        priors = pred_instances.priors
        # 预测框
        decode_bboxes = pred_instances.bboxes
        # 预测分数
        pred_scores = pred_instances.scores
        # 动量网络预测框
        decode_bboxes_momentum = pred_instances.bboxes_momentum
        # 动量网络预测分数
        pred_scores_momentum = pred_instances.scores_momentum
        # 真实框
        gt_bboxes = gt_instances.bboxes
        # 真实标签
        gt_labels = gt_instances.labels

        # torch.Size([17950, 4])
        # torch.Size([17950, 4])
        # torch.Size([17950, 10])
        # tensor([[720., 635., 738., 666.]], device='cuda:0')
        # tensor([5], device='cuda:0')
        # print("test")
        # print(priors.shape)
        # print(decode_bboxes.shape)
        # print(pred_scores.shape)
        # print(gt_bboxes)
        # print(gt_labels)

        priors = priors[:, :4]
        num_gt, num_bboxes = gt_bboxes.size(0), priors.size(0)
        # compute alignment metric between all bbox and gt
        # overlaps
        # torch.Size([11289, 2])
        # torch.Size([17950, 1])
        # torch.Size([11289, 2])
        # torch.Size([11289, 1])
        # torch.Size([17950, 1])
        # torch.Size([17950, 3])
        overlaps = self.iou_calculator(decode_bboxes, gt_bboxes).detach()
        bbox_scores = pred_scores[:, gt_labels].detach()
        # assign 0 by default
        assigned_gt_inds = priors.new_full((num_bboxes, ), 0, dtype=torch.long)
        assign_metrics = priors.new_zeros((num_bboxes, ))

        # 没有gt框或者预测框
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = priors.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No gt boxes, assign everything to background
                assigned_gt_inds[:] = 0
            assigned_labels = priors.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
            assign_result = AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
            assign_result.assign_metrics = assign_metrics
            return assign_result

        # select top-k bboxes as candidates for each gt
        # 论文里的t
        # alignment_metrics
        # torch.Size([17950, 1])
        # torch.Size([17950, 1])
        # torch.Size([11289, 2])
        # torch.Size([11289, 5])
        alignment_metrics = bbox_scores**alpha * overlaps**beta
        topk = min(self.topk, alignment_metrics.size(0))
        # 从17950个点中找到前topk个
        # values=tensor([
        # [2.6525e-03, 1.1220e-04],
        # [2.1921e-03, 9.2715e-05],
        # ...], device='cuda:0'),
        # indices=tensor([
        # [10766, 16160],
        # [10882, 16161],
        # ...], device='cuda:0')
        _, candidate_idxs = alignment_metrics.topk(topk, dim=0, largest=True)
        # 把topk个预测框揪出来
        # tensor([[4.6773e-03, 8.0645e-04],
        # [2.4557e-03, 7.3716e-04],
        # [2.2720e-03, 6.5592e-04],
        # [2.2075e-03, 4.5721e-04],
        # [2.0888e-03, 4.4806e-04],
        # [1.6399e-03, 4.1867e-04],
        # [1.5601e-03, 1.8411e-04],
        # [1.5251e-03, 1.3055e-04],
        # [1.2078e-03, 4.2603e-05],
        # [1.1679e-03, 3.1375e-05],
        # [1.1220e-03, 1.5857e-05],
        # [1.0860e-03, 1.1223e-05],
        # [9.1087e-04, 1.0124e-05]], device='cuda:0')
        candidate_metrics = alignment_metrics[candidate_idxs,
                                              torch.arange(num_gt)]
        is_pos = candidate_metrics > 0

        # limit the positive sample's center in gt
        priors_cx = (priors[:, 0] + priors[:, 2]) / 2.0
        priors_cy = (priors[:, 1] + priors[:, 3]) / 2.0
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        ep_priors_cx = priors_cx.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        ep_priors_cy = priors_cy.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        candidate_idxs = candidate_idxs.view(-1)

        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side
        l_ = ep_priors_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_priors_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_priors_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_priors_cy[candidate_idxs].view(-1, num_gt)
        is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
        # 需要同时满足t为topk个且在gtbox内
        is_pos = is_pos & is_in_gts

        # if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected.
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1
        assign_metrics[max_overlaps != -INF] = alignment_metrics[
            max_overlaps != -INF, argmax_overlaps[max_overlaps != -INF]]

        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        pos_inds = torch.nonzero(
            assigned_gt_inds > 0, as_tuple=False).squeeze()
        if pos_inds.numel() > 0:
            assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] -
                                                  1]
        assign_result = AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        assign_result.assign_metrics = assign_metrics
        return assign_result
