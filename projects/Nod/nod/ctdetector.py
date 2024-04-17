# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.semi_base import SemiBaseDetector
from .utils import filter_invalid, filter_invalid_tuple

import copy

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional, Tuple, Union
from mmdet.structures import OptSampleList, SampleList
from mmdet.structures.bbox import bbox_project
from mmdet.models.utils import filter_gt_instances, rename_loss_dict, reweight_loss_dict

import numpy as np
try:
    import sklearn.mixture as skm
except ImportError:
    skm = None

@MODELS.register_module()
class CTDetector(SemiBaseDetector):
    r"""Implementation of `TOOD: Task-aligned One-stage Object Detection.
    <https://arxiv.org/abs/2108.07755>`_

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone module.
        neck (:obj:`ConfigDict` or dict): The neck module.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head module.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of TOOD. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of TOOD. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 detector: ConfigType,
                 semi_train_cfg: OptConfigType = None,
                 semi_test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            detector=detector,
            semi_train_cfg=semi_train_cfg,
            semi_test_cfg=semi_test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        if self.semi_train_cfg.get('freeze_teacher', True) is True:
            self.unsup_weight = self.semi_train_cfg.unsup_weight
        num_classes = self.teacher.bbox_head.num_classes
        num_scores = self.semi_train_cfg.num_scores
        self.register_buffer(
            'scores', torch.zeros((num_classes, num_scores)))
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
    
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        losses = dict()
        # 有监督样本计算loss
        losses.update(**self.loss_by_gt_instances(
            batch_inputs, batch_data_samples))
        # print('******************1')
        # print(batch_data_samples)

        # 用teacher网络得到伪目标
        origin_pseudo_data_samples, batch_info = self.get_pseudo_instances(
            batch_inputs,
            batch_data_samples)
        # print('******************2')
        # print(origin_pseudo_data_samples)
        # print(batch_data_samples)
        # 得到用于训练student网络的伪样本
        batch_data_samples_unsup_student = self.project_pseudo_instances(
                origin_pseudo_data_samples,
                batch_data_samples)
        losses.update(**self.loss_by_pseudo_instances(
            batch_inputs,
            batch_data_samples_unsup_student, batch_info))
        # print('******************3')
        # print(batch_data_samples_unsup_student)
        return losses
    
    def loss_by_pseudo_instances(self,
                                 batch_inputs: Tensor,
                                 batch_data_samples: SampleList,
                                 batch_info: Optional[dict] = None) -> dict:
        """Calculate losses from a batch of inputs and pseudo data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`,
                which are `pseudo_instance` or `pseudo_panoptic_seg`
                or `pseudo_sem_seg` in fact.
            batch_info (dict): Batch information of teacher model
                forward propagation process. Defaults to None.

        Returns:
            dict: A dictionary of loss components
        """

        # 第一次阈值筛选，按比例筛选
        thrs = []
        for proposals in batch_data_samples:
            dynamic_ratio = self.semi_train_cfg.dynamic_ratio
            scores = proposals.gt_instances.scores.clone()
            scores = scores.sort(descending=True)[0]
            if len(scores) == 0:
                thrs.append(1)  # no kept pseudo boxes
            else:
                # num_gt = int(scores.sum() + 0.5)
                num_gt = int(scores.sum() * dynamic_ratio + 0.5)
                num_gt = min(num_gt, len(scores) - 1)
                thrs.append(scores[num_gt] - 1e-5)

        batch_data_samples = filter_invalid(
            batch_data_samples, score_thr=thrs)
        # batch_data_samples = filter_gt_instances(
        #     batch_data_samples, score_thr=self.semi_train_cfg.cls_pseudo_thr)
        
        # 第二次阈值筛选，用GMM阈值筛选
        scores = torch.cat([proposal.gt_instances.scores for proposal in batch_data_samples])
        labels = torch.cat([proposal.gt_instances.labels for proposal in batch_data_samples])
        thrs = torch.zeros_like(scores)
        for label in torch.unique(labels):
            label = int(label)
            # 把所有相同标签的分数放到一块
            scores_add = (scores[labels == label])
            num_buffers = len(self.scores[label])
            # scores_new = torch.cat([scores_add, self.scores[label]])[:num_buffers]
            scores_new= torch.cat([scores_add.float(), self.scores[label].float()])[:num_buffers]
            self.scores[label] = scores_new
            thr = self.gmm_policy(
                scores_new[scores_new > 0],
                given_gt_thr=self.semi_train_cfg.get('given_gt_thr', 0),
                policy=self.semi_train_cfg.get('policy', 'high'))
            thrs[labels == label] = thr
        mean_thr = thrs.mean()
        if len(thrs) == 0:
            mean_thr.fill_(0)
        mean_thr = float(mean_thr)
        # log_every_n({"gmm_thr": mean_thr})
        # batch_info["gmm_thr"] = mean_thr
        thrs = torch.split(thrs, [len(p.gt_instances.scores) for p in batch_data_samples])
        batch_data_samples = filter_invalid_tuple(
            batch_data_samples, score_thr=thrs)

        # batch_data_samples = filter_gt_instances(
        #     batch_data_samples, score_thr=self.semi_train_cfg.cls_pseudo_thr)
        losses = self.student.loss(batch_inputs, batch_data_samples)
        losses['gmm_thr'] = torch.tensor(mean_thr).to(batch_data_samples[0].gt_instances.labels.device)
        # pseudo_instances_num = sum([
        #     len(data_samples.gt_instances)
        #     for data_samples in batch_data_samples
        # ])
        # unsup_weight = self.semi_train_cfg.get(
        #     'unsup_weight', 1.) if pseudo_instances_num > 0 else 0.
        unsup_weight = self.unsup_weight
        if self.epoch < self.semi_train_cfg.get('warmup_step', -1):
            unsup_weight = 0
        return rename_loss_dict('unsup_',
                                reweight_loss_dict(losses, unsup_weight))
    
    @torch.no_grad()
    def get_pseudo_instances(
            self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Tuple[SampleList, Optional[dict]]:
        """Get pseudo instances from teacher model."""
        self.teacher.eval()
        results_list = self.teacher.predict(
            batch_inputs, batch_data_samples, rescale=False)
        batch_info = {}
        for data_samples, results in zip(batch_data_samples, results_list):
            data_samples.gt_instances = results.pred_instances
            data_samples.gt_instances.bboxes = bbox_project(
                data_samples.gt_instances.bboxes,
                torch.from_numpy(data_samples.homography_matrix).inverse().to(
                    self.data_preprocessor.device), data_samples.ori_shape)
        return batch_data_samples, batch_info

    def project_pseudo_instances(self, batch_pseudo_instances: SampleList,
                                 batch_data_samples: SampleList) -> SampleList:
        """Project pseudo instances."""
        for pseudo_instances, data_samples in zip(batch_pseudo_instances,
                                                  batch_data_samples):
            data_samples.gt_instances = copy.deepcopy(
                pseudo_instances.gt_instances)
            data_samples.gt_instances.bboxes = bbox_project(
                data_samples.gt_instances.bboxes,
                torch.tensor(data_samples.homography_matrix).to(
                    self.data_preprocessor.device), data_samples.img_shape)
        wh_thr = self.semi_train_cfg.get('min_pseudo_bbox_wh', (1e-2, 1e-2))
        return filter_gt_instances(batch_data_samples, wh_thr=wh_thr)
    
    def gmm_policy(self, scores, given_gt_thr=0.5, policy='high'):
        """The policy of choosing pseudo label.

        The previous GMM-B policy is used as default.
        1. Use the predicted bbox to fit a GMM with 2 center.
        2. Find the predicted bbox belonging to the positive
            cluster with highest GMM probability.
        3. Take the class score of the finded bbox as gt_thr.

        Args:
            scores (nd.array): The scores.

        Returns:
            float: Found gt_thr.

        """
        if len(scores) < 4:
            return given_gt_thr
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if len(scores.shape) == 1:
            scores = scores[:, np.newaxis]
        means_init = [[np.min(scores)], [np.max(scores)]]
        weights_init = [1 / 2, 1 / 2]
        precisions_init = [[[1.0]], [[1.0]]]
        gmm = skm.GaussianMixture(
            2,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init)
        gmm.fit(scores)
        gmm_assignment = gmm.predict(scores)
        gmm_scores = gmm.score_samples(scores)
        assert policy in ['middle', 'high']
        if policy == 'high':
            if (gmm_assignment == 1).any():
                gmm_scores[gmm_assignment == 0] = -np.inf
                indx = np.argmax(gmm_scores, axis=0)
                pos_indx = (gmm_assignment == 1) & (
                    scores >= scores[indx]).squeeze()
                pos_thr = float(scores[pos_indx].min())
                # pos_thr = max(given_gt_thr, pos_thr)
            else:
                pos_thr = given_gt_thr
        elif policy == 'middle':
            if (gmm_assignment == 1).any():
                pos_thr = float(scores[gmm_assignment == 1].min())
                # pos_thr = max(given_gt_thr, pos_thr)
            else:
                pos_thr = given_gt_thr

        return pos_thr