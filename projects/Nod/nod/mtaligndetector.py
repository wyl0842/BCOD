# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.semi_base import SemiBaseDetector

import copy

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Union
from mmdet.structures import OptSampleList, SampleList
from mmdet.structures.bbox import bbox_project
from mmdet.models.utils import (filter_gt_instances, rename_loss_dict,
                                reweight_loss_dict)


@MODELS.register_module()
class MTAlignDetector(SemiBaseDetector):
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
    #     num_classes = self.teacher.bbox_head.num_classes
    #     num_scores = self.train_cfg.num_scores
    #     self.register_buffer(
    #         'scores', torch.zeros((num_classes, num_scores)))
    #     self.iter = 0

    # def set_iter(self, step):
    #     self.iter = step
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
    
    def loss(self, multi_batch_inputs: Dict[str, Tensor],
             multi_batch_data_samples: Dict[str, SampleList]) -> dict:
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

        if self.epoch < self.semi_train_cfg.get('warmup_step', -1):
            # 有监督样本计算loss
            losses.update(**self.loss_by_gt_instances(
            multi_batch_inputs['weak_student'], multi_batch_data_samples['weak_student']))
        else:
            batch_data_samples_copy = copy.deepcopy(multi_batch_data_samples['strong_teacher'])

            # 用teacher网络得到伪目标
            origin_pseudo_data_samples, batch_info = self.get_pseudo_instances(
                multi_batch_inputs['strong_teacher'],
                batch_data_samples_copy)
            # 得到用于训练student网络的伪样本
            batch_data_samples_unsup_student = self.project_pseudo_instances(
                    origin_pseudo_data_samples,
                    batch_data_samples_copy)
            batch_data_samples_unsup_student = filter_gt_instances(
                batch_data_samples_unsup_student, score_thr=self.semi_train_cfg.cls_pseudo_thr)
            for data_samples, data_samples_unsup_student in zip(multi_batch_data_samples['strong_teacher'], batch_data_samples_unsup_student):
                # data_samples.gt_instances = results.pred_instances
                data_samples.gt_instances.labels = torch.cat((data_samples.gt_instances.labels, data_samples_unsup_student.gt_instances.labels))
                data_samples.gt_instances.bboxes = torch.cat((data_samples.gt_instances.bboxes, data_samples_unsup_student.gt_instances.bboxes))
            losses.update(**self.loss_by_pseudo_instances(
                multi_batch_inputs['weak_student'],
                multi_batch_data_samples['strong_teacher'], batch_info))
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
        losses = self.student.loss(batch_inputs, batch_data_samples)
        pseudo_instances_num = sum([
            len(data_samples.gt_instances)
            for data_samples in batch_data_samples
        ])
        unsup_weight = self.semi_train_cfg.get(
            'unsup_weight', 1.) if pseudo_instances_num > 0 else 0.
        return rename_loss_dict('unsup_',
                                reweight_loss_dict(losses, unsup_weight))