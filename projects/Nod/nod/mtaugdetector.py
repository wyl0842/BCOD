# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.semi_base import SemiBaseDetector

import copy

import torch
import torch.nn as nn
from torch import Tensor
from mmengine.structures import InstanceData
from typing import Dict, List, Optional, Tuple, Union
from mmdet.structures import OptSampleList, SampleList
from mmdet.structures.bbox import bbox_project
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.models.utils import (filter_gt_instances, rename_loss_dict,
                                reweight_loss_dict, multi_apply)


@MODELS.register_module()
class MTAUGDetector(SemiBaseDetector):
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
        self.lc_assigner = TASK_UTILS.build(
                self.semi_train_cfg['lc_assigner'])
        self.tau_c_max = 0.95
        self.tau_clean = 0.75
        self.warmup_epochs = self.semi_train_cfg.get('warmup_step', -1)
        self.one_hot_smoother = self.semi_train_cfg.get('one_hot_smoother', 0)
        self.num_classes = self.semi_train_cfg.get('num_classes', 0)
        self.num_epochs = self.semi_train_cfg.get('num_epochs', 0)

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

        # 有监督样本计算loss
        losses.update(**self.loss_by_gt_instances(
        multi_batch_inputs['strong_student'], multi_batch_data_samples['strong_student']))

        # 为防止改变原先的数据，进行备份
        teacher_batch_data_samples_copy = copy.deepcopy(multi_batch_data_samples['weak_teacher'])
        student_batch_data_samples_copy = copy.deepcopy(multi_batch_data_samples['strong_student'])
        # 用teacher网络得到伪目标，此时origin_pseudo_data_samples和teacher_batch_data_samples_copy一致，gt_instances都有改变
        origin_pseudo_data_samples, batch_info = self.get_pseudo_instances(
            multi_batch_inputs['weak_teacher'],
            teacher_batch_data_samples_copy)
        # print(origin_pseudo_data_samples)
        # 得到用于训练student网络的伪样本，返回的为过滤后的用于训练的标签样本
        batch_data_samples_unsup_student = self.project_pseudo_instances(
                origin_pseudo_data_samples,
                student_batch_data_samples_copy)
        losses.update(**self.loss_by_pseudo_instances(
            multi_batch_inputs['strong_student'],
            batch_data_samples_unsup_student, batch_info))
        return losses
