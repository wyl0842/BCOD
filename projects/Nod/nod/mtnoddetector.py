# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.semi_base import SemiBaseDetector

import copy

import torch
import torch.nn as nn
import pickle
from torch import Tensor
from mmengine.structures import InstanceData
from typing import Dict, List, Optional, Tuple, Union
from mmdet.structures import OptSampleList, SampleList
from mmdet.structures.bbox import bbox_project
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.models.utils import (filter_gt_instances, rename_loss_dict,
                                reweight_loss_dict, multi_apply)


@MODELS.register_module()
class MTNODDetector(SemiBaseDetector):
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
        self.tau_c_max = self.semi_train_cfg.get('tau_c_max', 1)
        self.tau_clean = self.semi_train_cfg.get('tau_clean', 0.5)
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

        if self.epoch < self.semi_train_cfg.get('warmup_step', -1):
            # 有监督样本计算loss
            losses.update(**self.loss_by_gt_instances(
            multi_batch_inputs['strong_student'], multi_batch_data_samples['strong_student']))
        else:
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
            # batch_data_samples_unsup_student = filter_gt_instances(
            #     batch_data_samples_unsup_student, score_thr=self.semi_train_cfg.cls_pseudo_thr)
            # assign_result = multi_apply(self.assign_pseudo_gt, batch_data_samples_unsup_student, multi_batch_data_samples['strong_teacher'])
            # 计算当前epoch的标签清洗阈值
            threshold_clean = (self.tau_c_max - self.tau_clean) * (self.epoch - self.warmup_epochs) / (self.num_epochs - self.warmup_epochs) + self.tau_clean
            gt_data_samples = []
            for i in range(len(batch_data_samples_unsup_student)):
                gt_data_sample = self.assign_pseudo_gt(batch_data_samples_unsup_student[i], multi_batch_data_samples['weak_teacher'][i])
                # 可信度小于阈值，加入到忽略样本中
                gt_data_sample.gt_instances_ignore = gt_data_sample.gt_instances[
                gt_data_sample.gt_instances.scores <= threshold_clean]
                gt_data_samples.append(gt_data_sample)

            # 如果某个gt_instance的可信度小于阈值，就筛掉
            gt_data_samples = filter_gt_instances(
                    gt_data_samples, score_thr=threshold_clean)
            # losses.update(**self.loss_by_pseudo_instances(
            #     multi_batch_inputs['weak_student'],
            #     multi_batch_data_samples['strong_teacher'], batch_info))
            losses.update(**self.loss_by_pseudo_instances(
                multi_batch_inputs['strong_student'],
                gt_data_samples, batch_info))
        return losses
    
    @torch.no_grad()
    def get_pseudo_instances(
            self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Tuple[SampleList, Optional[dict]]:
        """Get pseudo instances from teacher model."""
        self.teacher.eval()
        feat = self.teacher.extract_feat(batch_inputs)
        results_list = self.teacher.bbox_head.predict_prob(feat, batch_data_samples, rescale=False)
        batch_info = {}
        for data_samples, results in zip(batch_data_samples, results_list):
            data_samples.gt_instances = results
            data_samples.gt_instances.bboxes = bbox_project(
                data_samples.gt_instances.bboxes,
                torch.from_numpy(data_samples.homography_matrix).inverse().to(
                    self.data_preprocessor.device), data_samples.ori_shape)
        return batch_data_samples, batch_info
    
    def assign_pseudo_gt(self, pred: SampleList,
                            gt: SampleList):
        # a = InstanceData()
        # a.bboxes = torch.FloatTensor([[0, 0, 1, 1], [8, 8, 9, 9]])
        # a.labels = torch.LongTensor([1, 3])
        # b = InstanceData()
        # b.scores = torch.rand((10, 5))
        # b.bboxes = torch.rand((10, 4))*10
        # print(b)
        # # b.scores = torch.rand((1, 81))
        # # b.bboxes = torch.rand((1, 4))
        # img_meta = dict(img_shape=(10, 8))
        # assign_result = self.lc_assigner.assign(
        #     b, a, img_meta=img_meta)
        # print(assign_result)
        # print(assign_result.gt_inds)
        # # 以上仅为匈牙利算法测试
        
        # 获取匈牙利算法匹配结果
        temp_pred_instances = InstanceData()
        temp_pred_instances.scores = copy.deepcopy(pred.gt_instances.prob_scores)
        temp_pred_instances.bboxes = copy.deepcopy(pred.gt_instances.bboxes)
        print('**********************')
        print('temp_pred_instances',temp_pred_instances)
        print(temp_pred_instances.scores.shape)
        print('gt.gt_instances:',gt.gt_instances)
        assign_result = self.lc_assigner.assign(pred_instances=temp_pred_instances,
            gt_instances=gt.gt_instances,
            img_meta=gt.metainfo)

        # # 计算当前epoch的标签清洗阈值
        # threshold_clean = (self.tau_c_max - self.tau_clean) * (self.epoch - self.warmup_epochs) / (self.num_epochs - self.warmup_epochs) + self.tau_clean

        # 从匹配结果中找出匹配到的元素位置
        nonzero_elements = assign_result.gt_inds.nonzero().squeeze(dim=1)
        # 根据位置挑出对应的预测概率分布
        select_preds = pred.gt_instances.prob_scores[nonzero_elements]
        # 从匹配结果中获取预测和真实框的对应关系
        sorted_indices = assign_result.gt_inds[nonzero_elements] - 1
        # 将预测框和真实框按顺序匹配起来
        sorted_preds = torch.index_select(select_preds, dim=0, index=sorted_indices)
        # 得到真实标注的onehot标签
        gt_labels_one_hot = torch.nn.functional.one_hot(gt.gt_instances.labels, num_classes=self.num_classes)
        # 进行标签平滑，防止计算js散度时分母为0
        gt_labels_one_hot_smooth = gt_labels_one_hot * (
                1 - self.one_hot_smoother
            ) + self.one_hot_smoother / self.num_classes
        # 计算js散度得到预测和真实标签之间的差异，作为标签可信度指标
        sorted_preds = self.normalize_distribution_tensor(sorted_preds)
        print('sorted_preds:',sorted_preds)
        print('gt_labels_one_hot_smooth:',gt_labels_one_hot_smooth)
        prob_clean = 1 - self.js_div(sorted_preds, gt_labels_one_hot_smooth)
        print('prob_clean:',prob_clean)
        return_gt = copy.deepcopy(gt)
        return_gt.gt_instances.scores = prob_clean

        # # 如果某个gt_instance的可信度小于阈值，就筛掉
        # print(return_gt)
        # return_gt = filter_gt_instances(
        #         return_gt, score_thr=threshold_clean)
        
        return return_gt
    
    def normalize_distribution_tensor(self, distribution):
        distribution_sums = distribution.sum(dim=1, keepdim=True)
        norm_distribution = distribution / distribution_sums
        return norm_distribution
    
    def js_div(self, p, q):
        # Jensen-Shannon divergence, value is in (0, 1)
        m = 0.5 * (p + q)
        return 0.5 * self.kl_div(p, m) + 0.5 * self.kl_div(q, m)
    
    def kl_div(self, p, q):
        # p, q is in shape (batch_size, n_classes)
        return (p * p.log2() - p * q.log2()).sum(dim=1)
    
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