from mmdet.structures import SampleList
from mmdet.structures.mask import BitmapMasks
from typing import Tuple, List, Optional, Dict
from torch import Tensor
import torch

def filter_invalid(batch_data_samples: SampleList,
                        score_thr: List[Tensor] = None,
                        wh_thr: tuple = None):
    if score_thr is not None:
        batch_data_samples = _filter_gt_instances_by_score(
            batch_data_samples, score_thr)
    if wh_thr is not None:
        batch_data_samples = _filter_gt_instances_by_size(
            batch_data_samples, wh_thr)
    return batch_data_samples

def filter_invalid_tuple(batch_data_samples: SampleList,
                        score_thr: Tuple[List[Tensor]] = None,
                        wh_thr: tuple = None):
    if score_thr is not None:
        batch_data_samples = _filter_gt_instances_by_score_tuple(
            batch_data_samples, score_thr)
    if wh_thr is not None:
        batch_data_samples = _filter_gt_instances_by_size(
            batch_data_samples, wh_thr)
    return batch_data_samples
        
def _filter_gt_instances_by_score(batch_data_samples: SampleList,
                                  score_thr: List[Tensor]):
    for i, data_samples in enumerate(batch_data_samples):
        assert 'scores' in data_samples.gt_instances, \
            'there does not exit scores in instances'
        if data_samples.gt_instances.bboxes.shape[0] > 0:
            data_samples.gt_instances = data_samples.gt_instances[
                data_samples.gt_instances.scores > score_thr[i]]
    return batch_data_samples

def _filter_gt_instances_by_score_tuple(batch_data_samples: SampleList,
                                  score_thr: Tuple[List[Tensor]]):
    for i, data_samples in enumerate(batch_data_samples):
        assert 'scores' in data_samples.gt_instances, \
            'there does not exit scores in instances'
        if data_samples.gt_instances.bboxes.shape[0] > 0:
            data_samples.gt_instances = data_samples.gt_instances[
                data_samples.gt_instances.scores > score_thr[i]]
    return batch_data_samples

def _filter_gt_instances_by_size(batch_data_samples: SampleList,
                                 wh_thr: tuple) -> SampleList:
    """Filter ground truth (GT) instances by size.

    Args:
        batch_data_samples (SampleList): The Data
            Samples. It usually includes information such as
            `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
        wh_thr (tuple):  Minimum width and height of bbox.

    Returns:
        SampleList: The Data Samples filtered by score.
    """
    for data_samples in batch_data_samples:
        bboxes = data_samples.gt_instances.bboxes
        if bboxes.shape[0] > 0:
            w = bboxes[:, 2] - bboxes[:, 0]
            h = bboxes[:, 3] - bboxes[:, 1]
            data_samples.gt_instances = data_samples.gt_instances[
                (w > wh_thr[0]) & (h > wh_thr[1])]
    return batch_data_samples

def my_filter_scores_and_topk(scores, score_thr, topk, results=None):
    """Filter results using score threshold and topk candidates.

    Args:
        scores (Tensor): The scores, shape (num_bboxes, K).
        score_thr (float): The score filter threshold.
        topk (int): The number of topk candidates.
        results (dict or list or Tensor, Optional): The results to
           which the filtering rule is to be applied. The shape
           of each item is (num_bboxes, N).

    Returns:
        tuple: Filtered results

            - scores (Tensor): The scores after being filtered, \
                shape (num_bboxes_filtered, ).
            - labels (Tensor): The class labels, shape \
                (num_bboxes_filtered, ).
            - anchor_idxs (Tensor): The anchor indexes, shape \
                (num_bboxes_filtered, ).
            - filtered_results (dict or list or Tensor, Optional): \
                The filtered results. The shape of each item is \
                (num_bboxes_filtered, N).
    """
    # score_thr = 0.0
    my_max = scores.max(dim=-1, keepdim=False)
    my_valid_idxs = my_max[1].unsqueeze(1)
    my_max = my_max[0]
    my_valid_mask = my_max > score_thr
    my_max = my_max[my_valid_mask]

    my_scores = scores[my_valid_mask]

    my_valid_idxs = torch.cat((torch.nonzero(my_valid_mask), my_valid_idxs[my_valid_mask]), 1)

    my_num_topk = min(topk, my_valid_idxs.size(0))
    # torch.sort is actually faster than .topk (at least on GPUs)

    _, my_idxs = my_max.sort(descending=True)

    my_scores = my_scores[my_idxs][:my_num_topk]

    my_topk_idxs = my_valid_idxs[my_idxs[:my_num_topk]]
    my_keep_idxs, my_labels = my_topk_idxs.unbind(dim=1)

    my_filtered_results = None
    if results is not None:
        if isinstance(results, dict):
            my_filtered_results = {k: v[my_keep_idxs] for k, v in results.items()}
        elif isinstance(results, list):
            my_filtered_results = [result[my_keep_idxs] for result in results]
        elif isinstance(results, torch.Tensor):
            my_filtered_results = results[my_keep_idxs]
        else:
            raise NotImplementedError(f'Only supports dict or list or Tensor, '
                                      f'but get {type(results)}.')
    return my_scores, my_labels, my_keep_idxs, my_filtered_results

def my_batched_nms(boxes: Tensor,
                scores: Tensor,
                idxs: Tensor,
                nms_cfg: Optional[Dict],
                class_agnostic: bool = False) -> Tuple[Tensor, Tensor]:
    r"""Performs non-maximum suppression in a batched fashion.

    Modified from `torchvision/ops/boxes.py#L39
    <https://github.com/pytorch/vision/blob/
    505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39>`_.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.

    Note:
        In v1.4.1 and later, ``batched_nms`` supports skipping the NMS and
        returns sorted raw results when `nms_cfg` is None.

    Args:
        boxes (torch.Tensor): boxes in shape (N, 4) or (N, 5).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict | optional): Supports skipping the nms when `nms_cfg`
            is None, otherwise it should specify nms type and other
            parameters like `iou_thr`. Possible keys includes the following.

            - iou_threshold (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
              number of boxes is large (e.g., 200k). To avoid OOM during
              training, the users could set `split_thr` to a small value.
              If the number of boxes is greater than the threshold, it will
              perform NMS on each group of boxes separately and sequentially.
              Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class. Defaults to False.

    Returns:
        tuple: kept dets and indice.

        - boxes (Tensor): Bboxes with score after nms, has shape
          (num_bboxes, 5). last dimension 5 arrange as
          (x1, y1, x2, y2, score)
        - keep (Tensor): The indices of remaining boxes in input
          boxes.
    """
    my_scores = scores
    scores = scores.max(dim=-1, keepdim=False)[0]
    # skip nms when nms_cfg is None
    if nms_cfg is None:
        scores, inds = scores.sort(descending=True)
        boxes = boxes[inds]
        return torch.cat([boxes, scores[:, None]], -1), inds

    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        # When using rotated boxes, only apply offsets on center.
        if boxes.size(-1) == 5:
            # Strictly, the maximum coordinates of the rotating box
            # (x,y,w,h,a) should be calculated by polygon coordinates.
            # But the conversion from rotated box to polygon will
            # slow down the speed.
            # So we use max(x,y) + max(w,h) as max coordinate
            # which is larger than polygon max coordinate
            # max(x1, y1, x2, y2,x3, y3, x4, y4)
            max_coordinate = boxes[..., :2].max() + boxes[..., 2:4].max()
            offsets = idxs.to(boxes) * (
                max_coordinate + torch.tensor(1).to(boxes))
            boxes_ctr_for_nms = boxes[..., :2] + offsets[:, None]
            boxes_for_nms = torch.cat([boxes_ctr_for_nms, boxes[..., 2:5]],
                                      dim=-1)
        else:
            max_coordinate = boxes.max()
            offsets = idxs.to(boxes) * (
                max_coordinate + torch.tensor(1).to(boxes))
            boxes_for_nms = boxes + offsets[:, None]

    nms_op = nms_cfg_.pop('type', 'nms')
    if isinstance(nms_op, str):
        nms_op = eval(nms_op)

    split_thr = nms_cfg_.pop('split_thr', 10000)
    # Won't split to multiple nms nodes when exporting to onnx
    if boxes_for_nms.shape[0] < split_thr:
        dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]
        my_scores = my_scores[keep]

        # This assumes `dets` has arbitrary dimensions where
        # the last dimension is score.
        # Currently it supports bounding boxes [x1, y1, x2, y2, score] or
        # rotated boxes [cx, cy, w, h, angle_radian, score].

        scores = dets[:, -1]
    else:
        max_num = nms_cfg_.pop('max_num', -1)
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        # Some type of nms would reweight the score, such as SoftNMS
        scores_after_nms = scores.new_zeros(scores.size())
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True
            scores_after_nms[mask[keep]] = dets[:, -1]
        keep = total_mask.nonzero(as_tuple=False).view(-1)

        scores, inds = scores_after_nms[keep].sort(descending=True)
        keep = keep[inds]
        boxes = boxes[keep]

        if max_num > 0:
            keep = keep[:max_num]
            boxes = boxes[:max_num]
            scores = scores[:max_num]

    boxes = torch.cat([boxes, scores[:, None]], -1)
    # boxes = torch.cat([boxes, my_scores], -1)
    return my_scores, boxes, keep
    # return boxes, keep