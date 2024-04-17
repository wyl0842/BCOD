input torch

# 输入
bboxes_dets = torch.tensor([[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]], dtype=torch.float32)
bboxes_scores = torch.tensor([0.8, 0.7, 0.6], dtype=torch.float32)
k = 2
agg_thr = 0.5

# 调用 agg_scores
output_bboxes_dets, output_bboxes_scores = agg_scores(bboxes_dets, bboxes_scores, k, agg_thr)

# 输出
print("Output bboxes_dets:")
print(output_bboxes_dets)
print("Output bboxes_scores:")
print(output_bboxes_scores)


def agg_scores(bboxes_dets, bboxes_scores, k, agg_thr):
    bboxes_dets = bboxes_dets.clone()
    bboxes_scores = bboxes_scores.clone()
    iou = calc_iou(bboxes_dets, bboxes_dets)

    agg_inds = iou >= agg_thr
    iou_weight = iou[agg_inds] - 1
    iou_weight = k ** iou_weight

    # 保存原始 scores 的形状
    orig_scores_shape = bboxes_scores.shape

    # 将 bboxes_scores 从 2D 张量转换为 1D 张量
    bboxes_scores_flat = bboxes_scores.view(-1)

    # 获取 agg_inds 的索引位置
    agg_inds_flat = agg_inds.view(-1)

    # 创建一个与 bboxes_scores_flat 相同形状的零张量，用于存储 agg_scores
    agg_scores_flat = torch.zeros_like(bboxes_scores_flat)

    # 使用索引来计算 agg_scores
    agg_scores_flat[agg_inds_flat] = (iou_weight * bboxes_scores_flat[agg_inds_flat]).sum()

    # 将 agg_scores 重新恢复为原始形状
    agg_scores = agg_scores_flat.view(orig_scores_shape)

    # 更新 bboxes_scores
    bboxes_scores[agg_inds] = agg_scores[agg_inds]

    return bboxes_dets, bboxes_scores