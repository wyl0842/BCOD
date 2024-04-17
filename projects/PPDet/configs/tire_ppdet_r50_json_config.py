# CUDA_VISIBLE_DEVICES=1 python tools/train.py projects/PPDet/configs/tire_ppdet_r50_json_config.py
_base_ = [
    'tire_base.py'
]

custom_imports = dict(
    imports=['projects.PPDet.ppdet'], allow_failed_imports=False)
# custom_imports = dict(
#     imports=['projects.DiffusionDet.diffusiondet'], allow_failed_imports=False)

# schedule_1x
# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    # dict(
    #     type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=500,
        by_epoch=True,
        milestones=[24, 33],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01/8*2, momentum=0.9, weight_decay=0.0001))

# # learning rate
# param_scheduler = [
#     # type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=500,
#         by_epoch=True,
#         milestones=[24, 33],
#         gamma=0.1)
# ]

# # optimizer
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001),
#     paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
#     clip_grad=dict(max_norm=35, norm_type=2))

auto_scale_lr = dict(enable=False, base_batch_size=16)

load_from = 'checkpoints/fovea_r50_fpn_4x4_1x_coco_20200219-ee4d5303.pth'
# load_from = None

# model
model = dict(
    type='PPDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        # dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        # stage_with_dcn=(False, True, True, True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        num_outs=5,
        add_extra_convs='on_input'),
    bbox_head=dict(
        type='PPDetHead',
        num_classes=10,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        base_edge_list=[16, 32, 64, 128, 256],
        scale_ranges=((1, 64), (32, 128), (64, 256), (128, 512), (256, 2048)),
        sigma=0.4,
        with_deform=False,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=1.50,
            alpha=0.4,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.75)),
    # training and testing settings
    train_cfg=dict(),
    test_cfg=dict(
        nms_pre=1000,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100,
        k=40,
        agg_thr=0.6))

work_dir = './outputs/tire_ppdet_r50'
seed = 10086