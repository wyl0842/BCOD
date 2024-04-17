# CUDA_VISIBLE_DEVICES=1 python tools/train.py projects/PPDet/configs/tire_ppdet_r50_json_config.py
_base_ = [
    'voc2coco_base_0.4.py', '../nod_r50.py'
]

custom_imports = dict(
    imports=['projects.Nod.nod'], allow_failed_imports=False)
# custom_imports = dict(
#     imports=['projects.DiffusionDet.diffusiondet'], allow_failed_imports=False)

num_epochs = 36
# schedule_1x
# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=num_epochs, val_interval=1)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    # type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=500,
        by_epoch=True,
        milestones=[24, 33],
        gamma=0.1)
]

set_batch_size = 4

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001/8*set_batch_size, momentum=0.9, weight_decay=0.0001))

auto_scale_lr = dict(enable=False, base_batch_size=16)

# load_from = 'checkpoints/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco-ae4d8b3d.pth'
# load_from = None
load_from = 'checkpoints/tood_r50_fpn_mstrain_2x_coco_20211210_144231-3b23174c.pth'

num_classes=20
# model
detector = _base_.model
detector.bbox_head = dict(
    type='MTLCHead',
    num_classes=num_classes,
    in_channels=256,
    stacked_convs=6,
    feat_channels=256,
    anchor_type='anchor_free',
    anchor_generator=dict(
        type='AnchorGenerator',
        ratios=[1.0],
        octave_base_scale=8,
        scales_per_octave=1,
        strides=[8, 16, 32, 64, 128]),
    bbox_coder=dict(
        type='DeltaXYWHBBoxCoder',
        target_means=[.0, .0, .0, .0],
        target_stds=[0.1, 0.1, 0.2, 0.2]),
    initial_loss_cls=dict(
        type='FocalLoss',
        use_sigmoid=True,
        activated=True,  # use probability instead of logit as input
        gamma=2.0,
        alpha=0.25,
        loss_weight=1.0),
    loss_cls=dict(
        type='QualityFocalLoss',
        use_sigmoid=True,
        activated=True,  # use probability instead of logit as input
        beta=2.0,
        loss_weight=1.0),
    loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
    num_dcn=2)
detector.train_cfg = dict(
        initial_epoch=4,
        initial_assigner=dict(type='ATSSAssigner', topk=9),
        # assigner=dict(type='TaskAlignedAssignerModify', topk=13),
        assigner=dict(type='TaskAlignedAssigner', topk=13),
        alpha=1,
        beta=6,
        allowed_border=-1,
        pos_weight=-1,
        debug=False)

model = dict(
    _delete_=True,
    type='MTLCMDDetector',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    semi_train_cfg=dict(
        freeze_teacher=True,
        sup_weight=1.0,
        unsup_weight=1.0,
        cls_pseudo_thr=0.9,
        min_pseudo_bbox_wh=(1e-2, 1e-2),
        one_hot_smoother=0.1,
        warmup_step=10,
        tau_c_max=0.95,
        tau_clean=0.75,
        num_classes=num_classes,
        num_epochs=num_epochs,
        lc_assigner = dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='ClassificationCost', weight=1.),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    semi_test_cfg=dict(predict_on='teacher'))

train_dataloader = dict(batch_size=set_batch_size)

custom_hooks = [dict(type='MeanTeacherHook')]

work_dir = './outputs/voc_dataset/voc_0.4_nod_mtlc_r50_modi'
seed = 10086