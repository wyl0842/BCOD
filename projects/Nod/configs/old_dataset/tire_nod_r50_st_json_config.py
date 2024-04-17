# CUDA_VISIBLE_DEVICES=1 python tools/train.py projects/PPDet/configs/tire_ppdet_r50_json_config.py
_base_ = [
    'tire_base.py', 'nod_r50.py'
]

custom_imports = dict(
    imports=['projects.Nod.nod'], allow_failed_imports=False)
# custom_imports = dict(
#     imports=['projects.DiffusionDet.diffusiondet'], allow_failed_imports=False)

# schedule_1x
# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
val_cfg = dict(type='ValLoop')
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

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01/8*2, momentum=0.9, weight_decay=0.0001))

auto_scale_lr = dict(enable=False, base_batch_size=16)

# load_from = 'checkpoints/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco-ae4d8b3d.pth'
# load_from = None
load_from = 'checkpoints/tood_r50_fpn_mstrain_2x_coco_20211210_144231-3b23174c.pth'

# model
detector = _base_.model
detector.train_cfg.assigner = dict(type='NODAssigner', topk=13)

model = dict(
    _delete_=True,
    type='STDetector',
    detector=detector,
    data_preprocessor=detector.data_preprocessor,
    semi_train_cfg=dict(freeze_teacher=True),
    semi_test_cfg=dict(predict_on='teacher'))

custom_hooks = [dict(type='MeanTeacherHook')]

work_dir = './outputs/tire_nod_r50'
seed = 10086