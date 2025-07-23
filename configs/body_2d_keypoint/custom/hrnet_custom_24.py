_base_ = ['../../_base_/default_runtime.py']

# configs/_base_/datasets/custom_coco_24.py

dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = '../data/foot_ap_mmpose/'

codec = dict(
    type='MSRAHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    #dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs'),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations_train.json',
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
        metainfo=dict(from_file='configs/_base_/datasets/custom_24_keypoints_metainfo.py')
    )
)

val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations_val.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=dict(from_file='configs/_base_/datasets/custom_24_keypoints_metainfo.py')
    )
)

test_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations_test.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=dict(from_file='configs/_base_/datasets/custom_24_keypoints_metainfo.py')
    )
)

#val_evaluator = dict(type='CocoMetric')
#test_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations_test.json')

val_evaluator = dict(type='NME',norm_mode = 'keypoint_distance')
test_evaluator = dict(type='NME',norm_mode = 'keypoint_distance')

#load_from = 'https://download.openmmlab.com/mmpose/pretrain_models/hrformer_base-32815020_20220226.pth'

train_cfg = dict(max_epochs=100, val_interval=10)

optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=5e-4,
        momentum=0.9,
        weight_decay=0.0001
    )
)


param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,   # 초기 LR = base_lr × 0.001
        end=5,                # 5 epoch 동안 warm-up
        by_epoch=True
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=95,             # 95 epoch 동안 cosine decay
        begin=5,              # 5 epoch부터 시작
        end=100,              # 전체 학습은 100 epoch
        by_epoch=True
    )
]

auto_scale_lr = dict(base_batch_size=32)
#default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))
default_hooks = dict(checkpoint=dict(save_best='NME', rule='less'))


norm_cfg = dict(type='SyncBN', requires_grad=True)



model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict( # config of backbone
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
        init_cfg=dict(
            type='Pretrained', # load pretrained weights to backbone
            checkpoint='https://download.openmmlab.com/mmpose'
            '/pretrain_models/hrnet_w32-36af842e.pth'),
        #init_cfg=None
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=32,
        out_channels=24,  # ← 커스텀 keypoints 수
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True
    )
)

fp16 = dict(loss_scale='dynamic')