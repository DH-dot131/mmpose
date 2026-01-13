from mmengine.config import read_base

custom_imports = dict(
    imports=['mmpose.models', 'mmpose.visualization', 'mmpose.datasets.kfold_dataset', 'mmpose.engine.optim_wrappers.layer_decay_optim_wrapper'],
    allow_failed_imports=False
)

with read_base():
    from mmpose.configs._base_.default_runtime import *  # noqa

# work_dir 설정 (lazy import 환경에서 os 모듈 사용을 피하기 위해 문자열 직접 사용)
work_dir = '../work_dirs/foot_lat/td-hm_ViTPose-base_1xb8-200e_foot-lat-384x384_4x'

fp16 = dict(loss_scale='dynamic')
auto_scale_lr = dict(base_batch_size=512)

dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = '../data/foot_lat_mmpose/'

codec = dict(
    type='UDPHeatmap', input_size=(384, 384), heatmap_size=(384, 384), sigma=2)

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale', padding=1.25),
    dict(type='RandomFlip', direction=['horizontal', 'vertical']),
    #ict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='PackPoseInputs')
]




train_dataloader = dict(
    batch_size=8,
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
        metainfo=dict(from_file='configs/_base_/datasets/custom_foot_lat_metainfo.py')
    )
)

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations_val.json',
        data_prefix=dict(img='images/'),
        pipeline=val_pipeline,
        metainfo=dict(from_file='configs/_base_/datasets/custom_foot_lat_metainfo.py'),
        bbox_file = None
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations_test.json',
        data_prefix=dict(img='images/'),
        pipeline=val_pipeline,
        metainfo=dict(from_file='configs/_base_/datasets/custom_foot_lat_metainfo.py'),
        bbox_file = None
    )
)

#val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations_val.json')
#test_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations_test.json')

val_evaluator = [
    dict(type='NME',norm_mode = 'keypoint_distance', keypoint_indices = [0,17], collect_device='gpu'),
    dict(type='AUC', collect_device = 'gpu'), 
    dict(type='EPE', collect_device='gpu'),
    dict(type='PCKAccuracy',thr=0.005, collect_device = 'gpu'),
    ]
test_evaluator = val_evaluator


train_cfg = dict(max_epochs=200, val_interval=2)

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.1),
    paramwise_cfg=dict(
        num_layers=12,
        layer_decay_rate=0.75,
        custom_keys={
            'bias': dict(decay_multi=0.0),
            'pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        },
    ),
    constructor='LayerDecayOptimWrapperConstructor',
    clip_grad=dict(max_norm=1., norm_type=2),
)

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=200,
        milestones=[160, 190],
        gamma=0.1,
        by_epoch=True)
]


model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='mmpretrain.VisionTransformer',
        arch='base',
        img_size=(384, 384),
        patch_size=16,
        qkv_bias=True,
        drop_path_rate=0.3,
        with_cls_token=False,
        out_type='featmap',
        patch_cfg=dict(padding=2),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/'
            'v1/pretrained_models/mae_pretrain_vit_base_20230913.pth'),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=768,
        out_channels=16,  # ← 커스텀 keypoints 수
        deconv_out_channels=(256, 256, 256, 256),  # deconv 4개로 16배 upsampling (18x24 -> 36x48 -> 72x96 -> 144x192 -> 288x384)
        deconv_kernel_sizes=(4, 4, 4, 4),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=False,
    ))

# custom_hooks에 추가
custom_hooks = [
    dict(type='SyncBuffersHook'),
    dict(
        type='EnhancedTensorBoardHook',
        log_grad_norm=True,
        log_weight_hist=True,
        log_val_metrics=True,
        log_train_images=True,
        # log_val_images removed (not effective, causes large file sizes)
        grad_norm_interval=100,
        train_image_interval=500,
    ),
]

