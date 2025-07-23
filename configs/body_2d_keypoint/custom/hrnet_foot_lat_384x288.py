_base_ = ['../../_base_/default_runtime.py']

# configs/_base_/datasets/custom_coco_24.py
work_dir = '../work_dirs/foot_lat'

dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = '../data/foot_LAT_mmpose/'

codec = dict(
    type='MSRAHeatmap', input_size=(288, 384), heatmap_size=(72, 96), sigma=3)

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction=['horizontal', 'vertical'], prob=0.5),
    #dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]
'''
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),

    # — Geometric Augmentation —
    # 좌우 뒤집기 (발 좌/우 반전)
    dict(type='RandomFlip', direction='horizontal', prob=0.5),

    # BBox 기반 중심/스케일 소폭 변형
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.1,    # 중심 ±10%
        scale_factor=(0.2, 0.2)     # 스케일 ±20%
    ),

    # — Photometric Augmentation (흑백 전용) —
    dict(
        type='Albumentation',
        transforms=[
            # 대비 강화 (CLAHE)
            dict(type='CLAHE', clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            # 감마 보정
            dict(type='RandomGamma', gamma_limit=(80, 120), p=0.5),
            # 가우시안 노이즈
            dict(type='GaussNoise', var_limit=(5.0, 30.0), p=0.5),
            # 가벼운 블러
            dict(type='GaussianBlur', blur_limit=3, p=0.3),
        ],
        keymap={'img': 'image'},
    ),

    # — 공통 변환 & 타겟 생성 —
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]
'''
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs'),
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
        metainfo=dict(from_file='configs/_base_/datasets/custom_foot_lat_metainfo.py')
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
        metainfo=dict(from_file='configs/_base_/datasets/custom_foot_lat_metainfo.py')
    )
)

#val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations_val.json')
#test_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations_test.json')

val_evaluator = dict(type='NME',norm_mode = 'keypoint_distance', keypoint_indices = [0,12])
test_evaluator = dict(type='NME',norm_mode = 'keypoint_distance', keypoint_indices = [0,12])

#load_from = 'https://download.openmmlab.com/mmpose/pretrain_models/hrformer_base-32815020_20220226.pth'

train_cfg = dict(max_epochs=300, val_interval=5)
'''
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=5e-2,  # base learning rate
        momentum=0.9,
        weight_decay=0.0001
    )
)
'''
# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=1e-2,
))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,   # 초기 LR = base_lr × 0.001
        end=300,                # 1 epoch 동안 warm-up
        by_epoch=False
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=295,             # 95 epoch 동안 cosine decay
        begin=5,              # 5 epoch부터 시작
        end=300,              # 전체 학습은 100 epoch
        by_epoch=True
    )
]


auto_scale_lr = dict(base_batch_size=256)
'''
#default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))
'''
# default_hooks = dict(checkpoint=dict(save_best='NME', rule='less'))
default_hooks = dict(
    checkpoint=dict(
        save_best='NME',
        rule='less'
    ),
    early_stop=dict(
        type='EarlyStoppingHook',
        monitor='NME',    # 혹은 로그에 찍히는 정확한 키로 바꿔주세요
        rule='less',          # NME 는 낮을수록 좋으므로 'less'
        patience=5,           # 개선 없을 때 몇 에폭 대기할지
        min_delta=0
    )
)




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
        out_channels=16,  # ← 커스텀 keypoints 수
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