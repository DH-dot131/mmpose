import os, inspect
custom_imports = dict(
    imports=['mmpose.datasets.kfold_dataset'],
    allow_failed_imports=False
)

_base_ = ['../../_base_/default_runtime.py', '../topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-384x288.py']

# 현재 실행 중인 config 파일 경로
_config_path = inspect.getfile(inspect.currentframe())
# 파일명만 추출 (확장자 .py 제외)
_cfg_name = os.path.splitext(os.path.basename(_config_path))[0]

# work_dir에 동적으로 추가
work_dir = os.path.join(
    '..', 'work_dirs', 'foot_ap', _cfg_name
)

fp16 = dict(loss_scale='dynamic')
auto_scale_lr = dict(base_batch_size=256, enable = True)


dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = '../data/foot_ap_mmpose/'

codec = dict(
    type='MSRAHeatmap', input_size=(288, 384), heatmap_size=(72, 96), sigma=2)

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
    dict(type='PackPoseInputs')
]

base_dataset_train = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file= 'annotations_train_val.json',
    pipeline=train_pipeline,
    data_prefix=dict(img='images/'),
    metainfo=dict(from_file='configs/_base_/datasets/custom_20_keypoints_metainfo.py')
)

base_dataset_val = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file= 'annotations_train_val.json',
    pipeline=val_pipeline,
    data_prefix=dict(img='images/'),
    metainfo=dict(from_file='configs/_base_/datasets/custom_20_keypoints_metainfo.py')
)

# train_dataloader = dict(
#     batch_size=8,
#     num_workers=8,
#     dataset=dict(
#         _delete_=True, 
#         type= 'KFoldDataset',
#         dataset=base_dataset_train,
#         fold = 0,
#         num_splits=5,
#         test_mode=False,
#         seed=42,
#     ),
#     sampler=dict(type='DefaultSampler', shuffle=True)
# )

# val_dataloader = dict(
#     batch_size=8,
#     num_workers=8,
#     dataset=dict(
#         _delete_=True, 
#         type= 'KFoldDataset',
#         dataset=base_dataset_val,
#         fold = 0,
#         num_splits=5,
#         test_mode=True,
#         seed=42,
#     ),
#     sampler=dict(type='DefaultSampler', shuffle=False)
# )


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
        metainfo=dict(from_file='configs/_base_/datasets/custom_20_keypoints_metainfo.py')
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
        metainfo=dict(from_file='configs/_base_/datasets/custom_20_keypoints_metainfo.py'),
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
        metainfo=dict(from_file='configs/_base_/datasets/custom_20_keypoints_metainfo.py'),
        bbox_file = None
    )
)

#val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations_val.json')
#test_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations_test.json')

val_evaluator = [
    dict(type='EPE', collect_device='gpu'),
    dict(type='NME',norm_mode = 'keypoint_distance', keypoint_indices = [0,17], collect_device='gpu'),
    dict(type='AUC', collect_device = 'gpu'), 
    dict(type='PCKAccuracy',thr=0.005, collect_device = 'gpu'),
    ]
test_evaluator = val_evaluator


train_cfg = dict(max_epochs=100, val_interval=1)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='AdamW',
    lr=1e-2,
    weight_decay=0.05,
))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.01,   # 초기 LR = base_lr × 0.001
        end=3,                # 10 epoch 동안 warm-up
        by_epoch=True
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=97,             # 95 epoch 동안 cosine decay
        begin=3,              # 5 epoch부터 시작
        end=100,              # 전체 학습은 100 epoch
        by_epoch=True
    )
]


norm_cfg = dict(type='SyncBN', requires_grad=True)


model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
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
        frozen_stages=-1,
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=32,
        out_channels=20,  # ← 커스텀 keypoints 수
    ),
)


visualizer = dict(
    name='visualizer',
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='TensorboardVisBackend'),
    ])