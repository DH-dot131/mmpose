import os, inspect
custom_imports = dict(
    imports=['mmpose.datasets.kfold_dataset'],
    allow_failed_imports=False
)

_base_ = ['../../_base_/default_runtime.py']

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

# DINOv3 ViT-L/16는 patch_size=16이므로 512x512 -> 32x32 feature map
codec = dict(
    type='MSRAHeatmap', input_size=(512, 512), heatmap_size=(32, 32), sigma=3)

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
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
        metainfo=dict(from_file='configs/_base_/datasets/custom_20_keypoints_metainfo.py')
    )
)

val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations_val.json',
        data_prefix=dict(img='images/'),
        pipeline=val_pipeline,
        metainfo=dict(from_file='configs/_base_/datasets/custom_20_keypoints_metainfo.py'),
        bbox_file = None
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations_test.json',
        data_prefix=dict(img='images/'),
        pipeline=val_pipeline,
        metainfo=dict(from_file='configs/_base_/datasets/custom_20_keypoints_metainfo.py'),
        bbox_file = None
    )
)

val_evaluator = [
    dict(type='NME',norm_mode = 'keypoint_distance', keypoint_indices = [0,17], collect_device='gpu'),
    dict(type='AUC', collect_device = 'gpu'), 
    dict(type='PCKAccuracy', collect_device = 'gpu'),
]
test_evaluator = val_evaluator


train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='AdamW',
    lr=1e-3,
    weight_decay=0.05,
))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        end=2,
        by_epoch=True
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=48,
        begin=2,
        end=50,
        by_epoch=True
    )
]


model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[0., 0., 0.],
        std=[1., 1., 1.],
        bgr_to_rgb=True),
    backbone=dict(
        type='DINOv3',
        pretrained='facebook/dinov3-vitl16-pretrain-lvd1689m',
        frozen=True,  # False: fine-tuning, True: frozen feature extractor
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=1024,  # DINOv3 ViT-L embed_dim
        out_channels=20,  # 커스텀 keypoints 수
        deconv_out_channels=None,  # No deconv, use native 32x32 resolution
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec,
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    )
)


visualizer = dict(
    name='visualizer',
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='TensorboardVisBackend'),
    ])

# hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='AUC',
        rule='greater',
        max_keep_ckpts=5,
        interval=2,
        by_epoch=True,
        save_last=False,
    ),
    logger=dict(type='LoggerHook', interval=100),
)

# 환경 설정
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(
    type='LogProcessor', 
    window_size=50, 
    by_epoch=True, 
    num_digits=6
)

log_level = 'INFO'
load_from = None
resume = False

