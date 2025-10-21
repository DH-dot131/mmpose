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

# RAD-DINO는 37×37 feature map을 생성하므로 heatmap_size 조정
codec = dict(
    type='MSRAHeatmap', input_size=(518, 518), heatmap_size=(37, 37), sigma=3) # 384x288 -> 518x518

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
        start_factor=0.001,   # 초기 LR = base_lr × 0.001
        end=2,              # 100 iteration 동안 warm-up
        by_epoch=True
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=48,             # 95 epoch 동안 cosine decay
        begin=2,              # 5 epoch부터 시작
        end=50,              # 전체 학습은 100 epoch
        by_epoch=True
    )
]


norm_cfg = dict(type='SyncBN', requires_grad=True)


model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        # mean=[123.675, 116.28, 103.53],
        # std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='RadDINO',
        pretrained='microsoft/rad-dino',
        frozen=True,  # False: fine-tuning, True: frozen feature extractor
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=768,  # RAD-DINO 출력 채널 (37×37×768)
        out_channels=20,  # 커스텀 keypoints 수
        # deconv_out_channels=(256,), # heatmap 해상도 증가
        # deconv_kernel_sizes=(4,),
        deconv_out_channels=None,
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

