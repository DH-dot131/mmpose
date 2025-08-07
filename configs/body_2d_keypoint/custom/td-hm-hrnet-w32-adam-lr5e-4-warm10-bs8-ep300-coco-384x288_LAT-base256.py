import os, inspect
from mmpretrain.datasets import KFoldDataset

_base_ = ['td-hm-hrnet-w32-adam-lr5e-4-warm10-bs8-ep300-coco-384x288_AP-base256.py']

# 현재 실행 중인 config 파일 경로
_config_path = inspect.getfile(inspect.currentframe())
# 파일명만 추출 (확장자 .py 제외)
_cfg_name = os.path.splitext(os.path.basename(_config_path))[0]

# work_dir에 동적으로 추가
work_dir = os.path.join(
    '..', 'work_dirs', 'foot_lat', _cfg_name
)


data_root = '../data/foot_LAT_mmpose/'
dataset_type = 'CocoDataset'

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

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

base_dataset_train = dict(
    type=dataset_type,
    data_root=data_root,
    metainfo=dict(from_file='configs/_base_/datasets/custom_foot_lat_metainfo.py')
)

base_dataset_val = dict(
    type=dataset_type,
    data_root=data_root,
    metainfo=dict(from_file='configs/_base_/datasets/custom_foot_lat_metainfo.py')
)

train_dataloader = dict(
    dataset=dict(
        type= 'KFoldDataset',
        dataset=base_dataset_train,
    ),
)

val_dataloader = dict(
    dataset=dict(
        type= 'KFoldDataset',
        dataset=base_dataset_val,
    ),
)

test_dataloader = dict(

    dataset=dict(
        data_root=data_root,
        metainfo=dict(from_file='configs/_base_/datasets/custom_foot_lat_metainfo.py'),
    )
)

#val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations_val.json')
#test_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations_test.json')

val_evaluator = [dict(type='NME',norm_mode = 'keypoint_distance', keypoint_indices = [0,12])]
test_evaluator = [
    dict(
        keypoint_indices=[
            0,
            12,
        ], norm_mode='keypoint_distance', type='NME'),
    dict(out_file_path= work_dir + '/results.pkl', type='DumpResults'),
]


model = dict(
    head=dict(
        type='HeatmapHead',
        in_channels=32,
        out_channels=16,  # ← 커스텀 keypoints 수
    ),
)
