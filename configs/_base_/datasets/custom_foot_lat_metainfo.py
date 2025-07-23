# configs/_base_/datasets/custom_24_keypoints_metainfo.py

keypoint_info = {
    i: dict(name=f'{i+1}', id=i, color=[255, 0, 0], type='upper')
    for i in range(16)
}

dataset_info = dict(
    dataset_name='custom16',
    paper_info=dict(),
    keypoint_info=keypoint_info,
    skeleton_info=dict(),
    joint_weights=[1.0] * 16,
    sigmas=[0.05] * 16
)

