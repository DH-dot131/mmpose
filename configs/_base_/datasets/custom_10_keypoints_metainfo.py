# configs/_base_/datasets/custom_20_keypoints_metainfo.py

keypoint_info = {
    i: dict(name=f'{i+1}', id=i, color=[255, 0, 0], type='upper')
    for i in range(10)
}

dataset_info = dict(
    dataset_name='custom10',
    paper_info=dict(),
    keypoint_info=keypoint_info,
    skeleton_info=dict(),
    joint_weights=[1.0] * 10,
    sigmas=[0.05] * 10
)

