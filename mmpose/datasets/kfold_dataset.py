# mmpose/datasets/kfold_dataset.py

from mmpretrain.datasets.dataset_wrappers import KFoldDataset as _PretrainKFold
from mmpose.registry import DATASETS

@DATASETS.register_module()
class KFoldDataset(_PretrainKFold):
    """Expose MMPretrain's KFoldDataset under the mmpose::dataset registry."""
    pass
