"""
mmpose 모델에서 heatmap을 추출하는 스크립트

이 스크립트는 inferencer_demo와 유사하게 개별 이미지나 이미지 폴더를 입력받아
heatmap을 추출하고 PNG 형식으로 저장합니다.
"""

import argparse
import inspect
import json
import os
import pickle
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from mmpose.apis.inference import inference_topdown, inference_bottomup, init_model
from mmpose.apis.visualization import visualize
from mmpose.registry import DATASETS
from mmpose.visualization import PoseLocalVisualizer
from mmpose.apis.inferencers.base_mmpose_inferencer import BaseMMPoseInferencer
from mmpose.apis.inferencers.utils import default_det_models
from mmpose.evaluation.functional import nms
from mmpose.structures.utils import revert_heatmap
from PIL import Image
from tqdm import tqdm

try:
    from mmdet.apis.det_inferencer import DetInferencer
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract heatmaps from mmpose model for individual images')
    parser.add_argument(
        'inputs',
        type=str,
        nargs='?',
        help='Input image path or folder path containing images')
    parser.add_argument('config', help='model config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='heatmaps_output',
        help='directory to save extracted heatmaps')
    parser.add_argument(
        '--format',
        choices=['pkl', 'npy', 'png', 'all'],
        default='all',
        help='output format for heatmaps (pkl, npy, png, or all)')
    parser.add_argument(
        '--save-individual',
        action='store_true',
        help='save individual heatmap for each keypoint')
    parser.add_argument(
        '--save-combined',
        action='store_true',
        help='save combined heatmap showing all keypoints')
    parser.add_argument(
        '--no-save-combined',
        dest='save_combined',
        action='store_false',
        help='do not save combined heatmap')
    parser.set_defaults(save_combined=True)
    parser.add_argument(
        '--save-paper-combined',
        action='store_true',
        help='save combined heatmap using only paper keypoints (3, 4, 9, 10, 13, 14 in 1-based)')
    parser.add_argument(
        '--paper-keypoint-indices',
        type=int,
        nargs='+',
        default=[3, 4, 9, 10, 13, 14],
        help='1-based keypoint indices for paper (default: 3, 4, 9, 10, 13, 14)')
    parser.add_argument(
        '--overlay-alpha',
        type=float,
        default=0.5,
        help='alpha value for overlaying heatmap on original image (0.0-1.0)')
    parser.add_argument(
        '--bbox',
        type=float,
        nargs=4,
        default=None,
        help='bounding box in format [x1, y1, x2, y2]. Only used for top-down models when det-model is not provided. If not provided, uses entire image')
    parser.add_argument(
        '--det-model',
        type=str,
        default=None,
        help='Config path or alias of detection model for top-down models. If not provided and model is top-down, uses entire image as bbox.')
    parser.add_argument(
        '--det-weights',
        type=str,
        default=None,
        help='Path to the checkpoints of detection model.')
    parser.add_argument(
        '--det-cat-ids',
        type=int,
        nargs='+',
        default=0,
        help='Category id for detection model.')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold for detection model')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='device used for inference')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config')
    parser.add_argument(
        '--save-keypoints',
        action='store_true',
        help='save keypoint predictions to JSON files (similar to --pred-out-dir in inferencer_demo)')
    parser.add_argument(
        '--pred-out-dir',
        type=str,
        default=None,
        help='directory to save keypoint predictions (if not specified, saves in output-dir)')
    parser.add_argument(
        '--vis-out-dir',
        type=str,
        default=None,
        help='directory to save visualized results with keypoints drawn')
    parser.add_argument(
        '--draw-bbox',
        action='store_true',
        help='whether to draw the bounding boxes')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='keypoint score threshold for visualization')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='link thickness for visualization')
    parser.add_argument(
        '--show-kpt-idx',
        type=int,
        nargs='*',
        default=None,
        help='display only specific keypoint indices (1-based). Other keypoints will be hidden. Example: --show-kpt-idx 3 4 9 10 13 14')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='skeleton style selection')
    return parser.parse_args()


def get_image_paths(input_path: str) -> List[str]:
    """입력 경로에서 이미지 파일 경로 리스트를 반환합니다."""
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise ValueError(f"입력 경로가 존재하지 않습니다: {input_path}")
    
    if input_path.is_file():
        # 단일 파일
        return [str(input_path)]
    elif input_path.is_dir():
        # 디렉토리 내의 모든 이미지 파일
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = [
            str(p) for p in input_path.iterdir()
            if p.suffix.lower() in image_extensions
        ]
        if not image_paths:
            raise ValueError(f"디렉토리에 이미지 파일이 없습니다: {input_path}")
        return sorted(image_paths)
    else:
        raise ValueError(f"유효하지 않은 입력 경로: {input_path}")


def resize_heatmap_to_image(heatmap: np.ndarray, target_size: tuple, 
                           input_center: Optional[np.ndarray] = None,
                           input_scale: Optional[np.ndarray] = None,
                           input_size: Optional[tuple] = None) -> np.ndarray:
    """Heatmap을 원본 이미지 크기로 resize합니다.
    
    Args:
        heatmap: (H, W) 형태의 heatmap 배열
        target_size: (width, height) 원본 이미지 크기
        input_center: bbox 중심점 (top-down 모델의 경우)
        input_scale: bbox 스케일 (top-down 모델의 경우)
        input_size: 입력 이미지 크기 (width, height)
    
    Returns:
        원본 이미지 크기로 resize된 heatmap
    """
    target_w, target_h = target_size
    hm_h, hm_w = heatmap.shape
    
    if input_center is not None and input_scale is not None:
        # Top-down 모델: revert_heatmap 함수를 사용하여 원본 이미지 좌표계로 변환
        # input_center와 input_scale을 numpy 배열로 변환
        if isinstance(input_center, (list, tuple)):
            input_center = np.array(input_center)
        if isinstance(input_scale, (list, tuple)):
            input_scale = np.array(input_scale)
        
        # revert_heatmap은 (K, H, W) 또는 (H, W) 형태를 받음
        heatmap_reverted = revert_heatmap(
            heatmap, 
            input_center, 
            input_scale, 
            (target_h, target_w)  # img_shape는 (height, width)
        )
        return heatmap_reverted
    else:
        # Bottom-up 모델 또는 전체 이미지: 단순 resize
        from scipy.ndimage import zoom
        scale_w = target_w / hm_w
        scale_h = target_h / hm_h
        heatmap_resized = zoom(heatmap, (scale_h, scale_w), order=1)
        return heatmap_resized


def save_heatmap_png(heatmap: np.ndarray, output_path: Path, 
                     original_img: Optional[np.ndarray] = None,
                     keypoint_idx: Optional[int] = None,
                     colormap: str = 'jet', alpha: float = 0.5):
    """Heatmap을 PNG 파일로 저장합니다. 원본 이미지가 있으면 overlay합니다.
    
    Args:
        heatmap: (H, W) 형태의 heatmap 배열
        output_path: 저장할 파일 경로
        original_img: 원본 이미지 (H, W, 3) 또는 (H, W) 형태
        keypoint_idx: keypoint 인덱스 (None이면 combined heatmap)
        colormap: matplotlib colormap 이름
        alpha: overlay 투명도 (0-1)
    """
    # 정규화 (0-1 범위)
    if heatmap.max() > 1.0:
        heatmap = heatmap / heatmap.max()
    
    # Colormap 적용
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # RGB만 추출
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # 원본 이미지와 overlay
    if original_img is not None:
        # 원본 이미지가 grayscale이면 RGB로 변환
        if len(original_img.shape) == 2:
            original_img = np.stack([original_img] * 3, axis=-1)
        elif original_img.shape[2] == 1:
            original_img = np.repeat(original_img, 3, axis=2)
        
        # 원본 이미지 정규화 (0-255 범위)
        if original_img.max() <= 1.0:
            original_img = (original_img * 255).astype(np.uint8)
        else:
            original_img = original_img.astype(np.uint8)
        
        # 크기 맞추기
        if heatmap_colored.shape[:2] != original_img.shape[:2]:
            from scipy.ndimage import zoom
            scale_h = original_img.shape[0] / heatmap_colored.shape[0]
            scale_w = original_img.shape[1] / heatmap_colored.shape[1]
            heatmap_colored = zoom(heatmap_colored, (scale_h, scale_w, 1), order=1)
            heatmap_colored = np.clip(heatmap_colored, 0, 255).astype(np.uint8)
        
        # Alpha blending
        overlay = (alpha * heatmap_colored + (1 - alpha) * original_img).astype(np.uint8)
        img = Image.fromarray(overlay)
    else:
        # Heatmap만 저장
        img = Image.fromarray(heatmap_colored)
    
    img.save(output_path)
    return heatmap_colored


def save_combined_heatmap(combined_heatmap: np.ndarray, output_path: Path,
                          original_img: Optional[np.ndarray] = None,
                          colormap: str = 'jet', alpha: float = 0.5):
    """모든 keypoint의 heatmap을 합쳐서 저장합니다. 원본 이미지가 있으면 overlay합니다.
    
    Args:
        combined_heatmap: (H, W) 형태의 combined heatmap 배열 (이미 max 연산 적용됨)
        output_path: 저장할 파일 경로
        original_img: 원본 이미지 (H, W, 3) 또는 (H, W) 형태
        colormap: matplotlib colormap 이름
        alpha: overlay 투명도 (0-1)
    """
    # 정규화
    if combined_heatmap.max() > 0:
        combined = combined_heatmap / combined_heatmap.max()
    else:
        combined = combined_heatmap
    
    # Colormap 적용
    cmap = plt.get_cmap(colormap)
    combined_colored = cmap(combined)[:, :, :3]
    combined_colored = (combined_colored * 255).astype(np.uint8)
    
    # 원본 이미지와 overlay
    if original_img is not None:
        # 원본 이미지가 grayscale이면 RGB로 변환
        if len(original_img.shape) == 2:
            original_img = np.stack([original_img] * 3, axis=-1)
        elif original_img.shape[2] == 1:
            original_img = np.repeat(original_img, 3, axis=2)
        
        # 원본 이미지 정규화 (0-255 범위)
        if original_img.max() <= 1.0:
            original_img = (original_img * 255).astype(np.uint8)
        else:
            original_img = original_img.astype(np.uint8)
        
        # 크기 맞추기
        if combined_colored.shape[:2] != original_img.shape[:2]:
            from scipy.ndimage import zoom
            scale_h = original_img.shape[0] / combined_colored.shape[0]
            scale_w = original_img.shape[1] / combined_colored.shape[1]
            combined_colored = zoom(combined_colored, (scale_h, scale_w, 1), order=1)
            combined_colored = np.clip(combined_colored, 0, 255).astype(np.uint8)
        
        # Alpha blending
        overlay = (alpha * combined_colored + (1 - alpha) * original_img).astype(np.uint8)
        img = Image.fromarray(overlay)
    else:
        # Heatmap만 저장
        img = Image.fromarray(combined_colored)
    
    img.save(output_path)
    return combined_colored


def save_keypoints_json(keypoints: np.ndarray, keypoint_scores: np.ndarray,
                        output_path: Path, metainfo: Optional[dict] = None):
    """Keypoint 예측 결과를 JSON 파일로 저장합니다 (inferencer_demo.py 스타일).
    
    Args:
        keypoints: (N, K, 2) 또는 (K, 2) 형태의 keypoint 좌표 배열
        keypoint_scores: (N, K) 또는 (K,) 형태의 keypoint score 배열
        output_path: 저장할 JSON 파일 경로
        metainfo: 메타 정보 딕셔너리 (img_path, img_id 등)
    """
    # numpy 배열을 리스트로 변환
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.cpu().numpy()
    if isinstance(keypoint_scores, torch.Tensor):
        keypoint_scores = keypoint_scores.cpu().numpy()
    
    # 차원 정규화: (K, 2) -> (1, K, 2)
    if keypoints.ndim == 2:
        keypoints = keypoints[np.newaxis, :, :]
    if keypoint_scores.ndim == 1:
        keypoint_scores = keypoint_scores[np.newaxis, :]
    
    # 각 인스턴스별로 저장
    predictions = []
    for inst_idx in range(keypoints.shape[0]):
        inst_keypoints = keypoints[inst_idx]  # (K, 2)
        inst_scores = keypoint_scores[inst_idx]  # (K,)
        
        # COCO 형식으로 변환: [x1, y1, v1, x2, y2, v2, ...]
        # v는 visibility: 0=not labeled, 1=labeled but not visible, 2=labeled and visible
        coco_keypoints = []
        for kp_idx in range(inst_keypoints.shape[0]):
            x, y = float(inst_keypoints[kp_idx, 0]), float(inst_keypoints[kp_idx, 1])
            score = float(inst_scores[kp_idx])
            # score > 0이면 visible (2), 아니면 not visible (0)
            visibility = 2 if score > 0 else 0
            coco_keypoints.extend([x, y, visibility])
        
        pred_dict = {
            'keypoints': coco_keypoints,
            'keypoint_scores': inst_scores.tolist(),
        }
        
        # 메타 정보 추가
        if metainfo is not None:
            if 'img_id' in metainfo:
                pred_dict['image_id'] = metainfo['img_id']
            if 'img_path' in metainfo:
                pred_dict['img_path'] = metainfo['img_path']
        
        predictions.append(pred_dict)
    
    # JSON 파일로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)


def init_detector(model_cfg: Config, det_model: Optional[str] = None,
                 det_weights: Optional[str] = None,
                 det_cat_ids: Optional[Union[int, List]] = None,
                 device: str = 'cuda:0'):
    """Detection 모델을 초기화합니다."""
    if not has_mmdet:
        return None, (0,)
    
    object_type = DATASETS.get(model_cfg.dataset_type).__module__.split(
        'datasets.')[-1].split('.')[0].lower()
    
    if det_model in ('whole_image', 'whole-image') or \
        (det_model is None and object_type not in default_det_models):
        return None, (0,)
    
    det_scope = 'mmdet'
    if det_model is None:
        det_info = default_det_models[object_type]
        det_model, det_weights, det_cat_ids = det_info[
            'model'], det_info['weights'], det_info['cat_ids']
    elif os.path.exists(det_model):
        det_cfg = Config.fromfile(det_model)
        det_scope = det_cfg.default_scope
    
    det_kwargs = dict(
        model=det_model,
        weights=det_weights,
        device=device,
        scope=det_scope,
    )
    # for compatibility with low version of mmdet
    if 'show_progress' in inspect.signature(DetInferencer).parameters:
        det_kwargs['show_progress'] = False
    
    detector = DetInferencer(**det_kwargs)
    
    if isinstance(det_cat_ids, (tuple, list)):
        det_cat_ids_tuple = det_cat_ids
    else:
        det_cat_ids_tuple = (det_cat_ids, )
    
    return detector, det_cat_ids_tuple


def get_bboxes_from_detector(detector, img_path: str, bbox_thr: float = 0.3,
                            nms_thr: float = 0.3, det_cat_ids: tuple = (0,)):
    """Detection 모델을 사용하여 bbox를 찾습니다."""
    try:
        det_results = detector(img_path, return_datasamples=True)['predictions']
    except ValueError:
        # for compatibility with low version of mmdet
        det_results = detector(img_path, return_datasample=True)['predictions']
    
    if len(det_results) == 0:
        return np.array([])
    
    pred_instance = det_results[0].pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    
    # Filter by category
    label_mask = np.zeros(len(bboxes), dtype=np.uint8)
    for cat_id in det_cat_ids:
        label_mask = np.logical_or(label_mask,
                                   pred_instance.labels == cat_id)
    
    bboxes = bboxes[np.logical_and(label_mask,
                                   pred_instance.scores > bbox_thr)]
    
    # NMS
    if len(bboxes) > 0:
        bboxes = bboxes[nms(bboxes, nms_thr)]
    
    return bboxes


def process_single_image(model: torch.nn.Module, model_cfg: Config, img_path: str, 
                        bbox: Optional[List[float]] = None,
                        detector: Optional[object] = None,
                        det_cat_ids: tuple = (0,),
                        bbox_thr: float = 0.3,
                        nms_thr: float = 0.3,
                        test_cfg: Optional[dict] = None) -> dict:
    """단일 이미지에 대해 heatmap을 추출합니다.
    
    Args:
        model: mmpose 모델
        img_path: 이미지 경로
        bbox: bounding box [x1, y1, x2, y2] (top-down 모델에서만 사용, None이면 전체 이미지)
        test_cfg: test config (output_heatmaps=True 포함)
    
    Returns:
        dict: heatmap, keypoints, metainfo를 포함한 딕셔너리
    """
    # test_cfg 설정
    if test_cfg is None:
        test_cfg = {}
    if model.test_cfg is None:
        model.test_cfg = {}
    model.test_cfg.update(test_cfg)
    model.test_cfg['output_heatmaps'] = True
    
    # data_mode 확인 (topdown 또는 bottomup)
    data_mode = None
    if hasattr(model_cfg, 'data_mode'):
        data_mode = model_cfg.data_mode
    elif hasattr(model_cfg, 'test_dataloader') and hasattr(model_cfg.test_dataloader, 'dataset'):
        data_mode = model_cfg.test_dataloader.dataset.get('data_mode', None)
    
    # data_mode가 없으면 모델 타입으로 판단
    if data_mode is None:
        model_type_str = str(type(model))
        if 'Bottomup' in model_type_str:
            data_mode = 'bottomup'
        elif 'Topdown' in model_type_str:
            data_mode = 'topdown'
        else:
            # 기본값은 topdown
            data_mode = 'topdown'
            print(f"Warning: data_mode를 확인할 수 없어 기본값 'topdown'을 사용합니다.")
    
    # 모델에 cfg 설정 (inference 함수들이 model.cfg를 사용하므로)
    if not hasattr(model, 'cfg'):
        model.cfg = model_cfg
    
    # Inference 수행
    if data_mode == 'bottomup':
        # Bottom-up 모델: bbox 불필요
        results = inference_bottomup(model, img_path)
    else:
        # Top-down 모델: bbox 필요
        bboxes_to_use = None
        
        # 1. Detector가 있으면 detector로 bbox 찾기
        if detector is not None:
            det_bboxes = get_bboxes_from_detector(
                detector, img_path, bbox_thr, nms_thr, det_cat_ids)
            if len(det_bboxes) > 0:
                bboxes_to_use = det_bboxes[:, :4]  # x1, y1, x2, y2만 추출
            else:
                print(f"Warning: {img_path}에서 bbox를 찾을 수 없어 전체 이미지를 사용합니다.")
        
        # 2. Detector가 없거나 bbox를 찾지 못한 경우
        if bboxes_to_use is None or len(bboxes_to_use) == 0:
            if bbox is not None:
                bboxes_to_use = np.array([bbox])
            else:
                # 전체 이미지를 bbox로 사용
                img = Image.open(img_path)
                w, h = img.size
                bboxes_to_use = np.array([[0, 0, w, h]])
        
        results = inference_topdown(model, img_path, bboxes=bboxes_to_use, bbox_format='xyxy')
    
    if len(results) == 0:
        return None
    
    result = results[0]
    
    # Heatmap 추출
    heatmaps = None
    if hasattr(result, 'pred_fields') and result.pred_fields is not None:
        heatmaps = result.pred_fields.heatmaps
        if isinstance(heatmaps, torch.Tensor):
            heatmaps = heatmaps.cpu().numpy()
    
    # Keypoints 추출
    keypoints = None
    keypoint_scores = None
    if hasattr(result, 'pred_instances'):
        keypoints = result.pred_instances.keypoints
        keypoint_scores = result.pred_instances.keypoint_scores
    
    # Metainfo
    metainfo = {
        'img_path': img_path,
        'data_mode': data_mode,
        'img_id': result.metainfo.get('img_id', -1),
        'input_size': result.metainfo.get('input_size', None),
        'input_center': result.metainfo.get('input_center', None),
        'input_scale': result.metainfo.get('input_scale', None),
    }
    
    return {
        'heatmaps': heatmaps,
        'keypoints': keypoints,
        'keypoint_scores': keypoint_scores,
        'metainfo': metainfo
    }


def main():
    args = parse_args()
    
    if args.inputs is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('inputs', type=str)
        parser.print_help()
        return
    
    # 이미지 경로 리스트 가져오기
    image_paths = get_image_paths(args.inputs)
    print(f"총 {len(image_paths)}개의 이미지를 처리합니다.")
    
    # 모델 초기화 (init_model 함수 사용 - dataset_meta와 cfg를 자동으로 설정)
    model = init_model(args.config, args.checkpoint, device=args.device, cfg_options=args.cfg_options if args.cfg_options else None)
    
    # cfg는 모델에서 가져오기 (init_model이 model.cfg에 설정함)
    cfg = model.cfg
    
    # test_cfg 설정
    if model.test_cfg is None:
        model.test_cfg = {}
    model.test_cfg['output_heatmaps'] = True
    
    # Detector 초기화 (top-down 모델인 경우)
    detector = None
    det_cat_ids_tuple = (args.det_cat_ids,)
    data_mode = None
    if hasattr(cfg, 'data_mode'):
        data_mode = cfg.data_mode
    elif hasattr(cfg, 'test_dataloader') and hasattr(cfg.test_dataloader, 'dataset'):
        data_mode = cfg.test_dataloader.dataset.get('data_mode', None)
    
    if data_mode != 'bottomup' and args.det_model is not None:
        detector, det_cat_ids_tuple = init_detector(
            cfg, args.det_model, args.det_weights, args.det_cat_ids, args.device)
        if detector is None:
            print("Warning: Detector 초기화에 실패했습니다. bbox를 수동으로 지정하거나 전체 이미지를 사용합니다.")
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 각 이미지 처리
    all_results = []
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        img_name = Path(img_path).stem
        
        # Heatmap 추출
        result = process_single_image(
            model, cfg, img_path, 
            bbox=args.bbox,
            detector=detector,
            det_cat_ids=det_cat_ids_tuple,
            bbox_thr=args.bbox_thr,
            nms_thr=args.nms_thr,
            test_cfg={'output_heatmaps': True}
        )
        
        if result is None or result['heatmaps'] is None:
            print(f"Warning: {img_path}에서 heatmap을 추출할 수 없습니다.")
            continue
        
        heatmaps = result['heatmaps']
        metainfo = result['metainfo']
        img_output_dir = output_dir / img_name
        img_output_dir.mkdir(exist_ok=True)
        
        # 원본 이미지 로드
        original_img = np.array(Image.open(img_path))
        if len(original_img.shape) == 3:
            # RGB 이미지
            img_h, img_w = original_img.shape[:2]
        else:
            # Grayscale 이미지
            img_h, img_w = original_img.shape
        
        # Heatmap을 원본 이미지 크기로 resize
        input_center = metainfo.get('input_center')
        input_scale = metainfo.get('input_scale')
        input_size = metainfo.get('input_size')
        
        # PNG 저장
        if args.format in ['png', 'all']:
            # 개별 keypoint heatmap 저장
            if args.save_individual:
                for kp_idx in range(heatmaps.shape[0]):
                    kp_heatmap = heatmaps[kp_idx]  # (H, W)
                    
                    # 원본 이미지 크기로 resize
                    kp_heatmap_resized = resize_heatmap_to_image(
                        kp_heatmap, (img_w, img_h),
                        input_center=input_center,
                        input_scale=input_scale,
                        input_size=input_size
                    )
                    
                    save_path = img_output_dir / f'keypoint_{kp_idx:02d}.png'
                    save_heatmap_png(
                        kp_heatmap_resized, save_path,
                        original_img=original_img,
                        keypoint_idx=kp_idx,
                        alpha=args.overlay_alpha
                    )
            
            # Combined heatmap 저장 (모든 keypoint)
            if args.save_combined:
                # Combined heatmap 계산
                combined_heatmap = np.max(heatmaps, axis=0)  # (H, W)
                
                # 원본 이미지 크기로 resize
                combined_heatmap_resized = resize_heatmap_to_image(
                    combined_heatmap, (img_w, img_h),
                    input_center=input_center,
                    input_scale=input_scale,
                    input_size=input_size
                )
                
                combined_path = img_output_dir / 'combined_heatmap.png'
                save_combined_heatmap(
                    combined_heatmap_resized, combined_path,
                    original_img=original_img,
                    alpha=args.overlay_alpha
                )
            
            # 논문용 Combined heatmap 저장 (특정 keypoint만)
            if args.save_paper_combined:
                # 1-based 인덱스를 0-based로 변환
                paper_kp_indices = [idx - 1 for idx in args.paper_keypoint_indices]
                # 유효한 인덱스만 필터링
                valid_indices = [idx for idx in paper_kp_indices if 0 <= idx < heatmaps.shape[0]]
                
                if len(valid_indices) > 0:
                    # 선택된 keypoint들의 heatmap만 사용
                    paper_heatmaps = heatmaps[valid_indices]  # (N, H, W)
                    paper_combined_heatmap = np.max(paper_heatmaps, axis=0)  # (H, W)
                    
                    # 원본 이미지 크기로 resize
                    paper_combined_heatmap_resized = resize_heatmap_to_image(
                        paper_combined_heatmap, (img_w, img_h),
                        input_center=input_center,
                        input_scale=input_scale,
                        input_size=input_size
                    )
                    
                    paper_combined_path = img_output_dir / 'combined_heatmap_paper.png'
                    save_combined_heatmap(
                        paper_combined_heatmap_resized, paper_combined_path,
                        original_img=original_img,
                        alpha=args.overlay_alpha
                    )
                else:
                    print(f"Warning: {img_path}에서 유효한 논문 keypoint 인덱스를 찾을 수 없습니다.")
        
        # Keypoint 저장 (JSON 형식)
        if args.save_keypoints:
            if result['keypoints'] is not None and result['keypoint_scores'] is not None:
                # pred_out_dir이 지정되면 그곳에 저장, 아니면 output_dir에 저장
                if args.pred_out_dir:
                    pred_output_dir = Path(args.pred_out_dir)
                    pred_output_dir.mkdir(parents=True, exist_ok=True)
                    json_path = pred_output_dir / f'{img_name}.json'
                else:
                    json_path = img_output_dir / 'keypoints.json'
                
                save_keypoints_json(
                    result['keypoints'],
                    result['keypoint_scores'],
                    json_path,
                    metainfo=metainfo
                )
            else:
                print(f"Warning: {img_path}에서 keypoint를 추출할 수 없습니다.")
        
        # Keypoint 시각화 저장
        if args.vis_out_dir:
            if result['keypoints'] is not None and result['keypoint_scores'] is not None:
                vis_output_dir = Path(args.vis_out_dir)
                vis_output_dir.mkdir(parents=True, exist_ok=True)
                vis_path = vis_output_dir / f'{img_name}.png'
                
                # keypoints를 numpy 배열로 변환
                kpts = result['keypoints']
                kpt_scores = result['keypoint_scores']
                
                if isinstance(kpts, torch.Tensor):
                    kpts = kpts.cpu().numpy()
                if isinstance(kpt_scores, torch.Tensor):
                    kpt_scores = kpt_scores.cpu().numpy()
                
                # 차원 정규화: (K, 2) -> (1, K, 2)
                if kpts.ndim == 2:
                    kpts = kpts[np.newaxis, :, :]
                if kpt_scores.ndim == 1:
                    kpt_scores = kpt_scores[np.newaxis, :]
                
                # 첫 번째 인스턴스만 사용 (일반적으로 단일 인스턴스)
                kpts_single = kpts[0]  # (K, 2)
                kpt_scores_single = kpt_scores[0]  # (K,)
                
                # dataset_meta 가져오기 (model에서)
                dataset_meta = None
                if hasattr(model, 'dataset_meta') and model.dataset_meta is not None:
                    dataset_meta = model.dataset_meta
                elif hasattr(cfg, 'dataset_meta') and cfg.dataset_meta is not None:
                    dataset_meta = cfg.dataset_meta
                
                # 시각화 수행
                # 특정 keypoint만 표시하는 경우 필터링
                if isinstance(args.show_kpt_idx, list) and len(args.show_kpt_idx) > 0:
                    # 1-based를 0-based로 변환
                    kpt_indices_to_show = [idx - 1 if idx > 0 else idx for idx in args.show_kpt_idx]
                    # 유효한 인덱스만 필터링
                    valid_indices = [idx for idx in kpt_indices_to_show if 0 <= idx < len(kpts_single)]
                    
                    if len(valid_indices) > 0:
                        # 특정 keypoint만 필터링
                        filtered_kpts = kpts_single[valid_indices]  # (N, 2)
                        filtered_scores = kpt_scores_single[valid_indices]  # (N,)
                        
                        # Visualizer를 직접 사용해서 특정 keypoint만 그리기
                        visualizer = PoseLocalVisualizer()
                        if dataset_meta is not None:
                            visualizer.set_dataset_meta(dataset_meta, skeleton_style=args.skeleton_style)
                        
                        # 원본 이미지 로드 (RGB 형식)
                        img = np.array(Image.open(img_path))
                        if len(img.shape) == 2:
                            # Grayscale을 RGB로 변환
                            img = np.stack([img] * 3, axis=-1)
                        elif img.shape[2] == 1:
                            img = np.repeat(img, 3, axis=2)
                        # PIL은 RGB, OpenCV는 BGR이므로 변환 필요 없음 (visualize 함수가 처리)
                        visualizer.set_image(img)
                        
                        # 특정 keypoint만 그리기 (skeleton은 그리지 않음)
                        # keypoint만 직접 그리기 (인덱스 번호 없이)
                        # 파란색으로 고정 (RGB: (0, 0, 255), BGR: (255, 0, 0))
                        blue_color = (0, 0, 255)  # RGB 형식 (OpenCV 사용)
                        
                        for i, (kpt, score) in enumerate(zip(filtered_kpts, filtered_scores)):
                            if score >= args.kpt_thr:
                                # keypoint만 그리기 (인덱스 텍스트 없이, 파란색)
                                # kpt는 (2,) 형태의 배열이어야 함
                                visualizer.draw_circles(
                                    kpt,  # (2,) 형태
                                    radius=np.array([args.radius]),
                                    face_colors=blue_color,
                                    edge_colors=blue_color,
                                    alpha=1.0,
                                    line_widths=args.radius
                                )
                        
                        vis_img = visualizer.get_image()
                    else:
                        # 유효한 인덱스가 없으면 일반 시각화
                        vis_img = visualize(
                            img=img_path,
                            keypoints=kpts_single,
                            keypoint_score=kpt_scores_single,
                            metainfo=dataset_meta,
                            show_kpt_idx=False,
                            skeleton_style=args.skeleton_style,
                            show=False,
                            kpt_thr=args.kpt_thr
                        )
                else:
                    # 모든 keypoint 표시
                    vis_img = visualize(
                        img=img_path,
                        keypoints=kpts_single,
                        keypoint_score=kpt_scores_single,
                        metainfo=dataset_meta,
                        show_kpt_idx=False,
                        skeleton_style=args.skeleton_style,
                        show=False,
                        kpt_thr=args.kpt_thr
                    )
                
                # 이미지 저장
                vis_img_pil = Image.fromarray(vis_img)
                vis_img_pil.save(vis_path)
            else:
                print(f"Warning: {img_path}에서 keypoint를 추출할 수 없어 시각화를 저장할 수 없습니다.")
        
        # 결과 저장
        all_results.append({
            'img_path': img_path,
            'img_name': img_name,
            **result
        })
    
    # 전체 결과를 파일로 저장
    if args.format in ['pkl', 'all']:
        pkl_path = output_dir / 'all_heatmaps.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"\n전체 결과가 {pkl_path}에 저장되었습니다.")
    
    if args.format in ['npy', 'all']:
        # Heatmap만 numpy 배열로 저장
        heatmap_list = [r['heatmaps'] for r in all_results if r['heatmaps'] is not None]
        if heatmap_list:
            npy_path = output_dir / 'all_heatmaps.npy'
            np.save(npy_path, np.array(heatmap_list))
            print(f"Heatmap 배열이 {npy_path}에 저장되었습니다.")
            
            # 메타 정보도 별도 저장
            metainfo_list = [r['metainfo'] for r in all_results]
            metainfo_path = output_dir / 'metainfo.pkl'
            with open(metainfo_path, 'wb') as f:
                pickle.dump(metainfo_list, f)
            print(f"메타 정보가 {metainfo_path}에 저장되었습니다.")
    
    print(f"\n✅ 완료! 총 {len(all_results)}개의 이미지에서 heatmap을 추출했습니다.")
    if len(all_results) > 0 and all_results[0]['heatmaps'] is not None:
        hm_shape = all_results[0]['heatmaps'].shape
        print(f"   Heatmap shape: {hm_shape} (Keypoints: {hm_shape[0]}, H: {hm_shape[1]}, W: {hm_shape[2]})")
    
    if args.save_keypoints:
        if args.pred_out_dir:
            print(f"   Keypoint 예측 결과가 {args.pred_out_dir}에 저장되었습니다.")
        else:
            print(f"   Keypoint 예측 결과가 각 이미지 디렉토리의 keypoints.json 파일에 저장되었습니다.")
    
    if args.vis_out_dir:
        print(f"   Keypoint 시각화 결과가 {args.vis_out_dir}에 저장되었습니다.")


if __name__ == '__main__':
    main()
