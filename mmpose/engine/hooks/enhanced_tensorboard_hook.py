# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
from typing import Dict, Optional, Sequence

import cv2
import mmcv
import mmengine
import mmengine.fileio as fileio
import numpy as np
import torch
from mmengine.dist import master_only
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer
from mmengine.logging import print_log

from mmpose.registry import HOOKS
from mmpose.structures import PoseDataSample, merge_data_samples

try:
    from torch.utils.tensorboard import SummaryWriter
    from torchvision.utils import make_grid
except ImportError:
    SummaryWriter = None
    make_grid = None


@HOOKS.register_module()
class EnhancedTensorBoardHook(Hook):
    """Enhanced TensorBoard Logging Hook for MMPose.

    This hook logs comprehensive training information to TensorBoard including:
    - Gradient norms (layer-wise and global average)
    - Weight histograms (layer-wise)
    - Validation metrics (PCK, NME, AUC, EPE, etc.)
    - Per-keypoint losses
    - Training/validation images with predictions
    - Learning rate schedule
    - Model parameter statistics

    Args:
        log_grad_norm (bool): Whether to log gradient norms. Defaults to True.
        log_weight_hist (bool): Whether to log weight histograms. Defaults to True.
        log_val_metrics (bool): Whether to log validation metrics. Defaults to True.
        log_train_images (bool): Whether to log training images. Defaults to True.
        log_val_images (bool): Whether to log validation images. Defaults to True.
        log_lr (bool): Whether to log learning rate. Defaults to True.
        log_model_stats (bool): Whether to log model statistics. Defaults to True.
        grad_norm_interval (int): Interval for logging gradient norms. Defaults to 100.
        weight_hist_interval (int): Interval (in epochs) for logging weight histograms.
            Defaults to 1.
        train_image_interval (int): Interval for logging training images. Defaults to 500.
        val_image_interval (int): Interval for logging validation images. Defaults to 10.
        max_layers_to_log (int): Maximum number of layers to log individually.
            Defaults to 20.
        max_val_images (int): Maximum number of validation images to store. Defaults to 4.
    """

    # Set priority to run after other hooks
    priority = 'VERY_LOW'

    def __init__(
        self,
        log_grad_norm: bool = True,
        log_weight_hist: bool = True,
        log_val_metrics: bool = True,
        log_train_images: bool = True,
        log_val_images: bool = True,
        log_lr: bool = True,
        log_model_stats: bool = True,
        grad_norm_interval: int = 100,
        weight_hist_interval: int = 1,
        train_image_interval: int = 500,
        val_image_interval: int = 10,  # 기본값 1 -> 10으로 증가
        max_layers_to_log: int = 20,
        max_train_images: int = 4,  # 저장할 training 이미지 수 제한
        kpt_thr: float = 0.3,
        backend_args: Optional[dict] = None,
    ):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.log_grad_norm = log_grad_norm
        self.log_weight_hist = log_weight_hist
        self.log_val_metrics = log_val_metrics
        self.log_train_images = log_train_images
        self.log_val_images = log_val_images
        self.log_lr = log_lr
        self.log_model_stats = log_model_stats
        self.grad_norm_interval = grad_norm_interval
        self.weight_hist_interval = weight_hist_interval
        self.train_image_interval = train_image_interval
        self.val_image_interval = val_image_interval
        self.max_layers_to_log = max_layers_to_log
        self.max_train_images = max_train_images
        self.kpt_thr = kpt_thr
        self.backend_args = backend_args

        # TensorBoard writer will be initialized lazily
        self._tb_writer = None
        self._tb_writer_initialized = False

        # Store first batch data for consistent visualization (메모리 절약을 위해 작은 subset만)
        self._first_batch_data = None
        self._first_batch_saved = False

    def _get_model(self, runner: Runner) -> torch.nn.Module:
        """Get the actual model, unwrapping DDP/FSDP if necessary.

        Args:
            runner: The runner instance.

        Returns:
            The unwrapped model.
        """
        model = runner.model
        # Unwrap DDP/FSDP wrapper
        if hasattr(model, 'module'):
            return model.module
        return model

    def _get_tensorboard_writer(self, runner: Runner):
        """Get or initialize TensorBoard writer."""
        if self._tb_writer_initialized:
            return self._tb_writer

        if SummaryWriter is None:
            print_log(
                'TensorBoard is not installed. Please install it with: '
                'pip install tensorboard',
                logger='current',
                level=logging.WARNING
            )
            self._tb_writer_initialized = True
            return None

        # Try to get TensorBoard writer from vis_backends
        for backend_name, backend in self._visualizer._vis_backends.items():
            if 'tensorboard' in backend_name.lower() or 'TensorboardVisBackend' in str(type(backend)):
                # Try to get writer from backend
                if hasattr(backend, 'writer'):
                    self._tb_writer = backend.writer
                    self._tb_writer_initialized = True
                    print_log(
                        f'EnhancedTensorBoardHook: Found TensorBoard writer from {backend_name}',
                        logger='current',
                        level=logging.INFO
                    )
                    return self._tb_writer
                elif hasattr(backend, '_writer'):
                    self._tb_writer = backend._writer
                    self._tb_writer_initialized = True
                    print_log(
                        f'EnhancedTensorBoardHook: Found TensorBoard writer from {backend_name}',
                        logger='current',
                        level=logging.INFO
                    )
                    return self._tb_writer

        # If not found, try to create one from work_dir
        if hasattr(runner, 'work_dir'):
            try:
                log_dir = os.path.join(runner.work_dir, runner.timestamp)
                self._tb_writer = SummaryWriter(log_dir=log_dir)
                self._tb_writer_initialized = True
                print_log(
                    f'EnhancedTensorBoardHook: Created TensorBoard writer at {log_dir}',
                    logger='current',
                    level=logging.INFO
                )
                return self._tb_writer
            except Exception as e:
                print_log(
                    f'Failed to create TensorBoard writer: {e}',
                    logger='current',
                    level=logging.WARNING
                )

        self._tb_writer_initialized = True
        print_log(
            'EnhancedTensorBoardHook: No TensorBoard writer found. '
            'Please add TensorboardVisBackend to your config.',
            logger='current',
            level=logging.WARNING
        )
        return None

    @master_only
    def _log_gradient_norms(self, runner: Runner) -> None:
        """Log gradient norms for each layer and global average.

        Optimized to minimize GPU-CPU synchronization.
        """
        if not self.log_grad_norm:
            return

        model = self._get_model(runner)  # DDP-safe
        if not isinstance(model, torch.nn.Module):
            return

        # Collect all gradient norms as tensors first (GPU에서 계산)
        grad_norms = []
        param_info = []  # (name, norm_tensor) pairs

        for name, param in model.named_parameters():
            if param.grad is not None:
                norm = param.grad.data.norm(2)
                grad_norms.append(norm)

                # Store for individual logging (limited)
                if len(param_info) < self.max_layers_to_log:
                    # Simplify layer name for readability
                    layer_name = name.replace('.', '/')
                    param_info.append((layer_name, norm))

        if not grad_norms:
            return

        # Calculate total norm on GPU
        grad_norms_tensor = torch.stack(grad_norms)
        total_norm = grad_norms_tensor.norm(2)

        # Single GPU->CPU transfer for total norm
        total_norm_value = total_norm.item()

        # Get TensorBoard writer
        tb_writer = self._get_tensorboard_writer(runner)
        if tb_writer is None:
            return

        # Log global average gradient norm
        tb_writer.add_scalar(
            'train/gradient_norm/global',
            total_norm_value,
            runner.iter
        )

        # Log individual layer gradient norms (한 번에 CPU로 이동)
        for layer_name, norm_tensor in param_info:
            tb_writer.add_scalar(
                f'train/gradient_norm/layers/{layer_name}',
                norm_tensor.item(),  # 개별 변환
                runner.iter
            )

    @master_only
    def _log_weight_histograms(self, runner: Runner) -> None:
        """Log weight histograms for each layer."""
        if not self.log_weight_hist:
            return

        model = self._get_model(runner)  # DDP-safe
        if not isinstance(model, torch.nn.Module):
            return

        # Get TensorBoard writer
        tb_writer = self._get_tensorboard_writer(runner)
        if tb_writer is None:
            return

        layer_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.data.numel() > 0:
                # Simplify layer name for readability
                layer_name = name.replace('.', '/')

                # Log weight histogram - detach and move to CPU
                try:
                    param_data = param.data.detach().cpu()

                    # Sample if too large to avoid memory issues
                    if param_data.numel() > 100000:
                        indices = torch.randperm(param_data.numel())[:100000]
                        param_data = param_data.flatten()[indices]

                    tb_writer.add_histogram(
                        f'weights/{layer_name}',
                        param_data,
                        runner.epoch
                    )
                except Exception as e:
                    print_log(
                        f'Failed to log histogram for {layer_name}: {e}',
                        logger='current',
                        level=logging.DEBUG
                    )

                layer_count += 1
                if layer_count >= self.max_layers_to_log:
                    break

    @master_only
    def _log_validation_metrics(self, runner: Runner,
                                metrics: Optional[Dict[str, float]] = None) -> None:
        """Log validation metrics to TensorBoard."""
        if not self.log_val_metrics:
            return

        if metrics is None or not metrics:
            return

        # Get TensorBoard writer
        tb_writer = self._get_tensorboard_writer(runner)
        if tb_writer is None:
            return

        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                # Log validation metrics
                tb_writer.add_scalar(
                    f'val/{metric_name}',
                    metric_value,
                    runner.epoch
                )

    def _denormalize_image(self, img_tensor: torch.Tensor, mean: tuple, std: tuple) -> np.ndarray:
        """Denormalize image tensor to numpy array.

        Vectorized implementation for better performance.

        Args:
            img_tensor: Image tensor in CHW format, normalized.
            mean: Mean values for denormalization.
            std: Std values for denormalization.

        Returns:
            Denormalized image as numpy array in HWC format (uint8).
        """
        # Vectorized denormalization (더 빠름)
        device = img_tensor.device
        mean_tensor = torch.tensor(mean, device=device, dtype=img_tensor.dtype).view(3, 1, 1)
        std_tensor = torch.tensor(std, device=device, dtype=img_tensor.dtype).view(3, 1, 1)

        # Denormalize: x = x * std + mean
        img = img_tensor * std_tensor + mean_tensor

        # Clamp to [0, 1]
        img = torch.clamp(img, 0, 1)

        # Convert to numpy and change from CHW to HWC
        img_np = img.permute(1, 2, 0).cpu().numpy()

        # Convert to uint8
        img_np = (img_np * 255).astype(np.uint8)
        return img_np

    def _create_image_grid(self, images: list, nrow: int = 4, padding: int = 2) -> np.ndarray:
        """Create a grid of images."""
        if not images:
            return None

        # Convert all images to same size (resize to smallest)
        min_h = min(img.shape[0] for img in images)
        min_w = min(img.shape[1] for img in images)
        resized_images = []
        for img in images:
            if img.shape[0] != min_h or img.shape[1] != min_w:
                img = mmcv.imresize(img, (min_w, min_h))
            resized_images.append(img)

        # Convert to tensors for make_grid
        if make_grid is not None:
            tensors = []
            for img in resized_images:
                # Convert HWC to CHW and normalize to [0, 1]
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                tensors.append(img_tensor)

            grid_tensor = make_grid(tensors, nrow=nrow, padding=padding, pad_value=1.0)
            # Convert back to numpy HWC
            grid_np = grid_tensor.permute(1, 2, 0).cpu().numpy()
            grid_np = (grid_np * 255).astype(np.uint8)
            return grid_np
        else:
            # Fallback: manual grid creation
            n_images = len(resized_images)
            ncols = min(nrow, n_images)
            nrows = (n_images + ncols - 1) // ncols

            grid_h = nrows * min_h + (nrows - 1) * padding
            grid_w = ncols * min_w + (ncols - 1) * padding
            grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

            for idx, img in enumerate(resized_images):
                row = idx // ncols
                col = idx % ncols
                y1 = row * (min_h + padding)
                y2 = y1 + min_h
                x1 = col * (min_w + padding)
                x2 = x1 + min_w
                grid[y1:y2, x1:x2] = img

            return grid

    @master_only
    def _log_training_image(self, runner: Runner, data_batch: dict,
                            outputs: Sequence[PoseDataSample]) -> None:
        """Log training image with predictions in grid format."""
        if not self.log_train_images:
            return

        try:
            # Use first batch data for consistent visualization
            if self._first_batch_data is None:
                return

            inputs = self._first_batch_data['inputs']
            data_samples = self._first_batch_data['data_samples']

            # Get model's normalization parameters
            model = self._get_model(runner)
            if hasattr(model, 'data_preprocessor'):
                preprocessor = model.data_preprocessor
                mean = tuple(preprocessor.mean) if hasattr(preprocessor, 'mean') else (123.675, 116.28, 103.53)
                std = tuple(preprocessor.std) if hasattr(preprocessor, 'std') else (58.395, 57.12, 57.375)
            else:
                mean = (123.675, 116.28, 103.53)
                std = (58.395, 57.12, 57.375)

            # Auto-detect if mean/std are in 0-255 or 0-1 range
            # Heuristic: if mean > 2, assume 0-255 range
            if any(m > 2.0 for m in mean):
                mean_norm = tuple(m / 255.0 for m in mean)
                std_norm = tuple(s / 255.0 for s in std)
            else:
                mean_norm = mean
                std_norm = std

            # Try to use current outputs if available and matching
            pred_data_samples = None
            if outputs is not None:
                if isinstance(outputs, (list, tuple)) and len(outputs) >= len(data_samples):
                    pred_data_samples = outputs[:len(data_samples)]
                elif isinstance(outputs, dict):
                    if 'data_samples' in outputs and len(outputs['data_samples']) >= len(data_samples):
                        pred_data_samples = outputs['data_samples'][:len(data_samples)]

            # If outputs don't match, re-run inference on stored first batch
            if pred_data_samples is None and isinstance(inputs, torch.Tensor) and inputs.dim() == 4:
                try:
                    # Re-run inference with current model
                    runner.model.eval()
                    with torch.no_grad():
                        # Move inputs to model device
                        device = next(runner.model.parameters()).device
                        inputs_device = inputs.to(device)

                        inference_batch = {
                            'inputs': inputs_device,
                            'data_samples': data_samples
                        }
                        results = runner.model.test_step(inference_batch)
                        if results and len(results) >= len(data_samples):
                            pred_data_samples = results[:len(data_samples)]
                    runner.model.train()
                except Exception as e:
                    print_log(
                        f'Failed to run inference for training images: {e}',
                        logger='current',
                        level=logging.DEBUG
                    )

            # Set dataset meta
            if hasattr(runner, 'val_evaluator') and runner.val_evaluator is not None:
                if hasattr(runner.val_evaluator, 'dataset_meta'):
                    self._visualizer.set_dataset_meta(
                        runner.val_evaluator.dataset_meta)

            # Process images
            gt_images = []
            pred_images = []

            batch_size = len(data_samples)

            for i in range(batch_size):
                # Denormalize image
                if isinstance(inputs, torch.Tensor):
                    img_tensor = inputs[i] if inputs.dim() == 4 else inputs
                    img = self._denormalize_image(img_tensor, mean_norm, std_norm)
                else:
                    # Fallback: load from file
                    if hasattr(data_samples[i], 'metainfo') and 'img_path' in data_samples[i].metainfo:
                        img_path = data_samples[i].metainfo['img_path']
                        img_bytes = fileio.get(img_path, backend_args=self.backend_args)
                        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                    else:
                        continue

                # Create GT image
                gt_sample = PoseDataSample()
                if hasattr(data_samples[i], 'gt_instances'):
                    gt_sample.gt_instances = data_samples[i].gt_instances
                gt_img = self._visualizer.add_datasample(
                    f'train_gt_{i}',
                    img.copy(),
                    gt_sample,
                    draw_gt=True,
                    draw_pred=False,
                    draw_bbox=True,
                    kpt_thr=self.kpt_thr,
                    step=runner.iter)
                if gt_img is not None:
                    gt_images.append(gt_img)

                # Create prediction image
                if pred_data_samples and i < len(pred_data_samples):
                    pred_sample = PoseDataSample()
                    if hasattr(pred_data_samples[i], 'pred_instances'):
                        pred_sample.pred_instances = pred_data_samples[i].pred_instances
                    pred_img = self._visualizer.add_datasample(
                        f'train_pred_{i}',
                        img.copy(),
                        pred_sample,
                        draw_gt=False,
                        draw_pred=True,
                        draw_bbox=True,
                        kpt_thr=self.kpt_thr,
                        step=runner.iter)
                    if pred_img is not None:
                        pred_images.append(pred_img)

            # Create grids
            tb_writer = self._get_tensorboard_writer(runner)
            if tb_writer is not None and (gt_images or pred_images):
                # GT grid
                if gt_images:
                    gt_grid = self._create_image_grid(gt_images, nrow=2)
                    if gt_grid is not None:
                        gt_labeled = self._add_text_label(gt_grid, "Ground Truth")
                        tb_writer.add_image('train/images_gt', gt_labeled, runner.iter, dataformats='HWC')

                # Prediction grid
                if pred_images:
                    pred_grid = self._create_image_grid(pred_images, nrow=2)
                    if pred_grid is not None:
                        pred_labeled = self._add_text_label(pred_grid, "Prediction")
                        tb_writer.add_image('train/images_pred', pred_labeled, runner.iter, dataformats='HWC')

        except Exception as e:
            print_log(
                f'Failed to log training images: {e}',
                logger='current',
                level=logging.DEBUG
            )

    def _add_text_label(self, img: np.ndarray, label: str) -> np.ndarray:
        """Add text label to image."""
        img_labeled = img.copy()
        # Add text at top
        cv2.putText(img_labeled, label, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img_labeled

    @master_only
    def _log_validation_image(self, runner: Runner, batch_idx: int,
                              data_batch: dict,
                              outputs: Sequence[PoseDataSample]) -> None:
        """Log validation image with predictions in grid format."""
        if not self.log_val_images:
            return

        try:
            # Only log from first batch for consistency
            if batch_idx != 0:
                return

            if 'data_samples' not in data_batch or len(data_batch['data_samples']) == 0:
                return

            # Get inputs if available
            inputs = None
            if 'inputs' in data_batch:
                inputs = data_batch['inputs']

            data_samples = data_batch['data_samples']

            # Get model's normalization parameters
            model = self._get_model(runner)
            if hasattr(model, 'data_preprocessor'):
                preprocessor = model.data_preprocessor
                mean = tuple(preprocessor.mean) if hasattr(preprocessor, 'mean') else (123.675, 116.28, 103.53)
                std = tuple(preprocessor.std) if hasattr(preprocessor, 'std') else (58.395, 57.12, 57.375)
            else:
                mean = (123.675, 116.28, 103.53)
                std = (58.395, 57.12, 57.375)

            # Auto-detect range
            if any(m > 2.0 for m in mean):
                mean_norm = tuple(m / 255.0 for m in mean)
                std_norm = tuple(s / 255.0 for s in std)
            else:
                mean_norm = mean
                std_norm = std

            # Set dataset meta
            if hasattr(runner, 'val_evaluator') and runner.val_evaluator is not None:
                if hasattr(runner.val_evaluator, 'dataset_meta'):
                    self._visualizer.set_dataset_meta(
                        runner.val_evaluator.dataset_meta)

            # Process batch images
            gt_images = []
            pred_images = []

            batch_size = min(len(data_samples), 8)  # Limit to 8 images

            for i in range(batch_size):
                # Get image
                img = None
                if inputs is not None and isinstance(inputs, torch.Tensor):
                    if inputs.dim() == 4 and i < inputs.shape[0]:
                        img_tensor = inputs[i]
                        img = self._denormalize_image(img_tensor, mean_norm, std_norm)

                if img is None:
                    # Load from file
                    if hasattr(data_samples[i], 'metainfo') and 'img_path' in data_samples[i].metainfo:
                        img_path = data_samples[i].metainfo['img_path']
                        img_bytes = fileio.get(img_path, backend_args=self.backend_args)
                        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                    else:
                        continue

                # Create GT image
                gt_sample = PoseDataSample()
                if hasattr(data_samples[i], 'gt_instances'):
                    gt_sample.gt_instances = data_samples[i].gt_instances
                gt_img = self._visualizer.add_datasample(
                    f'val_gt_{i}',
                    img.copy(),
                    gt_sample,
                    draw_gt=True,
                    draw_pred=False,
                    draw_bbox=True,
                    kpt_thr=self.kpt_thr,
                    step=runner.epoch)
                if gt_img is not None:
                    gt_images.append(gt_img)

                # Create prediction image
                if outputs and i < len(outputs):
                    pred_sample = PoseDataSample()
                    if hasattr(outputs[i], 'pred_instances'):
                        pred_sample.pred_instances = outputs[i].pred_instances
                    pred_img = self._visualizer.add_datasample(
                        f'val_pred_{i}',
                        img.copy(),
                        pred_sample,
                        draw_gt=False,
                        draw_pred=True,
                        draw_bbox=True,
                        kpt_thr=self.kpt_thr,
                        step=runner.epoch)
                    if pred_img is not None:
                        pred_images.append(pred_img)

            # Create grids
            tb_writer = self._get_tensorboard_writer(runner)
            if tb_writer is not None and (gt_images or pred_images):
                # GT grid
                if gt_images:
                    gt_grid = self._create_image_grid(gt_images, nrow=4)
                    if gt_grid is not None:
                        gt_labeled = self._add_text_label(gt_grid, "Ground Truth")
                        tb_writer.add_image('val/images_gt', gt_labeled, runner.epoch, dataformats='HWC')

                # Prediction grid
                if pred_images:
                    pred_grid = self._create_image_grid(pred_images, nrow=4)
                    if pred_grid is not None:
                        pred_labeled = self._add_text_label(pred_grid, "Prediction")
                        tb_writer.add_image('val/images_pred', pred_labeled, runner.epoch, dataformats='HWC')

        except Exception as e:
            print_log(
                f'Failed to log validation images: {e}',
                logger='current',
                level=logging.DEBUG
            )

    @master_only
    def _log_learning_rate(self, runner: Runner) -> None:
        """Log current learning rate."""
        if not self.log_lr:
            return

        if hasattr(runner, 'optim_wrapper') and runner.optim_wrapper is not None:
            if hasattr(runner.optim_wrapper, 'param_groups'):
                # Get TensorBoard writer
                tb_writer = self._get_tensorboard_writer(runner)
                if tb_writer is None:
                    return

                for i, param_group in enumerate(runner.optim_wrapper.param_groups):
                    lr = param_group.get('lr', None)
                    if lr is not None:
                        group_name = f'group_{i}' if len(runner.optim_wrapper.param_groups) > 1 else 'lr'
                        tb_writer.add_scalar(
                            f'train/learning_rate/{group_name}',
                            lr,
                            runner.iter)

    @master_only
    def _log_model_statistics(self, runner: Runner) -> None:
        """Log model parameter statistics."""
        if not self.log_model_stats:
            return

        model = self._get_model(runner)  # DDP-safe
        if not isinstance(model, torch.nn.Module):
            return

        total_params = 0
        trainable_params = 0
        total_size = 0

        for param in model.parameters():
            num_params = param.numel()
            total_params += num_params
            total_size += num_params * param.element_size()
            if param.requires_grad:
                trainable_params += num_params

        # Get TensorBoard writer
        tb_writer = self._get_tensorboard_writer(runner)
        if tb_writer is None:
            return

        # Log parameter counts
        tb_writer.add_scalar(
            'model/total_parameters',
            total_params,
            runner.epoch)
        tb_writer.add_scalar(
            'model/trainable_parameters',
            trainable_params,
            runner.epoch)
        tb_writer.add_scalar(
            'model/model_size_mb',
            total_size / (1024 ** 2),  # Convert to MB
            runner.epoch)

    def after_train_iter(self, runner: Runner, batch_idx: int,
                         data_batch: dict = None,
                         outputs: dict = None) -> None:
        """Called after each training iteration."""
        # Save first batch data for consistent visualization (메모리 절약)
        if (self.log_train_images and
            not self._first_batch_saved and
            batch_idx == 0 and
            data_batch is not None and
            'inputs' in data_batch and
            'data_samples' in data_batch):
            try:
                inputs = data_batch['inputs']
                if isinstance(inputs, torch.Tensor) and inputs.dim() == 4:
                    # 메모리 절약: max_train_images개만 저장
                    num_to_save = min(self.max_train_images, inputs.shape[0])
                    self._first_batch_data = {
                        'inputs': inputs[:num_to_save].detach().cpu().clone(),
                        'data_samples': data_batch['data_samples'][:num_to_save]
                    }
                    self._first_batch_saved = True
            except Exception as e:
                print_log(
                    f'Failed to save first batch data: {e}',
                    logger='current',
                    level=logging.DEBUG
                )

        # Log gradient norms (interval-based)
        if self.log_grad_norm and runner.iter % self.grad_norm_interval == 0:
            self._log_gradient_norms(runner)

        # Log learning rate (interval-based, 성능 개선)
        if self.log_lr and runner.iter % self.grad_norm_interval == 0:
            self._log_learning_rate(runner)

        # Log training images (interval-based)
        if (self.log_train_images and
            runner.iter % self.train_image_interval == 0):
            # Extract PoseDataSample from outputs
            pred_data_samples = None
            if outputs is not None:
                if isinstance(outputs, dict):
                    if 'data_samples' in outputs:
                        pred_data_samples = outputs['data_samples']
                    elif 'predictions' in outputs:
                        pred_data_samples = outputs['predictions']
                elif isinstance(outputs, (list, tuple)):
                    pred_data_samples = outputs

            self._log_training_image(runner, data_batch, pred_data_samples)

    def after_train_epoch(self, runner: Runner) -> None:
        """Called after each training epoch."""
        # Log weight histograms (epoch-based interval)
        if self.log_weight_hist and runner.epoch % self.weight_hist_interval == 0:
            self._log_weight_histograms(runner)

        # Log model statistics
        if self.log_model_stats:
            self._log_model_statistics(runner)

    def after_val_iter(self, runner: Runner, batch_idx: int,
                       data_batch: dict = None,
                       outputs: Sequence[PoseDataSample] = None) -> None:
        """Called after each validation iteration."""
        # Log validation images from first batch only
        if self.log_val_images and batch_idx % self.val_image_interval == 0:
            self._log_validation_image(runner, batch_idx, data_batch, outputs)

    def after_val_epoch(self, runner: Runner, metrics: Optional[Dict] = None) -> None:
        """Called after validation epoch.

        Args:
            runner: The runner instance.
            metrics: Validation metrics dictionary. If None, will try to retrieve
                from message_hub.
        """
        if not self.log_val_metrics:
            return

        # Use provided metrics or get from message hub
        if metrics is None:
            try:
                # Try to get metrics from message hub (MMEngine 1.x 방식)
                if hasattr(runner, 'message_hub'):
                    metrics = runner.message_hub.get_scalar('val').data
            except Exception as e:
                print_log(
                    f'Failed to retrieve validation metrics from message_hub: {e}',
                    logger='current',
                    level=logging.DEBUG
                )
                return

        # Log metrics if available
        if metrics:
            self._log_validation_metrics(runner, metrics)
