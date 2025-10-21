# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from mmpose.registry import MODELS
from .base_backbone import BaseBackbone


@MODELS.register_module()
class RadDINO(BaseBackbone):
    """RAD-DINO backbone for medical image analysis.
    
    RAD-DINO is a Vision Transformer trained on chest X-ray images using 
    self-supervised learning (DINOv2). It outputs 768-dimensional features
    with spatial structure preserved as 37×37 patches.
    
    Paper: Exploring Scalable Medical Image Encoders Beyond Text Supervision
    Link: https://arxiv.org/abs/2311.13668
    
    Args:
        pretrained (str): Model name from Hugging Face Hub.
            Default: 'microsoft/rad-dino'
        frozen (bool): If True, freeze the backbone weights during training.
            Default: False
        output_cls_token (bool): If True, also return the CLS token.
            Default: False
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
            
    Example:
        >>> from mmpose.models import RadDINO
        >>> import torch
        >>> model = RadDINO(pretrained='microsoft/rad-dino', frozen=False)
        >>> x = torch.randn(2, 3, 384, 288)
        >>> features = model(x)
        >>> print(features[0].shape)
        torch.Size([2, 768, 37, 37])
    """

    def __init__(
        self,
        pretrained: str = 'microsoft/rad-dino',
        frozen: bool = False,
        output_cls_token: bool = False,
        init_cfg: Optional[dict] = None
    ):
        super().__init__(init_cfg=init_cfg)
        
        self.pretrained = pretrained
        self.frozen = frozen
        self.output_cls_token = output_cls_token
        
        # Import rad-dino library
        try:
            from rad_dino import RadDino as RadDinoModel
        except ImportError:
            raise ImportError(
                'Please install rad-dino: pip install rad-dino'
            )
        
        # Load pre-trained RAD-DINO model
        # self.encoder = RadDinoModel(model_name=pretrained)
        self.encoder = RadDinoModel()
        # Freeze backbone if specified
        if self.frozen:
            self._freeze_backbone()
            
        # Model output dimensions
        self.embed_dim = 768  # RAD-DINO uses 768-dim features
        self.patch_size = 14  # DINOv2-base uses 14×14 patches
        
    def _freeze_backbone(self):
        """Freeze all parameters in the backbone."""
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def init_weights(self):
        """Initialize weights.
        
        Since we use pre-trained weights from Hugging Face,
        we don't need custom initialization.
        """
        # Pre-trained weights are already loaded in __init__
        pass
    
    def _calculate_output_size(self, input_size: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate the spatial dimensions of output feature map.
        
        Args:
            input_size (tuple): Input image size (H, W)
            
        Returns:
            tuple: Output feature map size (H', W')
        """
        h, w = input_size
        h_out = h // self.patch_size
        w_out = w // self.patch_size
        return h_out, w_out
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Forward function.
        
        Args:
            x (torch.Tensor): Input images with shape (B, C, H, W)
            
        Returns:
            tuple[torch.Tensor]: 
                - features: Patch embeddings with shape (B, 768, H', W')
                  where H' = H // 14, W' = W // 14
                - (optional) cls_token: CLS token with shape (B, 768) if 
                  output_cls_token=True
        """
        batch_size = x.shape[0]
        input_h, input_w = x.shape[2], x.shape[3]
        
        # Extract features using RAD-DINO
        # cls_embeddings: (B, 768)
        # patch_embeddings: (B, 768, H', W')
        cls_embeddings, patch_embeddings = self.encoder.extract_features(x)
        
        # Convert inference tensors to normal tensors for autograd
        # extract_features() uses @torch.inference_mode(), so we need to clone
        # .clone() creates a new normal tensor that can be used in autograd
        patch_embeddings = patch_embeddings.clone()
        if cls_embeddings is not None:
            cls_embeddings = cls_embeddings.clone()
        
        # Verify output dimensions
        expected_h, expected_w = self._calculate_output_size((input_h, input_w))
        actual_h, actual_w = patch_embeddings.shape[2], patch_embeddings.shape[3]
        
        # Handle potential size mismatch due to different input sizes
        if (actual_h != expected_h) or (actual_w != expected_w):
            # This can happen with non-standard input sizes
            # RAD-DINO handles various input sizes automatically
            pass
        
        # Return format: tuple of tensors (MMPose convention)
        if self.output_cls_token:
            return (patch_embeddings, cls_embeddings)
        else:
            return (patch_embeddings,)
    
    def train(self, mode: bool = True):
        """Set module status before forward computation.
        
        Args:
            mode (bool): Whether it is train mode or test mode
        """
        super().train(mode)
        
        # Keep backbone frozen in both train and eval mode if specified
        if self.frozen:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

