# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from typing import Optional, Tuple

from mmpose.registry import MODELS
from .base_backbone import BaseBackbone


@MODELS.register_module()
class CLIPViT(BaseBackbone):
    """CLIP Vision Transformer backbone for vision tasks.
    
    CLIP (Contrastive Language-Image Pre-training) is a neural network trained
    on a variety of (image, text) pairs to learn visual concepts from natural 
    language supervision.
    
    Paper: Learning Transferable Visual Models From Natural Language Supervision
    Link: https://arxiv.org/abs/2103.00020
    
    Args:
        pretrained (str): Model name from Hugging Face Hub.
            Options:
            - 'openai/clip-vit-base-patch32' (embed_dim=768, patch_size=32)
            - 'openai/clip-vit-base-patch16' (embed_dim=768, patch_size=16)
            - 'openai/clip-vit-large-patch14' (embed_dim=1024, patch_size=14)
            Default: 'openai/clip-vit-base-patch32'
        frozen (bool): If True, freeze the backbone weights during training.
            Default: True
        output_cls_token (bool): If True, also return the CLS token (pooled output).
            Default: False
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
            
    Example:
        >>> from mmpose.models import CLIPViT
        >>> import torch
        >>> model = CLIPViT(pretrained='openai/clip-vit-base-patch16', frozen=True)
        >>> x = torch.randn(2, 3, 224, 224)
        >>> features = model(x)
        >>> print(features[0].shape)
        torch.Size([2, 768, 14, 14])
    """

    def __init__(
        self,
        pretrained: str = 'openai/clip-vit-base-patch16',
        frozen: bool = True,
        output_cls_token: bool = False,
        init_cfg: Optional[dict] = None
    ):
        super().__init__(init_cfg=init_cfg)
        
        self.pretrained = pretrained
        self.frozen = frozen
        self.output_cls_token = output_cls_token
        
        # Import transformers library
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            raise ImportError(
                'Please install transformers: pip install transformers'
            )
        
        # Load pre-trained CLIP model
        self.model = CLIPModel.from_pretrained(pretrained)
        self.processor = CLIPProcessor.from_pretrained(pretrained)
        
        # Get vision encoder configuration
        vision_config = self.model.vision_model.config
        self.embed_dim = vision_config.hidden_size
        self.patch_size = vision_config.patch_size
        
        # Freeze backbone if specified
        if self.frozen:
            self._freeze_backbone()
            
    def _freeze_backbone(self):
        """Freeze all parameters in the vision backbone."""
        for param in self.model.vision_model.parameters():
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
                Expected to be normalized by PoseDataPreprocessor
            
        Returns:
            tuple[torch.Tensor]: 
                - features: Patch embeddings with shape (B, embed_dim, H', W')
                  where H' = H // patch_size, W' = W // patch_size
                - (optional) cls_token: Pooled output with shape (B, embed_dim) if 
                  output_cls_token=True
        """
        batch_size, _, input_h, input_w = x.shape
        
        # Forward through CLIP vision encoder
        # CLIP's processor handles input normalization automatically
        # It expects images in [0, 255] range and will normalize internally
        with torch.set_grad_enabled(not self.frozen):
            vision_outputs = self.model.vision_model(pixel_values=x, output_hidden_states=True)
        
        # Extract features
        # last_hidden_state: (B, num_patches + 1, embed_dim)
        # First token is CLS token, rest are patch tokens
        last_hidden_state = vision_outputs.last_hidden_state
        
        # Separate CLS token and patch tokens
        cls_token = last_hidden_state[:, 0]  # (B, embed_dim)
        patch_tokens = last_hidden_state[:, 1:]  # (B, num_patches, embed_dim)
        
        # Reshape patch tokens to spatial format
        h_out, w_out = self._calculate_output_size((input_h, input_w))
        patch_embeddings = patch_tokens.transpose(1, 2).reshape(
            batch_size, self.embed_dim, h_out, w_out
        )
        
        # Clone to ensure normal tensors for autograd
        patch_embeddings = patch_embeddings.clone()
        cls_token = cls_token.clone()
        
        # Return format: tuple of tensors (MMPose convention)
        if self.output_cls_token:
            return (patch_embeddings, cls_token)
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
            self.model.vision_model.eval()
            for param in self.model.vision_model.parameters():
                param.requires_grad = False

