# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from typing import Optional, Tuple

from mmpose.registry import MODELS
from .base_backbone import BaseBackbone


@MODELS.register_module()
class DINOv3(BaseBackbone):
    """DINOv3 backbone for vision tasks.
    
    DINOv3 is a foundation model for vision producing high-quality dense features
    that achieve outstanding performance on various vision tasks without fine-tuning.
    
    Paper: DINOv3: Scaling Self-Supervised Vision Foundation Models
    Link: https://arxiv.org/abs/2508.10104
    
    Args:
        pretrained (str): Model name from Hugging Face Hub.
            Options: 
            - 'facebook/dinov3-vits16-pretrain-lvd1689m' (embed_dim=384, patch_size=16)
            - 'facebook/dinov3-vitb16-pretrain-lvd1689m' (embed_dim=768, patch_size=16)
            - 'facebook/dinov3-vitl16-pretrain-lvd1689m' (embed_dim=1024, patch_size=16)
            Default: 'facebook/dinov3-vitl16-pretrain-lvd1689m'
        frozen (bool): If True, freeze the backbone weights during training.
            Default: True
        output_cls_token (bool): If True, also return the CLS token.
            Default: False
        lora_config (dict, optional): LoRA configuration. If None, LoRA is not applied.
            Default: None
            Example:
                lora_config=dict(
                    r=16,                    # LoRA rank
                    lora_alpha=32,           # LoRA alpha (scaling factor)
                    target_modules=['qkv', 'proj'],  # Modules to apply LoRA
                    lora_dropout=0.1,       # LoRA dropout
                    bias='none',            # Bias handling: 'none', 'all', 'lora_only'
                )
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
            
    Example:
        >>> from mmpose.models import DINOv3
        >>> import torch
        >>> model = DINOv3(pretrained='facebook/dinov3-vitl16-pretrain-lvd1689m', frozen=True)
        >>> x = torch.randn(2, 3, 518, 518)
        >>> features = model(x)
        >>> print(features[0].shape)
        torch.Size([2, 1024, 32, 32])
    """

    def __init__(
        self,
        pretrained: str = 'facebook/dinov3-vitl16-pretrain-lvd1689m',
        frozen: bool = True,
        output_cls_token: bool = False,
        lora_config: Optional[dict] = None,
        init_cfg: Optional[dict] = None
    ):
        super().__init__(init_cfg=init_cfg)
        
        self.pretrained = pretrained
        self.frozen = frozen
        self.output_cls_token = output_cls_token
        self.lora_config = lora_config
        
        # Import transformers library
        try:
            from transformers import AutoModel, AutoImageProcessor
        except ImportError:
            raise ImportError(
                'Please install transformers: pip install transformers'
            )
        
        # Load pre-trained DINOv3 model
        self.model = AutoModel.from_pretrained(pretrained)
        self.processor = AutoImageProcessor.from_pretrained(pretrained)
        
        # Apply LoRA if config is provided
        if lora_config is not None:
            try:
                from peft import LoraConfig, get_peft_model, TaskType
            except ImportError:
                raise ImportError(
                    'Please install peft library for LoRA support: pip install peft'
                )
            
            # Default LoRA config values
            lora_r = lora_config.get('r', 16)
            lora_alpha = lora_config.get('lora_alpha', 32)
            target_modules = lora_config.get('target_modules', ['qkv', 'proj'])
            lora_dropout = lora_config.get('lora_dropout', 0.1)
            bias = lora_config.get('bias', 'none')
            
            # Create LoRA config
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias=bias,
            )
            
            # Apply LoRA to the model
            self.model = get_peft_model(self.model, peft_config)
            
            # Print trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"LoRA 적용 완료:")
            print(f"  - Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
            print(f"  - Total parameters: {total_params:,}")
            print(f"  - LoRA rank (r): {lora_r}")
            print(f"  - LoRA alpha: {lora_alpha}")
            print(f"  - Target modules: {target_modules}")
        
        # Get model configuration
        if hasattr(self.model, 'config'):
            config = self.model.config
        else:
            # If LoRA is applied, get config from base model
            if hasattr(self.model, 'get_base_model'):
                config = self.model.get_base_model().config
            else:
                config = self.model.config
        
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.num_registers = getattr(config, 'num_register_tokens', 4)
        
        # Freeze backbone if specified (LoRA params will still be trainable)
        if self.frozen:
            self._freeze_backbone()
            
    def _freeze_backbone(self):
        """Freeze backbone parameters, but keep LoRA parameters trainable."""
        if self.lora_config is not None:
            # If LoRA is applied, only freeze non-LoRA parameters
            for name, param in self.model.named_parameters():
                if 'lora' not in name.lower():
                    param.requires_grad = False
        else:
            # If no LoRA, freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False
            
    def init_weights(self):
        """Initialize weights.
        
        Since we use pre-trained weights from Hugging Face,
        we don't need custom initialization.
        LoRA weights are initialized by peft library.
        """
        # Pre-trained weights are already loaded in __init__
        # LoRA weights are initialized by peft
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
                - (optional) cls_token: CLS token with shape (B, embed_dim) if 
                  output_cls_token=True
        """
        batch_size, _, input_h, input_w = x.shape
        
        # Forward through DINOv3 (with LoRA if applied)
        # DINOv3's processor handles input normalization automatically
        # It expects images in [0, 255] range and will normalize internally
        with torch.set_grad_enabled(not self.frozen or self.lora_config is not None):
            outputs = self.model(pixel_values=x)
        
        # Extract features
        # last_hidden_state: (B, num_tokens, embed_dim)
        # num_tokens = 1 (CLS) + num_registers + H'*W' (patches)
        last_hidden_state = outputs.last_hidden_state
        
        # Separate CLS token, register tokens, and patch tokens
        # Token order: [CLS, register1, register2, ..., patch1, patch2, ...]
        cls_token = last_hidden_state[:, 0]  # (B, embed_dim)
        patch_tokens = last_hidden_state[:, 1 + self.num_registers:]  # (B, H'*W', embed_dim)
        
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
        # But LoRA parameters should remain trainable
        if self.frozen:
            if self.lora_config is not None:
                # With LoRA: set base model to eval, but keep LoRA trainable
                if hasattr(self.model, 'get_base_model'):
                    self.model.get_base_model().eval()
                # Ensure LoRA params are trainable
                for name, param in self.model.named_parameters():
                    if 'lora' in name.lower():
                        param.requires_grad = True
            else:
                # Without LoRA: freeze everything
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False

