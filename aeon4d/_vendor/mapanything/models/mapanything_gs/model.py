# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
MapAnything model class defined using UniCeption modules.
"""

import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from huggingface_hub import PyTorchModelHubMixin
import numpy as np
from einops import rearrange
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import cv2
import os

from aeon4d._vendor.mapanything.utils.geometry import (
    apply_log_to_norm,
    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
    get_rays_in_camera_frame,
    normalize_depth_using_non_zero_pixels,
    normalize_pose_translations,
    recover_pinhole_intrinsics_from_ray_directions,
    transform_pose_using_quats_and_trans_2_to_1,
    radius_to_sigma,
    dt_to_cov_t,
)
from aeon4d._vendor.mapanything.models.mapanything_gs.adaptors import build_pytorch_mlp
from aeon4d._vendor.mapanything.utils.easyvolcap_compat import multi_gather  # used by magic_filter(recalc_magic_pattern=True)
from aeon4d._vendor.uniception.models.encoders import (
    encoder_factory,
    EncoderGlobalRepInput,
    ViTEncoderInput,
    ViTEncoderNonImageInput,
)
from aeon4d._vendor.uniception.models.info_sharing.alternating_attention_transformer import (
    MultiViewAlternatingAttentionTransformer,
    MultiViewAlternatingAttentionTransformerIFR,
)
from aeon4d._vendor.uniception.models.info_sharing.base import MultiViewTransformerInput
from aeon4d._vendor.uniception.models.prediction_heads.adaptors import (
    CamTranslationPlusQuatsAdaptor,
    PointMapAdaptor,
    PointMapPlusRayDirectionsPlusDepthAdaptor,
    PointMapPlusRayDirectionsPlusDepthWithConfidenceAdaptor,
    PointMapPlusRayDirectionsPlusDepthWithConfidenceAndMaskAdaptor,
    PointMapPlusRayDirectionsPlusDepthWithMaskAdaptor,
    PointMapWithConfidenceAdaptor,
    PointMapWithConfidenceAndMaskAdaptor,
    PointMapWithMaskAdaptor,
    RayDirectionsPlusDepthAdaptor,
    RayDirectionsPlusDepthWithConfidenceAdaptor,
    RayDirectionsPlusDepthWithConfidenceAndMaskAdaptor,
    RayDirectionsPlusDepthWithMaskAdaptor,
    RayMapPlusDepthAdaptor,
    RayMapPlusDepthWithConfidenceAdaptor,
    RayMapPlusDepthWithConfidenceAndMaskAdaptor,
    RayMapPlusDepthWithMaskAdaptor,
    ScaleAdaptor,
)
from aeon4d._vendor.uniception.models.prediction_heads.base import (
    AdaptorInput,
    PredictionHeadInput,
    PredictionHeadLayeredInput,
    PredictionHeadTokenInput,
)
from aeon4d._vendor.uniception.models.prediction_heads.dpt import DPTFeature, DPTRegressionProcessor
from aeon4d._vendor.uniception.models.prediction_heads.linear import LinearFeature
from aeon4d._vendor.uniception.models.prediction_heads.mlp_head import MLPHead
from aeon4d._vendor.uniception.models.prediction_heads.pose_head import PoseHead

# Enable TF32 precision if supported (for GPU >= Ampere and PyTorch >= 1.12)
if hasattr(torch.backends.cuda, "matmul") and hasattr(
    torch.backends.cuda.matmul, "allow_tf32"
):
    torch.backends.cuda.matmul.allow_tf32 = True

def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # 1. 先计算平方和
    # 2. 在平方和内部加上 eps
    # 3. 最后再开根号
    # 这里的 eps 既防止了分母为0，也防止了 sqrt 求导溢出
    norm = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + eps)
    return x / norm


def build_pytorch_mlp(
    input_dim,
    hidden_dim,
    output_dim,
    act_layer=nn.GELU,
    depth=10,
    bias=False,
    use_weight_norm=False,
) -> nn.Sequential:
    """Build a PyTorch MLP with configurable depth and options.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output feature dimension
        act_layer: Activation layer class (default: nn.GELU)
        depth: Number of hidden layers (0 for direct linear projection)
        bias: Whether to use bias in linear layers
        use_weight_norm: Whether to apply weight normalization
    
    Returns:
        nn.Sequential: The constructed MLP
    """
    if depth == 0:
        mlp = [nn.Linear(input_dim, output_dim, bias=bias)]
    else:
        mlp = []
        mlp.append(nn.Linear(input_dim, hidden_dim, bias=bias))
        mlp.append(act_layer())
        for _ in range(depth - 1):
            mlp.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            mlp.append(act_layer())
        mlp.append(nn.Linear(hidden_dim, output_dim, bias=bias))

    if use_weight_norm:
        mlp = [
            torch.nn.utils.weight_norm(layer) if isinstance(layer, nn.Linear) else layer
            for layer in mlp
        ]
    mlp = nn.Sequential(*mlp)
    return mlp



class ImageInputMerger(nn.Module):
    """
    Simple convolutional network for processing input images.
    
    Structure:
        输入图像 (B, C, H, W)
            ↓
        Conv2d layers with ReLU (保持尺寸不变)
            ↓
        输出特征 (B, out_dim, H, W)
    """
    def __init__(
        self,
        in_chans: int = 3,
        hidden_dim: int = 64,
        out_dim: int = 256,
        num_layers: int = 4,
        use_residual: bool = True,
    ):
        """
        Args:
            in_chans: Number of input channels (default: 3 for RGB)
            hidden_dim: Hidden dimension for intermediate layers
            out_dim: Output feature dimension
            num_layers: Number of convolutional layers
            use_residual: Whether to use residual connections
        """
        super().__init__()
        self.use_residual = use_residual
        self.out_dim = out_dim
        
        layers = []
        
        # First layer: in_chans -> hidden_dim
        layers.append(nn.Conv2d(in_chans, hidden_dim, kernel_size=3, stride=1, padding=1))
        layers.append(nn.GELU())
        
        # Middle layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1))
            layers.append(nn.GELU())
        
        # Last layer: hidden_dim -> out_dim
        layers.append(nn.Conv2d(hidden_dim, out_dim, kernel_size=3, stride=1, padding=1))
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Optional: residual projection if in_chans != out_dim
        if use_residual and in_chans != out_dim:
            self.residual_proj = nn.Conv2d(in_chans, out_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.residual_proj = None
            
        # Layer norm at the end
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Output tensor of shape (B, out_dim, H, W)
        """
        out = self.conv_layers(x)
        
        # Apply residual connection if enabled
        if self.use_residual:
            if self.residual_proj is not None:
                out = out + self.residual_proj(x)
            elif x.shape[1] == self.out_dim:
                out = out + x
        
        # Apply layer norm: (B, C, H, W) -> (B, H, W, C) -> norm -> (B, C, H, W)
        out = out.permute(0, 2, 3, 1)  # (B, H, W, C)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        return out


class ImageAdapterOutputConv(nn.Module):
    """
    Output convolution module for image adapter.
    
    Takes merged image features and produces final output through conv layers.
    """
    def __init__(self, input_dim: int, head_features_1: int, head_features_2: int, output_dim: int):
        """
        Args:
            input_dim: Input feature dimension
            head_features_1: First hidden layer dimension
            head_features_2: Second hidden layer dimension  
            output_dim: Output feature dimension
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, head_features_1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_1, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Output tensor of shape (B, output_dim, H, W)
        """
        return self.conv(x)


class ImageAdapter(nn.Module):
    """
    Image Adapter module that enhances Gaussian parameters with direct image features.
    
    Structure:
        # DPT head 输出维度为 adapter_hidden_dim (如 256)
        # input_merger 输出维度也为 adapter_hidden_dim
        direct_img_feat = input_merger(imgs)  # (B, adapter_hidden_dim, H, W)
        combined = gs_param_output + direct_img_feat  # 融合
        out = output_conv(combined)  # (B, final_output_dim, H, W)
    """
    def __init__(
        self,
        param_name: str,
        input_merger_config: dict,
        output_conv_config: dict,
        adapter_hidden_dim: int,
        final_output_dim: int,
    ):
        """
        Args:
            param_name: Name of the Gaussian parameter (opacity, scale, rotation, rgb)
            input_merger_config: Configuration for input merger (simple conv network)
            output_conv_config: Configuration for output convolution
            adapter_hidden_dim: Hidden dimension for adapter (DPT head output & input_merger output)
            final_output_dim: Final output dimension for this parameter (1 for opacity, 3 for scale, etc.)
        """
        super().__init__()
        self.param_name = param_name
        self.adapter_hidden_dim = adapter_hidden_dim
        self.final_output_dim = final_output_dim
        
        # Build input merger (simple convolutional network)
        # Output dimension = adapter_hidden_dim to match DPT head output
        self.input_merger = ImageInputMerger(
            in_chans=input_merger_config.get("in_chans", 3),
            hidden_dim=input_merger_config.get("hidden_dim", 64),
            out_dim=adapter_hidden_dim,  # Output matches adapter_hidden_dim
            num_layers=input_merger_config.get("num_layers", 4),
            use_residual=input_merger_config.get("use_residual", True),
        )
        
        # Build output convolution
        # Input: adapter_hidden_dim (gs_param + img_feat)
        # Output: final_output_dim (1, 3, 4, etc.)
        self.output_conv = ImageAdapterOutputConv(
            input_dim=adapter_hidden_dim,
            head_features_1=output_conv_config["head_features_1"],
            head_features_2=output_conv_config["head_features_2"],
            output_dim=final_output_dim,
        )
        
    def forward(self, imgs: torch.Tensor, gs_param_output: torch.Tensor) -> torch.Tensor:
        """
        Apply image adapter enhancement.
        
        Args:
            imgs: Input images of shape (B*T, C, H, W)
            gs_param_output: Gaussian parameter output of shape (B*T, adapter_hidden_dim, H, W)
            
        Returns:
            Enhanced gs_param of shape (B*T, final_output_dim, H, W)
        """
        # Get image features through input merger (simple conv network, keeps size)
        direct_img_feat = self.input_merger(imgs)
        
        # Add gs_param output with image features (both have adapter_hidden_dim channels)
        combined_feat = gs_param_output + direct_img_feat
        
        # Apply output convolution to get final output dimension
        out = self.output_conv(combined_feat)
        
        return out


class FourierTimeEmbedding(nn.Module):
    """
    Fourier time embedding module that encodes time values using sinusoidal functions.
    
    This creates a positional encoding for time values using multiple frequencies,
    similar to the positional encoding used in Transformer models.
    """
    def __init__(self, num_frequencies: int = 10):
        """
        Args:
            num_frequencies: Number of frequency bands to use for Fourier features.
                           Output dimension will be 2*num_frequencies + 1 (sin, cos, and original time).
        """
        super().__init__()
        self.num_frequencies = num_frequencies
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier time embedding.
        
        Args:
            t: Time values of shape (..., 1), typically in range [0, 1]
            
        Returns:
            Fourier embedded time of shape (..., 2*num_frequencies + 1)
            Contains: [t, sin(2^0*π*t), cos(2^0*π*t), sin(2^1*π*t), cos(2^1*π*t), ...]
        """
        # Create frequency bands: [1, 2, 4, 8, 16, ...]
        freq = torch.arange(self.num_frequencies, device=t.device, dtype=t.dtype)
        freq = 2.0 ** freq
        
        # Compute angles for each frequency band
        # t: (..., 1), freq: (num_frequencies,) -> angles: (..., num_frequencies)
        angles = t * freq.unsqueeze(0) * np.pi
        
        # Compute sin and cos embeddings
        sin_embed = torch.sin(angles)  # (..., num_frequencies)
        cos_embed = torch.cos(angles)  # (..., num_frequencies)
        
        # Concatenate original time with sin and cos embeddings
        fourier_embed = torch.cat([t, sin_embed, cos_embed], dim=-1)  # (..., 2*num_frequencies + 1)
        
        return fourier_embed


class MapAnythingGS(nn.Module, PyTorchModelHubMixin):
    "Modular MapAnything model class that supports input of images & optional geometric modalities (multiple reconstruction tasks)."

    def __init__(
        self,
        name: str,
        encoder_config: Dict,
        info_sharing_config: Dict,
        pred_head_config: Dict,
        geometric_input_config: Dict,
        fusion_norm_layer: Union[Type[nn.Module], Callable[..., nn.Module]] = partial(
            nn.LayerNorm, eps=1e-6
        ),
        pretrained_checkpoint_path: str = None,
        load_specific_pretrained_submodules: bool = False,
        specific_pretrained_submodules: list = None,
        torch_hub_force_reload: bool = False,
    ):
        """
        Multi-view model containing an image encoder fused with optional geometric modalities followed by a multi-view attention transformer and respective downstream heads.
        The goal is to output scene representation.
        The multi-view attention transformer also takes as input a scale token to predict the metric scaling factor for the predicted scene representation.

        Args:
            name (str): Name of the model.
            encoder_config (Dict): Configuration for the encoder.
            info_sharing_config (Dict): Configuration for the multi-view attention transformer.
            pred_head_config (Dict): Configuration for the prediction heads.
            geometric_input_config (Dict): Configuration for the input of optional geometric modalities.
            fusion_norm_layer (Union[Type[nn.Module], Callable[..., nn.Module]]): Normalization layer to use after fusion (addition) of encoder and geometric modalities. (default: partial(nn.LayerNorm, eps=1e-6))
            pretrained_checkpoint_path (str): Path to pretrained checkpoint. (default: None)
            load_specific_pretrained_submodules (bool): Whether to load specific pretrained submodules. (default: False)
            specific_pretrained_submodules (list): List of specific pretrained submodules to load. Must be provided when load_specific_pretrained_submodules is True. (default: None)
            torch_hub_force_reload (bool): Whether to force reload the encoder from torch hub. (default: False)
        """
        super().__init__()

        # Initalize the attributes
        self.name = name
        self.encoder_config = encoder_config
        self.info_sharing_config = info_sharing_config
        self.pred_head_config = pred_head_config
        self.geometric_input_config = geometric_input_config
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        # Inference-time smoothing of the recovered intrinsics along time (see decoded_gs_params_v2)
        self.fix_decode_k_mode = "none"
        self.fix_decode_k_sigma = 3.0
        self.load_specific_pretrained_submodules = load_specific_pretrained_submodules
        self.specific_pretrained_submodules = specific_pretrained_submodules
        self.torch_hub_force_reload = torch_hub_force_reload
        self.class_init_args = {
            "name": self.name,
            "encoder_config": self.encoder_config,
            "info_sharing_config": self.info_sharing_config,
            "pred_head_config": self.pred_head_config,
            "geometric_input_config": self.geometric_input_config,
            "pretrained_checkpoint_path": self.pretrained_checkpoint_path,
            "load_specific_pretrained_submodules": self.load_specific_pretrained_submodules,
            "specific_pretrained_submodules": self.specific_pretrained_submodules,
            "torch_hub_force_reload": self.torch_hub_force_reload,
        }

        # Get relevant parameters from the configs
        self.info_sharing_type = info_sharing_config["model_type"]
        self.info_sharing_return_type = info_sharing_config["model_return_type"]
        self.pred_head_type = pred_head_config["type"]

        # Initialize image encoder
        if self.encoder_config["uses_torch_hub"]:
            self.encoder_config["torch_hub_force_reload"] = torch_hub_force_reload
        # Create a copy of the config before deleting the key to preserve it for serialization
        encoder_config_copy = self.encoder_config.copy()
        del encoder_config_copy["uses_torch_hub"]
        self.encoder = encoder_factory(**encoder_config_copy)

        # Initialize the encoder for ray directions
        ray_dirs_encoder_config = self.geometric_input_config["ray_dirs_encoder_config"]
        ray_dirs_encoder_config["enc_embed_dim"] = self.encoder.enc_embed_dim
        ray_dirs_encoder_config["patch_size"] = self.encoder.patch_size
        self.ray_dirs_encoder = encoder_factory(**ray_dirs_encoder_config)

        # Initialize the encoder for depth (normalized per view and values after normalization are scaled logarithmically)
        depth_encoder_config = self.geometric_input_config["depth_encoder_config"]
        depth_encoder_config["enc_embed_dim"] = self.encoder.enc_embed_dim
        depth_encoder_config["patch_size"] = self.encoder.patch_size
        self.depth_encoder = encoder_factory(**depth_encoder_config)

        # Initialize the encoder for log scale factor of depth
        # depth_scale_encoder_config = self.geometric_input_config["scale_encoder_config"]
        # depth_scale_encoder_config["enc_embed_dim"] = self.encoder.enc_embed_dim
        # self.depth_scale_encoder = encoder_factory(**depth_scale_encoder_config)

        # Initialize the encoder for camera rotation
        cam_rot_encoder_config = self.geometric_input_config["cam_rot_encoder_config"]
        cam_rot_encoder_config["enc_embed_dim"] = self.encoder.enc_embed_dim
        self.cam_rot_encoder = encoder_factory(**cam_rot_encoder_config)

        # Initialize the encoder for camera translation (normalized across all provided camera translations)
        cam_trans_encoder_config = self.geometric_input_config[
            "cam_trans_encoder_config"
        ]
        cam_trans_encoder_config["enc_embed_dim"] = self.encoder.enc_embed_dim
        self.cam_trans_encoder = encoder_factory(**cam_trans_encoder_config)

        # Initialize the encoder for log scale factor of camera translation
        cam_trans_scale_encoder_config = self.geometric_input_config[
            "scale_encoder_config"
        ]
        cam_trans_scale_encoder_config["enc_embed_dim"] = self.encoder.enc_embed_dim
        self.cam_trans_scale_encoder = encoder_factory(**cam_trans_scale_encoder_config)

        # Initialize the encoder for RGB (optional)
        # This is optional - only initialize if rgb_encoder_config is provided
        if "rgb_encoder_config" in self.geometric_input_config:
            rgb_encoder_config = self.geometric_input_config["rgb_encoder_config"].copy()
            rgb_encoder_config["enc_embed_dim"] = self.encoder.enc_embed_dim
            rgb_encoder_config["patch_size"] = self.encoder.patch_size
            self.rgb_encoder = encoder_factory(**rgb_encoder_config)
            print(f"Initialized RGB encoder with config: {rgb_encoder_config}")
        else:
            self.rgb_encoder = None

        # Initialize the encoder for time (normalized from 0 to 1 based on view index)
        # This is optional - only initialize if time_encoder_config is provided
        if "time_encoder_config" in self.geometric_input_config:
            # Check if Fourier time embedding is enabled
            if "num_time_frequencies" in self.geometric_input_config:
                # Use Fourier time embedding
                num_time_frequencies = self.geometric_input_config["num_time_frequencies"]
                self.fourier_time_embedding = FourierTimeEmbedding(num_frequencies=num_time_frequencies)
                self.use_fourier_time_embedding = True
                
                # Calculate the output dimension of Fourier embedding
                fourier_output_dim = 2 * num_time_frequencies + 1
                
                # Initialize time encoder with the correct input dimension
                time_encoder_config = self.geometric_input_config["time_encoder_config"].copy()
                time_encoder_config["enc_embed_dim"] = self.encoder.enc_embed_dim
                time_encoder_config["in_chans"] = fourier_output_dim  # Update to match Fourier embedding output
                self.time_encoder = encoder_factory(**time_encoder_config)
                
                print(f"Initialized time encoder with Fourier embedding (num_frequencies={num_time_frequencies}, output_dim={fourier_output_dim})")
            else:
                # Use original time values directly (backward compatibility)
                self.fourier_time_embedding = None
                self.use_fourier_time_embedding = False
                
                # Initialize time encoder with the original input dimension from config
                time_encoder_config = self.geometric_input_config["time_encoder_config"].copy()
                time_encoder_config["enc_embed_dim"] = self.encoder.enc_embed_dim
                # Keep the original in_chans from config (should be 1)
                self.time_encoder = encoder_factory(**time_encoder_config)
                
                print(f"Initialized time encoder without Fourier embedding (input_dim={time_encoder_config.get('in_chans', 1)})")
        else:
            self.time_encoder = None
            self.fourier_time_embedding = None
            self.use_fourier_time_embedding = False

        # Initialize the fusion norm layer
        self.fusion_norm_layer = fusion_norm_layer(self.encoder.enc_embed_dim)

        # Initialize the Scale Token
        # Used to scale the final scene predictions to metric scale
        # During inference extended to (B, C, T), where T is the number of tokens (i.e., 1)
        self.scale_token = nn.Parameter(torch.zeros(self.encoder.enc_embed_dim))
        torch.nn.init.trunc_normal_(self.scale_token, std=0.02)

        # Initialize the info sharing module (multi-view transformer)
        self._initialize_info_sharing(info_sharing_config)

        # Initialize the prediction heads
        self._initialize_prediction_heads(pred_head_config)

        # Initialize the final adaptors
        self._initialize_adaptors(pred_head_config)
        
        # Set default values for image adapter (may be overwritten by _initialize_image_adapters)
        if not hasattr(self, 'use_image_adapter'):
            self.use_image_adapter = False
            self.image_adapters = None
            self.shared_input_merger = None
            self.per_param_input_mergers = None
            self.image_adapter_output_convs = None
            self.adapter_hidden_dim = None
            self.share_input_merger = True

        # Load pretrained weights
        self._load_pretrained_weights()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _initialize_info_sharing(self, info_sharing_config):
        """
        Initialize the information sharing module based on the configuration.

        This method sets up the custom positional encoding if specified and initializes
        the appropriate multi-view transformer based on the configuration type.

        Args:
            info_sharing_config (Dict): Configuration for the multi-view attention transformer.
                Should contain 'custom_positional_encoding', 'model_type', and 'model_return_type'.

        Returns:
            None

        Raises:
            ValueError: If invalid configuration options are provided.
        """
        # Initialize Custom Positional Encoding if required
        custom_positional_encoding = info_sharing_config["custom_positional_encoding"]
        if custom_positional_encoding is not None:
            if isinstance(custom_positional_encoding, str):
                print(
                    f"Using custom positional encoding for multi-view attention transformer: {custom_positional_encoding}"
                )
                raise ValueError(
                    f"Invalid custom_positional_encoding: {custom_positional_encoding}. None implemented."
                )
            elif isinstance(custom_positional_encoding, Callable):
                print(
                    "Using callable function as custom positional encoding for multi-view attention transformer."
                )
                self.custom_positional_encoding = custom_positional_encoding
        else:
            self.custom_positional_encoding = None

        # Add dependecies to info_sharing_config
        info_sharing_config["module_args"]["input_embed_dim"] = (
            self.encoder.enc_embed_dim
        )
        info_sharing_config["module_args"]["custom_positional_encoding"] = (
            self.custom_positional_encoding
        )

        # Initialize Multi-View Transformer (Aeon release: alternating attention only)
        if self.info_sharing_type != "alternating_attention":
            raise ValueError(
                f"Unsupported info_sharing_type: {self.info_sharing_type}. The Aeon inference "
                "release only ships the 'alternating_attention' transformer."
            )
        if self.info_sharing_return_type == "no_intermediate_features":
            # Returns only normalized last layer features
            self.info_sharing = MultiViewAlternatingAttentionTransformer(
                **info_sharing_config["module_args"]
            )
        elif self.info_sharing_return_type == "intermediate_features":
            # Returns intermediate features and normalized last layer features
            self.info_sharing = MultiViewAlternatingAttentionTransformerIFR(
                **info_sharing_config["module_args"]
            )
            # Assess if the DPT needs to use encoder features
            if len(self.info_sharing.indices) == 2:
                self.use_encoder_features_for_dpt = True
            elif len(self.info_sharing.indices) == 3:
                self.use_encoder_features_for_dpt = False
            else:
                raise ValueError(
                    "Invalid number of indices provided for info sharing feature returner. Please provide 2 or 3 indices."
                )
        else:
            raise ValueError(
                f"Invalid info_sharing_return_type: {self.info_sharing_return_type}. Valid options: ['no_intermediate_features', 'intermediate_features']"
            )

    def _initialize_prediction_heads(self, pred_head_config):
        """
        Initialize the prediction heads based on the prediction head configuration.

        This method configures and initializes the appropriate prediction heads based on the
        specified prediction head type (linear, DPT, or DPT+pose). It sets up the necessary
        dependencies and creates the required model components.

        Args:
            pred_head_config (Dict): Configuration for the prediction heads.

        Returns:
            None

        Raises:
            ValueError: If an invalid pred_head_type is provided.
        """
        # Add dependencies to prediction head config
        pred_head_config["feature_head"]["patch_size"] = self.encoder.patch_size
        if self.pred_head_type == "linear":
            pred_head_config["feature_head"]["input_feature_dim"] = (
                self.info_sharing.dim
            )
        elif "dpt" in self.pred_head_type:
            # Add dependencies for DPT & Regressor head
            if self.use_encoder_features_for_dpt:
                pred_head_config["feature_head"]["input_feature_dims"] = [
                    self.encoder.enc_embed_dim
                ] + [self.info_sharing.dim] * 3
            else:
                pred_head_config["feature_head"]["input_feature_dims"] = [
                    self.info_sharing.dim
                ] * 4
            pred_head_config["regressor_head"]["input_feature_dim"] = pred_head_config[
                "feature_head"
            ]["feature_dim"]
            # Add dependencies for Pose head if required
            if "pose" in self.pred_head_type:
                pred_head_config["pose_head"]["patch_size"] = self.encoder.patch_size
                pred_head_config["pose_head"]["input_feature_dim"] = (
                    self.info_sharing.dim
                )
        else:
            raise ValueError(
                f"Invalid pred_head_type: {self.pred_head_type}. Valid options: ['linear', 'dpt', 'dpt+pose']"
            )
        pred_head_config["scale_head"]["input_feature_dim"] = self.info_sharing.dim

        # Initialize Prediction Heads
        if self.pred_head_type == "linear":
            # Initialize Dense Prediction Head for all views
            self.dense_head = LinearFeature(**pred_head_config["feature_head"])
        elif "dpt" in self.pred_head_type:
            # Initialze Dense Predction Head for all views
            self.dpt_feature_head = DPTFeature(**pred_head_config["feature_head"])
            self.dpt_regressor_head = DPTRegressionProcessor(
                **pred_head_config["regressor_head"]
            )
            self.dense_head = nn.Sequential(
                self.dpt_feature_head, self.dpt_regressor_head
            )


            self.omega_deg = pred_head_config["omega_deg"]
            self.ms3_deg = pred_head_config["ms3_deg"]
            self.color_dim = pred_head_config["color_dim"]
            # Control whether to checkpoint per-parameter MLP heads to save memory
            self.gradient_checkpointing = pred_head_config.get(
                "gradient_checkpointing", False
            )
            self.gs_params = [
                # "depth",
                "opacity",   # Alpha/transparency value for blending
                "scale",     # 3D scale factors (sx, sy, sz) for Gaussian size
                "rotation",  # Quaternion (qw, qx, qy, qz) for 3D orientation
                "dxyzt",     # Position residuals (dx, dy, dz, dt) for fine adjustments
                "ms3",       # Marginal scale for 4D motion (velocity components)
                "cov_t",     # Temporal covariance for time extent
                "omega",     # Angular velocity for rotation over time
                "rgb",       # Color features (RGB or RGBM depending on color_space)
            ]
            self.gs_params_split = [
                # 1,                    # depth: 1 channel
                1,                    # opacity: 1 channel  
                3,                    # scale: 3 channels (x, y, z)
                4,                    # rotation: 4 channels (quaternion)
                4,                    # dxyzt: 4 channels (3 spatial + 1 temporal)
                4 * self.ms3_deg,     # ms3: 4 channels per degree (3 direction + 1 magnitude)  4
                1,                    # cov_t: 1 channel
                4 * self.omega_deg,   # omega: 4 channels per degree (3 axis + 1 magnitude)  2
                self.color_dim,       # rgb: 3 for RGB or 4 for RGBM 4
            ]
            
            # Initialize separate MLP heads for each Gaussian Splatting parameter
            self.use_separate_dpt_heads = pred_head_config.get("use_separate_dpt_heads", False)
            if self.use_separate_dpt_heads:
                # 获取 image adapter 配置 (如果启用)
                use_image_adapter = pred_head_config.get("use_image_adapter", False)
                image_adapter_config = pred_head_config.get("image_adapter_config", {})
                adapter_hidden_dim = image_adapter_config.get("adapter_hidden_dim", 256)
                enabled_adapter_params = image_adapter_config.get("enabled_params", {})
                
                # 为每个高斯溅射参数创建独立的 DPT feature head 和 regressor head
                self.dpt_feature_heads_gs = nn.ModuleDict()
                self.dpt_regressor_heads_gs = nn.ModuleDict()
                self.dense_heads_gs = nn.ModuleDict()
                
                # 保存每个参数的 DPT 输出维度 (用于 adapter)
                self.gs_params_dpt_output_dims = {}
                
                for param_name, param_dim in zip(self.gs_params, self.gs_params_split):
                    # 为每个参数创建独立的 feature head
                    self.dpt_feature_heads_gs[param_name] = DPTFeature(**pred_head_config["feature_head_gs"])
                    
                    # 判断该参数是否启用 image adapter
                    # 如果启用，DPT head 输出 adapter_hidden_dim；否则输出 param_dim
                    param_uses_adapter = use_image_adapter and enabled_adapter_params.get(param_name, False)
                    
                    if param_uses_adapter:
                        # 带 adapter：DPT head 输出 adapter_hidden_dim
                        dpt_output_dim = adapter_hidden_dim
                    else:
                        # 不带 adapter：DPT head 输出最终维度
                        dpt_output_dim = param_dim
                    
                    self.gs_params_dpt_output_dims[param_name] = dpt_output_dim
                    
                    # 为每个参数创建独立的 regressor head
                    regressor_config = pred_head_config["regressor_head_gs"].copy()
                    regressor_config["output_dim"] = dpt_output_dim
                    self.dpt_regressor_heads_gs[param_name] = DPTRegressionProcessor(**regressor_config)
                    
                    # 组合 feature head 和 regressor head
                    self.dense_heads_gs[param_name] = nn.Sequential(
                        self.dpt_feature_heads_gs[param_name],
                        self.dpt_regressor_heads_gs[param_name]
                    )
                    
                    if param_uses_adapter:
                        print(f"  - {param_name}: DPT output_dim={adapter_hidden_dim} (with adapter) -> final_dim={param_dim}")
                    else:
                        print(f"  - {param_name}: DPT output_dim={param_dim} (no adapter)")

            else:
                self.dpt_feature_head_gs = DPTFeature(**pred_head_config["feature_head_gs"])
                self.dpt_regressor_head_gs = DPTRegressionProcessor(
                    **pred_head_config["regressor_head_gs"]
                )
                self.dense_head_gs = nn.Sequential(
                    self.dpt_feature_head_gs, self.dpt_regressor_head_gs
                )
            
            # Initialize Pose Head for all views if required
            if "pose" in self.pred_head_type:
                self.pose_head = PoseHead(**pred_head_config["pose_head"])
                
            # Initialize Image Adapters for Gaussian parameters (optional)
            self._initialize_image_adapters(pred_head_config)
        else:
            raise ValueError(
                f"Invalid pred_head_type: {self.pred_head_type}. Valid options: ['linear', 'dpt', 'dpt+pose']"
            )
        self.scale_head = MLPHead(**pred_head_config["scale_head"])

        self.high_resolution = pred_head_config["use_high_resolution"]

        self.perturb_xyz = pred_head_config.get("perturb_xyz", False)
        self.perturb_t = pred_head_config.get("perturb_t", False)

    def _initialize_adaptors(self, pred_head_config):
        """
        Initialize the adaptors based on the prediction head configuration.

        This method sets up the appropriate adaptors for different scene representation types,
        such as pointmaps, ray maps with depth, or ray directions with depth and pose.

        Args:
            pred_head_config (Dict): Configuration for the prediction heads including adaptor type.

        Returns:
            None

        Raises:
            ValueError: If an invalid adaptor_type is provided.
            AssertionError: If ray directions + depth + pose is used with an incompatible head type.
        """
        if pred_head_config["adaptor_type"] == "pointmap":
            self.dense_adaptor = PointMapAdaptor(**pred_head_config["adaptor"])
            self.scene_rep_type = "pointmap"
        elif pred_head_config["adaptor_type"] == "pointmap+confidence":
            self.dense_adaptor = PointMapWithConfidenceAdaptor(
                **pred_head_config["adaptor"]
            )
            self.scene_rep_type = "pointmap+confidence"
        elif pred_head_config["adaptor_type"] == "pointmap+mask":
            self.dense_adaptor = PointMapWithMaskAdaptor(**pred_head_config["adaptor"])
            self.scene_rep_type = "pointmap+mask"
        elif pred_head_config["adaptor_type"] == "pointmap+confidence+mask":
            self.dense_adaptor = PointMapWithConfidenceAndMaskAdaptor(
                **pred_head_config["adaptor"]
            )
            self.scene_rep_type = "pointmap+confidence+mask"
        elif pred_head_config["adaptor_type"] == "raymap+depth":
            self.dense_adaptor = RayMapPlusDepthAdaptor(**pred_head_config["adaptor"])
            self.scene_rep_type = "raymap+depth"
        elif pred_head_config["adaptor_type"] == "raymap+depth+confidence":
            self.dense_adaptor = RayMapPlusDepthWithConfidenceAdaptor(
                **pred_head_config["adaptor"]
            )
            self.scene_rep_type = "raymap+depth+confidence"
        elif pred_head_config["adaptor_type"] == "raymap+depth+mask":
            self.dense_adaptor = RayMapPlusDepthWithMaskAdaptor(
                **pred_head_config["adaptor"]
            )
            self.scene_rep_type = "raymap+depth+mask"
        elif pred_head_config["adaptor_type"] == "raymap+depth+confidence+mask":
            self.dense_adaptor = RayMapPlusDepthWithConfidenceAndMaskAdaptor(
                **pred_head_config["adaptor"]
            )
            self.scene_rep_type = "raymap+depth+confidence+mask"
        elif pred_head_config["adaptor_type"] == "raydirs+depth+pose":
            assert self.pred_head_type == "dpt+pose", (
                "Ray directions + depth + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = RayDirectionsPlusDepthAdaptor(
                **pred_head_config["dpt_adaptor"]
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "raydirs+depth+pose"
        elif pred_head_config["adaptor_type"] == "raydirs+depth+pose+confidence":
            assert self.pred_head_type == "dpt+pose", (
                "Ray directions + depth + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = RayDirectionsPlusDepthWithConfidenceAdaptor(
                **pred_head_config["dpt_adaptor"]
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "raydirs+depth+pose+confidence"
        elif pred_head_config["adaptor_type"] == "raydirs+depth+pose+mask":
            assert self.pred_head_type == "dpt+pose", (
                "Ray directions + depth + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = RayDirectionsPlusDepthWithMaskAdaptor(
                **pred_head_config["dpt_adaptor"]
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "raydirs+depth+pose+mask"
        elif pred_head_config["adaptor_type"] == "raydirs+depth+pose+confidence+mask":
            assert self.pred_head_type == "dpt+pose", (
                "Ray directions + depth + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = RayDirectionsPlusDepthWithConfidenceAndMaskAdaptor(
                **pred_head_config["dpt_adaptor"]
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "raydirs+depth+pose+confidence+mask"
        elif pred_head_config["adaptor_type"] == "campointmap+pose":
            assert self.pred_head_type == "dpt+pose", (
                "Camera pointmap + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = PointMapAdaptor(**pred_head_config["dpt_adaptor"])
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "campointmap+pose"
        elif pred_head_config["adaptor_type"] == "campointmap+pose+confidence":
            assert self.pred_head_type == "dpt+pose", (
                "Camera pointmap + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = PointMapWithConfidenceAdaptor(
                **pred_head_config["dpt_adaptor"]
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "campointmap+pose+confidence"
        elif pred_head_config["adaptor_type"] == "campointmap+pose+mask":
            assert self.pred_head_type == "dpt+pose", (
                "Camera pointmap + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = PointMapWithMaskAdaptor(
                **pred_head_config["dpt_adaptor"]
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "campointmap+pose+mask"
        elif pred_head_config["adaptor_type"] == "campointmap+pose+confidence+mask":
            assert self.pred_head_type == "dpt+pose", (
                "Camera pointmap + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = PointMapWithConfidenceAndMaskAdaptor(
                **pred_head_config["dpt_adaptor"]
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "campointmap+pose+confidence+mask"
        elif pred_head_config["adaptor_type"] == "pointmap+raydirs+depth+pose":
            assert self.pred_head_type == "dpt+pose", (
                "Pointmap + ray directions + depth + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = PointMapPlusRayDirectionsPlusDepthAdaptor(
                **pred_head_config["dpt_adaptor"]
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "pointmap+raydirs+depth+pose"
        elif (
            pred_head_config["adaptor_type"] == "pointmap+raydirs+depth+pose+confidence"
        ):
            assert self.pred_head_type == "dpt+pose", (
                "Pointmap + ray directions + depth + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = (
                PointMapPlusRayDirectionsPlusDepthWithConfidenceAdaptor(
                    **pred_head_config["dpt_adaptor"]
                )
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "pointmap+raydirs+depth+pose+confidence"
        elif pred_head_config["adaptor_type"] == "pointmap+raydirs+depth+pose+mask":
            assert self.pred_head_type == "dpt+pose", (
                "Pointmap + ray directions + depth + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = PointMapPlusRayDirectionsPlusDepthWithMaskAdaptor(
                **pred_head_config["dpt_adaptor"]
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "pointmap+raydirs+depth+pose+mask"
        elif (
            pred_head_config["adaptor_type"]
            == "pointmap+raydirs+depth+pose+confidence+mask"
        ):
            assert self.pred_head_type == "dpt+pose", (
                "Pointmap + ray directions + depth + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = (
                PointMapPlusRayDirectionsPlusDepthWithConfidenceAndMaskAdaptor(
                    **pred_head_config["dpt_adaptor"]
                )
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "pointmap+raydirs+depth+pose+confidence+mask"
        else:
            raise ValueError(
                f"Invalid adaptor_type: {pred_head_config['adaptor_type']}. \
                Valid options: ['pointmap', 'raymap+depth', 'raydirs+depth+pose', 'campointmap+pose', 'pointmap+raydirs+depth+pose' \
                                'pointmap+confidence', 'raymap+depth+confidence', 'raydirs+depth+pose+confidence', 'campointmap+pose+confidence', 'pointmap+raydirs+depth+pose+confidence' \
                                'pointmap+mask', 'raymap+depth+mask', 'raydirs+depth+pose+mask', 'campointmap+pose+mask', 'pointmap+raydirs+depth+pose+mask' \
                                'pointmap+confidence+mask', 'raymap+depth+confidence+mask', 'raydirs+depth+pose+confidence+mask', 'campointmap+pose+confidence+mask', 'pointmap+raydirs+depth+pose+confidence+mask']"
            )
        self.scale_adaptor = ScaleAdaptor(**pred_head_config["scale_adaptor"])


        self.mlp_depth = pred_head_config["gs_adaptor"]["mlp_depth"]
        self.mlp_dim = pred_head_config["gs_adaptor"]["mlp_dim"]
        self.use_bias = pred_head_config["gs_adaptor"]["use_bias"]
        self.use_weight_norm = pred_head_config["gs_adaptor"]["use_weight_norm"]
        # heads = {}
        # for key, channel in zip(self.gs_params, self.gs_params_split):
        #     head = build_pytorch_mlp(
        #         pred_head_config["regressor_head_gs"]["output_dim"],
        #         self.mlp_dim * channel,
        #         channel,
        #         depth=self.mlp_depth,
        #         bias=self.use_bias,
        #         use_weight_norm=self.use_weight_norm,
        #     )
        #     heads[key] = head
        # self.linear_heads = nn.ModuleDict(heads)
        # def output_head(x: torch.Tensor):
        #     res = []
        #     for head in self.linear_heads.values():
        #         print(head(x).shape)
        #         res.append(head(x))
        #     return torch.cat(res, dim=-1)
        # self.gs_adaptor = output_head

        self.opacity_min = pred_head_config["opacity_min"]
        self.opacity_max = pred_head_config["opacity_max"]
        self.opacity_bias = pred_head_config["opacity_bias"]
        # self.depth_near = pred_head_config["depth_near"]
        # self.depth_far = pred_head_config["depth_far"]
        # self.depth_bias = pred_head_config["depth_bias"]
        self.scale_min = pred_head_config["scale_min"]
        self.scale_max = pred_head_config["scale_max"]
        self.scale_bias = pred_head_config["scale_bias"]
        self.dxyzt_min = pred_head_config["dxyzt_min"]
        self.dxyzt_max = pred_head_config["dxyzt_max"]
        self.dxyzt_bias = pred_head_config["dxyzt_bias"]
        self.dxyzt_mult = pred_head_config["dxyzt_mult"]
        self.sigmoid_ms3_min = pred_head_config["sigmoid_ms3_min"]
        self.sigmoid_ms3_max = pred_head_config["sigmoid_ms3_max"]
        self.sigmoid_ms3_bias = pred_head_config["sigmoid_ms3_bias"]
        self.ms3_clamp = pred_head_config["ms3_clamp"]
        self.cov_t_min = pred_head_config["cov_t_min"]
        self.cov_t_max = pred_head_config["cov_t_max"]
        self.cov_t_bias = pred_head_config["cov_t_bias"]
        self.sigmoid_omega_min = pred_head_config["sigmoid_omega_min"]
        self.sigmoid_omega_max = pred_head_config["sigmoid_omega_max"]
        self.sigmoid_omega_bias = pred_head_config["sigmoid_omega_bias"]
        self.omega_clamp = pred_head_config["omega_clamp"]
        self.rgb_min = pred_head_config["rgb_min"]
        self.rgb_max = pred_head_config["rgb_max"]
        self.rgb_bias = pred_head_config["rgb_bias"]

        self.recalc_magic_pattern = pred_head_config["recalc_magic_pattern"]
        self.recalc_method = pred_head_config["recalc_method"]
        self.magic_num = pred_head_config["magic_num"]
        self.magic_pattern = pred_head_config["magic_pattern"]
        self.patch_size = pred_head_config["patch_size"]
        self.ms3_deg_downmax_mult = pred_head_config["ms3_deg_downmax_mult"]
        self.omega_deg_downmax_mult = pred_head_config["omega_deg_downmax_mult"]

    def _initialize_image_adapters(self, pred_head_config):
        """
        Initialize image adapters for Gaussian parameters.
        
        Image adapters enhance specific Gaussian parameters (opacity, scale, rotation, rgb)
        with direct image features through a simple conv network + output conv structure.
        
        Structure:
            # 根据 share_input_merger 配置决定是共享还是独立
            if share_input_merger:
                shared_img_feat = shared_input_merger(imgs)  # 所有参数共享
            else:
                img_feat = per_param_input_merger[param](imgs)  # 每个参数独立
            combined = gs_param_output + img_feat
            out = output_conv(combined)
        
        Args:
            pred_head_config: Configuration dictionary for prediction heads
        """
        # Check if image adapter is enabled
        self.use_image_adapter = pred_head_config.get("use_image_adapter", False)
        
        if not self.use_image_adapter:
            self.image_adapters = None
            self.shared_input_merger = None
            self.per_param_input_mergers = None
            self.adapter_hidden_dim = None
            self.share_input_merger = True
            return
        
        image_adapter_config = pred_head_config.get("image_adapter_config", {})
        enabled_params = image_adapter_config.get("enabled_params", {})
        input_merger_config = image_adapter_config.get("input_merger", {})
        output_conv_config = image_adapter_config.get("output_conv", {})
        self.adapter_hidden_dim = image_adapter_config.get("adapter_hidden_dim", 256)
        # 是否共享 input_merger (默认 True 以兼容之前的配置)
        self.share_input_merger = image_adapter_config.get("share_input_merger", True)
        
        # Get final output dimensions for each parameter
        param_final_output_dims = {
            "opacity": 1,
            "scale": 3,
            "rotation": 4,
            "rgb": self.color_dim,
        }
        
        print(f"Initializing ImageAdapters with adapter_hidden_dim={self.adapter_hidden_dim}, share_input_merger={self.share_input_merger}")
        
        if self.share_input_merger:
            # 创建共享的 input_merger (所有参数共用，只计算一次图像特征)
            self.shared_input_merger = ImageInputMerger(
                in_chans=input_merger_config.get("in_chans", 3),
                hidden_dim=input_merger_config.get("hidden_dim", 64),
                out_dim=self.adapter_hidden_dim,
                num_layers=input_merger_config.get("num_layers", 4),
                use_residual=input_merger_config.get("use_residual", True),
            )
            self.per_param_input_mergers = None
            print(f"  - Shared ImageInputMerger: in_chans=3 -> out_dim={self.adapter_hidden_dim}")
        else:
            # 为每个启用的参数创建独立的 input_merger
            self.shared_input_merger = None
            self.per_param_input_mergers = nn.ModuleDict()
            
            for param_name, is_enabled in enabled_params.items():
                if is_enabled and param_name in param_final_output_dims:
                    input_merger = ImageInputMerger(
                        in_chans=input_merger_config.get("in_chans", 3),
                        hidden_dim=input_merger_config.get("hidden_dim", 64),
                        out_dim=self.adapter_hidden_dim,
                        num_layers=input_merger_config.get("num_layers", 4),
                        use_residual=input_merger_config.get("use_residual", True),
                    )
                    self.per_param_input_mergers[param_name] = input_merger
                    print(f"  - Independent ImageInputMerger for '{param_name}': in_chans=3 -> out_dim={self.adapter_hidden_dim}")
        
        # Build output convs for enabled parameters (每个参数有自己的 output_conv)
        self.image_adapter_output_convs = nn.ModuleDict()
        
        for param_name, is_enabled in enabled_params.items():
            if is_enabled and param_name in param_final_output_dims:
                final_output_dim = param_final_output_dims[param_name]
                
                output_conv = ImageAdapterOutputConv(
                    input_dim=self.adapter_hidden_dim,
                    head_features_1=output_conv_config["head_features_1"],
                    head_features_2=output_conv_config["head_features_2"],
                    output_dim=final_output_dim,
                )
                self.image_adapter_output_convs[param_name] = output_conv
                print(f"  - OutputConv for '{param_name}': hidden_dim={self.adapter_hidden_dim} -> final_dim={final_output_dim}")
        
        # 为了兼容性，设置 image_adapters 为 enabled_params 的 keys
        self.image_adapters = self.image_adapter_output_convs
        
        print(f"Total ImageAdapter output_convs initialized: {len(self.image_adapter_output_convs)}")

        

    def _load_pretrained_weights(self):
        """
        Load pretrained weights from a checkpoint file.

        If load_specific_pretrained_submodules is True, only loads weights for the specified submodules.
        Otherwise, loads all weights from the checkpoint.

        Returns:
            None
        """
        if self.pretrained_checkpoint_path is not None:
            if not self.load_specific_pretrained_submodules:
                print(
                    f"Loading pretrained MapAnything weights from {self.pretrained_checkpoint_path} ..."
                )
                ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
                
                # Filter out parameters that have shape mismatch (e.g., view_pos_table when changing PE settings)
                # Also skip dpt_feature_head and dpt_regressor_head modules
                model_state = self.state_dict()
                filtered_ckpt = {}
                skipped_keys = []
                skip_prefixes = ("dpt_feature_head.", "dpt_regressor_head.")
                
                for key, value in ckpt["model"].items():
                    # Skip dpt_feature_head and dpt_regressor_head modules
                    if key.startswith(skip_prefixes):
                        skipped_keys.append(f"{key} (skipped: dpt head module)")
                        continue
                    
                    if key in model_state:
                        # Check if shapes match
                        if model_state[key].shape == value.shape:
                            filtered_ckpt[key] = value
                        else:
                            # Skip parameters with shape mismatch (will use initialized values)
                            skipped_keys.append(f"{key} (checkpoint: {value.shape} vs model: {model_state[key].shape})")
                    else:
                        # Key not in current model
                        skipped_keys.append(f"{key} (not in model)")
                
                if skipped_keys:
                    print(f"Skipped loading the following parameters due to shape mismatch or absence:")
                    for skipped_key in skipped_keys:
                        print(f"  - {skipped_key}")
                
                print(self.load_state_dict(filtered_ckpt, strict=False))
            else:
                print(
                    f"Loading pretrained MapAnything weights from {self.pretrained_checkpoint_path} for specific submodules: {self.specific_pretrained_submodules} ..."
                )
                assert self.pred_head_type is not None, (
                    "Specific submodules to load cannot be None."
                )
                ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
                filtered_ckpt = {}
                for ckpt_key, ckpt_value in ckpt["model"].items():
                    for submodule in self.specific_pretrained_submodules:
                        if ckpt_key.startswith(submodule):
                            filtered_ckpt[ckpt_key] = ckpt_value
                print(self.load_state_dict(filtered_ckpt, strict=False))


    def _encode_n_views(self, views):
        """
        Encode all the input views (batch of images) in a single forward pass.
        Assumes all the input views have the same image shape, batch size, and data normalization type.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.

        Returns:
            List[torch.Tensor]: A list containing the encoded features for all N views.
        """
        num_views = len(views)
        data_norm_type = views[0]["data_norm_type"][0]
        imgs_list = [view["img"] for view in views]
        all_imgs_across_views = torch.cat(imgs_list, dim=0)
        encoder_input = ViTEncoderInput(
            image=all_imgs_across_views, data_norm_type=data_norm_type
        )
        encoder_output = self.encoder(encoder_input)
        all_encoder_features_across_views = encoder_output.features.chunk(
            num_views, dim=0
        )

        return all_encoder_features_across_views

    def _compute_pose_quats_and_trans_for_across_views_in_ref_view(
        self,
        views,
        num_views,
        device,
        dtype,
        batch_size_per_view,
        per_sample_cam_input_mask,
    ):
        """
        Compute the pose quats and trans for all the views in the frame of the reference view 0.
        Returns identity pose for views where the camera input mask is False or the pose is not provided.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
            num_views (int): Number of views.
            device (torch.device): Device to use for the computation.
            dtype (torch.dtype): Data type to use for the computation.
            per_sample_cam_input_mask (torch.Tensor): Tensor containing the per sample camera input mask.

        Returns:
            torch.Tensor: A tensor containing the pose quats for all the views in the frame of the reference view 0. (batch_size_per_view * view, 4)
            torch.Tensor: A tensor containing the pose trans for all the views in the frame of the reference view 0. (batch_size_per_view * view, 3)
            torch.Tensor: A tensor containing the per sample camera input mask.
        """
        # Compute the pose quats and trans for all the non-reference views in the frame of the reference view 0
        pose_quats_non_ref_views = []
        pose_trans_non_ref_views = []
        pose_quats_ref_view_0 = []
        pose_trans_ref_view_0 = []
        for view_idx in range(num_views):
            per_sample_cam_input_mask_for_curr_view = per_sample_cam_input_mask[
                view_idx * batch_size_per_view : (view_idx + 1) * batch_size_per_view
            ]
            if (
                "camera_pose_quats" in views[view_idx]
                and "camera_pose_trans" in views[view_idx]
                and per_sample_cam_input_mask_for_curr_view.any()
            ):
                # Get the camera pose quats and trans for the current view
                cam_pose_quats = views[view_idx]["camera_pose_quats"][
                    per_sample_cam_input_mask_for_curr_view
                ]
                cam_pose_trans = views[view_idx]["camera_pose_trans"][
                    per_sample_cam_input_mask_for_curr_view
                ]
                # Append to the list
                pose_quats_non_ref_views.append(cam_pose_quats)
                pose_trans_non_ref_views.append(cam_pose_trans)
                # Get the camera pose quats and trans for the reference view 0
                cam_pose_quats = views[0]["camera_pose_quats"][
                    per_sample_cam_input_mask_for_curr_view
                ]
                cam_pose_trans = views[0]["camera_pose_trans"][
                    per_sample_cam_input_mask_for_curr_view
                ]
                # Append to the list
                pose_quats_ref_view_0.append(cam_pose_quats)
                pose_trans_ref_view_0.append(cam_pose_trans)
            else:
                per_sample_cam_input_mask[
                    view_idx * batch_size_per_view : (view_idx + 1)
                    * batch_size_per_view
                ] = False

        # Initialize the pose quats and trans for all views as identity
        pose_quats_across_views = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], dtype=dtype, device=device
        ).repeat(batch_size_per_view * num_views, 1)  # (q_x, q_y, q_z, q_w)
        pose_trans_across_views = torch.zeros(
            (batch_size_per_view * num_views, 3), dtype=dtype, device=device
        )

        # Compute the pose quats and trans for all the non-reference views in the frame of the reference view 0
        if len(pose_quats_non_ref_views) > 0:
            # Stack the pose quats and trans for all the non-reference views and reference view 0
            pose_quats_non_ref_views = torch.cat(pose_quats_non_ref_views, dim=0)
            pose_trans_non_ref_views = torch.cat(pose_trans_non_ref_views, dim=0)
            pose_quats_ref_view_0 = torch.cat(pose_quats_ref_view_0, dim=0)
            pose_trans_ref_view_0 = torch.cat(pose_trans_ref_view_0, dim=0)

            # Compute the pose quats and trans for all the non-reference views in the frame of the reference view 0
            (
                pose_quats_non_ref_views_in_ref_view_0,
                pose_trans_non_ref_views_in_ref_view_0,
            ) = transform_pose_using_quats_and_trans_2_to_1(
                pose_quats_ref_view_0,
                pose_trans_ref_view_0,
                pose_quats_non_ref_views,
                pose_trans_non_ref_views,
            )

            # Update the pose quats and trans for all the non-reference views
            pose_quats_across_views[per_sample_cam_input_mask] = (
                pose_quats_non_ref_views_in_ref_view_0.to(dtype=dtype)
            )
            pose_trans_across_views[per_sample_cam_input_mask] = (
                pose_trans_non_ref_views_in_ref_view_0.to(dtype=dtype)
            )

        return (
            pose_quats_across_views,
            pose_trans_across_views,
            per_sample_cam_input_mask,
        )

    def _encode_and_fuse_ray_dirs(
        self,
        views,
        num_views,
        batch_size_per_view,
        all_encoder_features_across_views,
        per_sample_ray_dirs_input_mask,
    ):
        """
        Encode the ray directions for all the views and fuse it with the other encoder features in a single forward pass.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
            num_views (int): Number of views.
            batch_size_per_view (int): Batch size per view.
            all_encoder_features_across_views (torch.Tensor): Tensor containing the encoded features for all N views.
            per_sample_ray_dirs_input_mask (torch.Tensor): Tensor containing the per sample ray direction input mask.

        Returns:
            torch.Tensor: A tensor containing the encoded features for all the views.
        """
        # Get the height and width of the images
        _, _, height, width = views[0]["img"].shape

        # Get the ray directions for all the views where info is provided and the ray direction input mask is True
        ray_dirs_list = []
        for view_idx in range(num_views):
            per_sample_ray_dirs_input_mask_for_curr_view = (
                per_sample_ray_dirs_input_mask[
                    view_idx * batch_size_per_view : (view_idx + 1)
                    * batch_size_per_view
                ]
            )
            ray_dirs_for_curr_view = torch.zeros(
                (batch_size_per_view, height, width, 3),
                dtype=all_encoder_features_across_views.dtype,
                device=all_encoder_features_across_views.device,
            )
            if (
                "ray_directions_cam" in views[view_idx]
                and per_sample_ray_dirs_input_mask_for_curr_view.any()
            ):
                ray_dirs_for_curr_view[per_sample_ray_dirs_input_mask_for_curr_view] = (
                    views[view_idx]["ray_directions_cam"][
                        per_sample_ray_dirs_input_mask_for_curr_view
                    ]
                )
            else:
                per_sample_ray_dirs_input_mask[
                    view_idx * batch_size_per_view : (view_idx + 1)
                    * batch_size_per_view
                ] = False
            ray_dirs_list.append(ray_dirs_for_curr_view)

        # Stack the ray directions for all the views and permute to (B * V, C, H, W)
        ray_dirs = torch.cat(ray_dirs_list, dim=0)  # (B * V, H, W, 3)
        ray_dirs = ray_dirs.permute(0, 3, 1, 2).contiguous()  # (B * V, 3, H, W)

        # Encode the ray directions
        ray_dirs_features_across_views = self.ray_dirs_encoder(
            ViTEncoderNonImageInput(data=ray_dirs)
        ).features

        # Fuse the ray direction features with the other encoder features (zero out the features where the ray direction input mask is False)
        ray_dirs_features_across_views = (
            ray_dirs_features_across_views
            * per_sample_ray_dirs_input_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )
        all_encoder_features_across_views = (
            all_encoder_features_across_views + ray_dirs_features_across_views
        )

        return all_encoder_features_across_views

    def _encode_and_fuse_depths(
        self,
        views,
        num_views,
        batch_size_per_view,
        all_encoder_features_across_views,
        per_sample_depth_input_mask,
    ):
        """
        Encode the z depths for all the views and fuse it with the other encoder features in a single forward pass.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
            num_views (int): Number of views.
            batch_size_per_view (int): Batch size per view.
            all_encoder_features_across_views (torch.Tensor): Tensor containing the encoded features for all N views.
            per_sample_depth_input_mask (torch.Tensor): Tensor containing the per sample depth input mask.

        Returns:
            torch.Tensor: A tensor containing the encoded features for all the views.
        """
        # Get the device and height and width of the images
        device = all_encoder_features_across_views.device
        _, _, height, width = views[0]["img"].shape

        # Decide to use randomly sampled sparse depth or dense depth
        # if torch.rand(1) < self.geometric_input_config["sparse_depth_prob"]:
        #     use_sparse_depth = True
        # else:
        #     use_sparse_depth = False
        
        use_sparse_depth = False

        # Get the depths for all the views
        depth_list = []
        depth_norm_factors_list = []
        metric_scale_depth_mask_list = []
        for view_idx in range(num_views):
            # Get the input mask for current view
            per_sample_depth_input_mask_for_curr_view = per_sample_depth_input_mask[
                view_idx * batch_size_per_view : (view_idx + 1) * batch_size_per_view
            ]
            depth_for_curr_view = torch.zeros(
                (batch_size_per_view, height, width, 1),
                dtype=all_encoder_features_across_views.dtype,
                device=device,
            )
            depth_norm_factor_for_curr_view = torch.zeros(
                (batch_size_per_view),
                dtype=all_encoder_features_across_views.dtype,
                device=device,
            )
            metric_scale_mask_for_curr_view = torch.zeros(
                (batch_size_per_view),
                dtype=torch.bool,
                device=device,
            )
            if (
                "depth_along_ray" in views[view_idx]
            ) and per_sample_depth_input_mask_for_curr_view.any():
                # Get depth for current view
                depth_for_curr_view_input = views[view_idx]["depth_along_ray"][
                    per_sample_depth_input_mask_for_curr_view
                ]
                # Get the metric scale mask
                if "is_metric_scale" in views[view_idx]:
                    metric_scale_mask = views[view_idx]["is_metric_scale"][
                        per_sample_depth_input_mask_for_curr_view
                    ]
                else:
                    metric_scale_mask = torch.zeros(
                        depth_for_curr_view_input.shape[0],
                        dtype=torch.bool,
                        device=device,
                    )
                # Turn off indication of metric scale samples based on the depth_scale_norm_all_prob
                depth_scale_norm_all_mask = (
                    torch.rand(metric_scale_mask.shape[0])
                    < self.geometric_input_config["depth_scale_norm_all_prob"]
                )
                if depth_scale_norm_all_mask.any():
                    metric_scale_mask[depth_scale_norm_all_mask] = False
                # Assign the metric scale mask to the respective indices
                metric_scale_mask_for_curr_view[
                    per_sample_depth_input_mask_for_curr_view
                ] = metric_scale_mask
                # Sparsely sample the depth if required
                if use_sparse_depth:
                    # Create a mask of ones
                    sparsification_mask = torch.ones_like(
                        depth_for_curr_view_input, device=device
                    )
                    # Create a mask for valid pixels (depth > 0)
                    valid_pixel_mask = depth_for_curr_view_input > 0
                    # Calculate the number of valid pixels
                    num_valid_pixels = valid_pixel_mask.sum().item()
                    # Calculate the number of valid pixels to set to zero
                    num_to_zero = int(
                        num_valid_pixels
                        * self.geometric_input_config["sparsification_removal_percent"]
                    )
                    if num_to_zero > 0:
                        # Get the indices of valid pixels
                        valid_indices = valid_pixel_mask.nonzero(as_tuple=True)
                        # Randomly select indices to zero out
                        indices_to_zero = torch.randperm(num_valid_pixels)[:num_to_zero]
                        # Set selected valid indices to zero in the mask
                        sparsification_mask[
                            valid_indices[0][indices_to_zero],
                            valid_indices[1][indices_to_zero],
                            valid_indices[2][indices_to_zero],
                            valid_indices[3][indices_to_zero],
                        ] = 0
                    # Apply the mask on the depth
                    depth_for_curr_view_input = (
                        depth_for_curr_view_input * sparsification_mask
                    )
                # Normalize the depth
                scaled_depth_for_curr_view_input, depth_norm_factor = (
                    normalize_depth_using_non_zero_pixels(
                        depth_for_curr_view_input, return_norm_factor=True
                    )
                )
                # Assign the depth and depth norm factor to the respective indices
                depth_for_curr_view[per_sample_depth_input_mask_for_curr_view] = (
                    scaled_depth_for_curr_view_input
                )
                depth_norm_factor_for_curr_view[
                    per_sample_depth_input_mask_for_curr_view
                ] = depth_norm_factor
            else:
                per_sample_depth_input_mask[
                    view_idx * batch_size_per_view : (view_idx + 1)
                    * batch_size_per_view
                ] = False
            # Append the depths, depth norm factor and metric scale mask for the current view
            depth_list.append(depth_for_curr_view)
            depth_norm_factors_list.append(depth_norm_factor_for_curr_view)
            metric_scale_depth_mask_list.append(metric_scale_mask_for_curr_view)

        # Stack the depths for all the views and permute to (B * V, C, H, W)
        depths = torch.cat(depth_list, dim=0)  # (B * V, H, W, 1)
        depths = apply_log_to_norm(
            depths
        )  # Scale logarithimically (norm is computed along last dim)
        depths = depths.permute(0, 3, 1, 2).contiguous()  # (B * V, 1, H, W)
        # Encode the depths using the depth encoder
        depth_features_across_views = self.depth_encoder(
            ViTEncoderNonImageInput(data=depths)
        ).features
        # Zero out the depth features where the depth input mask is False
        depth_features_across_views = (
            depth_features_across_views
            * per_sample_depth_input_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )

        # Stack the depth norm factors for all the views
        depth_norm_factors = torch.cat(depth_norm_factors_list, dim=0)  # (B * V, )
        # Encode the depth norm factors using the log scale encoder for depth
        log_depth_norm_factors = torch.log(depth_norm_factors + 1e-8)  # (B * V, )
        depth_scale_features_across_views = self.depth_scale_encoder(
            EncoderGlobalRepInput(data=log_depth_norm_factors.unsqueeze(-1))
        ).features
        # Zero out the depth scale features where the depth input mask is False
        depth_scale_features_across_views = (
            depth_scale_features_across_views
            * per_sample_depth_input_mask.unsqueeze(-1)
        )
        # Stack the metric scale mask for all the views
        metric_scale_depth_mask = torch.cat(
            metric_scale_depth_mask_list, dim=0
        )  # (B * V, )
        # Zero out the depth scale features where the metric scale mask is False
        # Scale encoding is only provided for metric scale samples
        depth_scale_features_across_views = (
            depth_scale_features_across_views * metric_scale_depth_mask.unsqueeze(-1)
        )

        # Fuse the depth features & depth scale features with the other encoder features
        all_encoder_features_across_views = (
            all_encoder_features_across_views
            + depth_features_across_views
            + depth_scale_features_across_views.unsqueeze(-1).unsqueeze(-1)
        )

        return all_encoder_features_across_views

    def _encode_and_fuse_cam_quats_and_trans(
        self,
        views,
        num_views,
        batch_size_per_view,
        all_encoder_features_across_views,
        pose_quats_across_views,
        pose_trans_across_views,
        per_sample_cam_input_mask,
    ):
        """
        Encode the camera quats and trans for all the views and fuse it with the other encoder features in a single forward pass.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
            num_views (int): Number of views.
            batch_size_per_view (int): Batch size per view.
            all_encoder_features_across_views (torch.Tensor): Tensor containing the encoded features for all N views.
            pose_quats_across_views (torch.Tensor): Tensor containing the pose quats for all the views in the frame of the reference view 0. (batch_size_per_view * view, 4)
            pose_trans_across_views (torch.Tensor): Tensor containing the pose trans for all the views in the frame of the reference view 0. (batch_size_per_view * view, 3)
            per_sample_cam_input_mask (torch.Tensor): Tensor containing the per sample camera input mask.

        Returns:
            torch.Tensor: A tensor containing the encoded features for all the views.
        """
        # Encode the pose quats
        pose_quats_features_across_views = self.cam_rot_encoder(
            EncoderGlobalRepInput(data=pose_quats_across_views)
        ).features
        # Zero out the pose quat features where the camera input mask is False
        pose_quats_features_across_views = (
            pose_quats_features_across_views * per_sample_cam_input_mask.unsqueeze(-1)
        )

        # Get the metric scale mask for all samples
        device = all_encoder_features_across_views.device
        metric_scale_pose_trans_mask = torch.ones(
            (batch_size_per_view * num_views), dtype=torch.bool, device=device
        )
        for view_idx in range(num_views):
            # if "is_metric_scale" in views[view_idx]:
            #     # Get the metric scale mask for the input pose priors 
            #     metric_scale_mask = views[view_idx]["is_metric_scale"]
            # else:
            metric_scale_mask = torch.zeros(
                batch_size_per_view, dtype=torch.bool, device=device
            )
            metric_scale_pose_trans_mask[
                view_idx * batch_size_per_view : (view_idx + 1) * batch_size_per_view
            ] = metric_scale_mask

        # Turn off indication of metric scale samples based on the pose_scale_norm_all_prob
        # pose_norm_all_mask = (
        #     torch.rand(batch_size_per_view * num_views)
        #     < self.geometric_input_config["pose_scale_norm_all_prob"]
        # )
        # if pose_norm_all_mask.any():
        #     metric_scale_pose_trans_mask[pose_norm_all_mask] = False

        # Get the scale norm factor for all the samples and scale the pose translations
        pose_trans_across_views = torch.split(
            pose_trans_across_views, batch_size_per_view, dim=0
        )  # Split into num_views chunks
        pose_trans_across_views = torch.stack(
            pose_trans_across_views, dim=1
        )  # Stack the views along a new dimension (batch_size_per_view, num_views, 3)
        scaled_pose_trans_across_views, pose_trans_norm_factors = (
            normalize_pose_translations(
                pose_trans_across_views, return_norm_factor=True
            )
        )

        # Resize the pose translation back to (batch_size_per_view * num_views, 3) and extend the norm factor to (batch_size_per_view * num_views, 1)
        scaled_pose_trans_across_views = scaled_pose_trans_across_views.unbind(
            dim=1
        )  # Convert back to list of views, where each view has batch_size_per_view tensor
        scaled_pose_trans_across_views = torch.cat(
            scaled_pose_trans_across_views, dim=0
        )  # Concatenate back to (batch_size_per_view * num_views, 3)
        pose_trans_norm_factors_across_views = pose_trans_norm_factors.unsqueeze(
            -1
        ).repeat(num_views, 1)  # (B, ) -> (B * V, 1)

        # Encode the pose trans
        pose_trans_features_across_views = self.cam_trans_encoder(
            EncoderGlobalRepInput(data=scaled_pose_trans_across_views)
        ).features
        # Zero out the pose trans features where the camera input mask is False
        pose_trans_features_across_views = (
            pose_trans_features_across_views * per_sample_cam_input_mask.unsqueeze(-1)
        )

        # Encode the pose translation norm factors using the log scale encoder for pose trans
        log_pose_trans_norm_factors_across_views = torch.log(
            pose_trans_norm_factors_across_views + 1e-8
        )
        pose_trans_scale_features_across_views = self.cam_trans_scale_encoder(
            EncoderGlobalRepInput(data=log_pose_trans_norm_factors_across_views)
        ).features
        # Zero out the pose trans scale features where the camera input mask is False
        pose_trans_scale_features_across_views = (
            pose_trans_scale_features_across_views
            * per_sample_cam_input_mask.unsqueeze(-1)
        )
        # Zero out the pose trans scale features where the metric scale mask is False
        # Scale encoding is only provided for metric scale samples
        pose_trans_scale_features_across_views = (
            pose_trans_scale_features_across_views
            * metric_scale_pose_trans_mask.unsqueeze(-1)
        )

        # Fuse the pose quat features, pose trans features, pose trans scale features and pose trans type PE features with the other encoder features
        all_encoder_features_across_views = (
            all_encoder_features_across_views
            + pose_quats_features_across_views.unsqueeze(-1).unsqueeze(-1)
            + pose_trans_features_across_views.unsqueeze(-1).unsqueeze(-1)
            + pose_trans_scale_features_across_views.unsqueeze(-1).unsqueeze(-1)
        )

        return all_encoder_features_across_views

    def _encode_and_fuse_time(
        self,
        num_views,
        batch_size_per_view,
        all_encoder_features_across_views,
        per_sample_time_input_mask,
    ):
        """
        Encode the time values for all the views and fuse it with the other encoder features in a single forward pass.
        Time values are linearly interpolated from 0 to 1 based on view index.
        
        If use_fourier_time_embedding is True, time values are passed through a Fourier time embedding before being encoded.
        Otherwise, original time values are directly encoded (backward compatibility).

        Args:
            num_views (int): Number of views.
            batch_size_per_view (int): Batch size per view.
            all_encoder_features_across_views (torch.Tensor): Tensor containing the encoded features for all N views.
            per_sample_time_input_mask (torch.Tensor): Tensor containing the per sample time input mask.

        Returns:
            torch.Tensor: A tensor containing the encoded features for all the views.
        """
        # Skip if time encoder is not initialized
        if self.time_encoder is None:
            return all_encoder_features_across_views
        
        # Get device and dtype from input features
        device = all_encoder_features_across_views.device
        dtype = all_encoder_features_across_views.dtype

        # Generate time values from 0 to 1 based on number of views
        # time_values shape: (num_views,)
        time_values = torch.linspace(0, 1, num_views, device=device, dtype=dtype)
        
        # Create time tensor for all views
        # Each view gets its corresponding time value repeated for all samples in the batch
        time_list = []
        for view_idx in range(num_views):
            time_for_curr_view = time_values[view_idx].unsqueeze(0).repeat(batch_size_per_view, 1)  # (B, 1)
            time_list.append(time_for_curr_view)
        
        # Stack the time values for all views: (B * V, 1)
        times = torch.cat(time_list, dim=0)
        
        # Apply Fourier time embedding if enabled, otherwise use original time values
        if self.use_fourier_time_embedding:
            # Apply Fourier time embedding to transform time values
            # Input: (B * V, 1) -> Output: (B * V, 2*num_frequencies + 1)
            time_input = self.fourier_time_embedding(times)
        else:
            # Use original time values directly (backward compatibility)
            # Input: (B * V, 1) -> Output: (B * V, 1)
            time_input = times
        
        # Encode the time values (with or without Fourier embedding) using the time encoder
        time_features_across_views = self.time_encoder(
            EncoderGlobalRepInput(data=time_input)
        ).features
        
        # Zero out the time features where the time input mask is False
        time_features_across_views = (
            time_features_across_views * per_sample_time_input_mask.unsqueeze(-1)
        )
        
        # Fuse the time features with the other encoder features
        all_encoder_features_across_views = (
            all_encoder_features_across_views
            + time_features_across_views.unsqueeze(-1).unsqueeze(-1)
        )
        
        return all_encoder_features_across_views

    def _encode_and_fuse_rgb(
        self,
        views,
        num_views,
        batch_size_per_view,
        all_encoder_features_across_views,
        per_sample_rgb_input_mask,
    ):
        """
        Encode the RGB values for all the views and fuse it with the other encoder features in a single forward pass.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
            num_views (int): Number of views.
            batch_size_per_view (int): Batch size per view.
            all_encoder_features_across_views (torch.Tensor): Tensor containing the encoded features for all N views.
            per_sample_rgb_input_mask (torch.Tensor): Tensor containing the per sample RGB input mask.

        Returns:
            torch.Tensor: A tensor containing the encoded features for all the views.
        """
        # Skip if rgb encoder is not initialized
        if self.rgb_encoder is None:
            return all_encoder_features_across_views
        
        # Get the height and width of the images
        _, _, height, width = views[0]["img"].shape

        # Get the RGB values for all the views where info is provided and the RGB input mask is True
        rgb_list = []
        for view_idx in range(num_views):
            per_sample_rgb_input_mask_for_curr_view = (
                per_sample_rgb_input_mask[
                    view_idx * batch_size_per_view : (view_idx + 1)
                    * batch_size_per_view
                ]
            )
            # img is already in [B, C, H, W] format, so we initialize rgb_for_curr_view in the same format
            rgb_for_curr_view = torch.zeros(
                (batch_size_per_view, 3, height, width),  # [B, C, H, W]
                dtype=all_encoder_features_across_views.dtype,
                device=all_encoder_features_across_views.device,
            )
            if (
                "img" in views[view_idx]
                and per_sample_rgb_input_mask_for_curr_view.any()
            ):
                # Use the image directly as RGB input (already in [B, C, H, W] format)
                # No permute needed unlike ray_directions_cam which is [B, H, W, C]
                rgb_for_curr_view[per_sample_rgb_input_mask_for_curr_view] = (
                    views[view_idx]["img"][
                        per_sample_rgb_input_mask_for_curr_view
                    ]
                )
            else:
                per_sample_rgb_input_mask[
                    view_idx * batch_size_per_view : (view_idx + 1)
                    * batch_size_per_view
                ] = False
            rgb_list.append(rgb_for_curr_view)

        # Stack the RGB values for all the views: (B * V, C, H, W)
        # Note: Unlike ray_dirs which needs permute, img is already in [B, C, H, W] format
        rgb = torch.cat(rgb_list, dim=0)  # (B * V, 3, H, W)

        # Encode the RGB values
        rgb_features_across_views = self.rgb_encoder(
            ViTEncoderNonImageInput(data=rgb)
        ).features

        # Fuse the RGB features with the other encoder features (zero out the features where the RGB input mask is False)
        rgb_features_across_views = (
            rgb_features_across_views
            * per_sample_rgb_input_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )
        all_encoder_features_across_views = (
            all_encoder_features_across_views + rgb_features_across_views
        )

        return all_encoder_features_across_views

    def _encode_and_fuse_optional_geometric_inputs(
        self, views, all_encoder_features_across_views_list
    ):
        """
        Encode all the input optional geometric modalities and fuses it with the image encoder features in a single forward pass.
        Assumes all the input views have the same shape and batch size.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
            all_encoder_features_across_views (List[torch.Tensor]): List of tensors containing the encoded image features for all N views.

        Returns:
            List[torch.Tensor]: A list containing the encoded features for all N views.
        """
        num_views = len(views)
        batch_size_per_view, _, _, _ = views[0]["img"].shape
        device = all_encoder_features_across_views_list[0].device
        dtype = all_encoder_features_across_views_list[0].dtype
        all_encoder_features_across_views = torch.cat(
            all_encoder_features_across_views_list, dim=0
        )

        # Get the overall input mask for all the views
        overall_geometric_input_mask = torch.ones(
            batch_size_per_view, device=device, dtype=torch.bool
        )
        overall_geometric_input_mask = overall_geometric_input_mask.repeat(num_views)

        # Get the per sample input mask after dropout
        # Per sample input mask is in view-major order so that index v*B + b in each mask corresponds to sample b of view v: (B * V)
        per_sample_geometric_input_mask = torch.ones(
            batch_size_per_view * num_views, device=device, dtype=torch.bool
        )
        per_sample_geometric_input_mask = (
            per_sample_geometric_input_mask & overall_geometric_input_mask
        )

        # Get the ray direction input mask (disabled when ray_dirs_prob == 0)
        _ray_dirs_enabled = self.geometric_input_config.get("ray_dirs_prob", 1.0) != 0
        per_sample_ray_dirs_input_mask = torch.ones(
            batch_size_per_view, device=device, dtype=torch.bool
        ) if _ray_dirs_enabled else torch.zeros(
            batch_size_per_view, device=device, dtype=torch.bool
        )
        per_sample_ray_dirs_input_mask = per_sample_ray_dirs_input_mask.repeat(
            num_views
        )
        per_sample_ray_dirs_input_mask = (
            per_sample_ray_dirs_input_mask & per_sample_geometric_input_mask
        )

        # Get the depth input mask
        per_sample_depth_input_mask = torch.zeros(
            batch_size_per_view, device=device, dtype=torch.bool
        )
        per_sample_depth_input_mask = per_sample_depth_input_mask.repeat(num_views)
        per_sample_depth_input_mask = (
            per_sample_depth_input_mask & per_sample_geometric_input_mask
        )

        # Get the camera input mask (disabled when cam_prob == 0)
        _cam_enabled = self.geometric_input_config.get("cam_prob", 1.0) != 0
        per_sample_cam_input_mask = torch.ones(
            batch_size_per_view, device=device, dtype=torch.bool
        ) if _cam_enabled else torch.zeros(
            batch_size_per_view, device=device, dtype=torch.bool
        )
        per_sample_cam_input_mask = per_sample_cam_input_mask.repeat(num_views)
        per_sample_cam_input_mask = (
            per_sample_cam_input_mask & per_sample_geometric_input_mask
        )

        # Get the time input mask
        per_sample_time_input_mask = torch.ones(
            batch_size_per_view, device=device, dtype=torch.bool
        )
        per_sample_time_input_mask = per_sample_time_input_mask.repeat(num_views)
        per_sample_time_input_mask = (
            per_sample_time_input_mask & per_sample_geometric_input_mask
        )

        # Get the RGB input mask
        per_sample_rgb_input_mask = torch.ones(
            batch_size_per_view, device=device, dtype=torch.bool
        )
        per_sample_rgb_input_mask = per_sample_rgb_input_mask.repeat(num_views)
        per_sample_rgb_input_mask = (
            per_sample_rgb_input_mask & per_sample_geometric_input_mask
        )

        # Compute the pose quats and trans for all the non-reference views in the frame of the reference view 0
        # Returned pose quats and trans represent identity pose for views/samples where the camera input mask is False
        pose_quats_across_views, pose_trans_across_views, per_sample_cam_input_mask = (
            self._compute_pose_quats_and_trans_for_across_views_in_ref_view(
                views,
                num_views,
                device,
                dtype,
                batch_size_per_view,
                per_sample_cam_input_mask,
            )
        )

        # Encode the ray directions and fuse with the image encoder features
        all_encoder_features_across_views = self._encode_and_fuse_ray_dirs(
            views,
            num_views,
            batch_size_per_view,
            all_encoder_features_across_views,
            per_sample_ray_dirs_input_mask,
        )

        # Encode the depths and fuse with the image encoder features
        # all_encoder_features_across_views = self._encode_and_fuse_depths(
        #     views,
        #     num_views,
        #     batch_size_per_view,
        #     all_encoder_features_across_views,
        #     per_sample_depth_input_mask,
        # )

        # Encode the cam quat and trans and fuse with the image encoder features
        all_encoder_features_across_views = self._encode_and_fuse_cam_quats_and_trans(
            views,
            num_views,
            batch_size_per_view,
            all_encoder_features_across_views,
            pose_quats_across_views,
            pose_trans_across_views,
            per_sample_cam_input_mask,
        )

        # Encode the time values and fuse with the image encoder features
        all_encoder_features_across_views = self._encode_and_fuse_time(
            num_views,
            batch_size_per_view,
            all_encoder_features_across_views,
            per_sample_time_input_mask,
        )

        # Encode the RGB values and fuse with the image encoder features
        all_encoder_features_across_views = self._encode_and_fuse_rgb(
            views,
            num_views,
            batch_size_per_view,
            all_encoder_features_across_views,
            per_sample_rgb_input_mask,
        )

        # Normalize the fused features (permute -> normalize -> permute)
        all_encoder_features_across_views = all_encoder_features_across_views.permute(
            0, 2, 3, 1
        ).contiguous()
        all_encoder_features_across_views = self.fusion_norm_layer(
            all_encoder_features_across_views
        )
        all_encoder_features_across_views = all_encoder_features_across_views.permute(
            0, 3, 1, 2
        ).contiguous()

        # Split the batched views into individual views
        fused_all_encoder_features_across_views = (
            all_encoder_features_across_views.chunk(num_views, dim=0)
        )

        return fused_all_encoder_features_across_views

    def _compute_adaptive_minibatch_size(
        self,
        memory_safety_factor: float = 0.95,
    ) -> int:
        """
        Compute adaptive minibatch size based on available PyTorch memory.

        Args:
            memory_safety_factor: Safety factor to avoid OOM (0.95 = use 95% of available memory)

        Returns:
            Computed minibatch size
        """
        override = getattr(self, '_dense_head_minibatch_override', None)
        if override is not None and override > 0:
            return override

        device = self.device

        if device.type == "cuda":
            torch.cuda.empty_cache()
            available_memory = torch.cuda.mem_get_info()[0]
            usable_memory = available_memory * memory_safety_factor
        else:
            print(
                "Non-CUDA device detected. Using conservative default minibatch size of 1 for memory efficient dense prediction head inference."
            )
            return 1

        max_estimated_memory_per_sample = (
            680 * 1024 * 1024
        )  # 680 MB per sample (upper bound profiling using a 518 x 518 input)

        # Separate DPT heads run sequentially but outputs accumulate;
        # scale estimate by number of GS heads to keep peak within budget.
        if getattr(self, 'use_separate_dpt_heads', False):
            n_gs = len(getattr(self, 'gs_params', []))
            max_estimated_memory_per_sample = int(
                max_estimated_memory_per_sample * max(n_gs, 2)
            )

        computed_minibatch_size = max(1, int(usable_memory / max_estimated_memory_per_sample))

        return computed_minibatch_size

    def downstream_dense_head(
        self,
        dense_head_inputs: Union[torch.Tensor, List[torch.Tensor]],
        img_shape: Tuple[int, int],
        imgs_for_adapter: torch.Tensor = None,
    ):
        """
        Run the downstream dense prediction head
        
        Args:
            dense_head_inputs: Input features for dense head
            img_shape: Output image shape (H, W)
            imgs_for_adapter: Input images for image adapter, shape (B*T, C, H, W)
        """
        if self.pred_head_type == "linear":
            dense_head_outputs = self.dense_head(
                PredictionHeadInput(last_feature=dense_head_inputs)
            )
            dense_final_outputs = self.dense_adaptor(
                AdaptorInput(
                    adaptor_feature=dense_head_outputs.decoded_channels,
                    output_shape_hw=img_shape,
                )
            )
        elif self.pred_head_type in ["dpt", "dpt+pose"]:
            dense_head_outputs = self.dense_head(
                PredictionHeadLayeredInput(
                    list_features=dense_head_inputs,
                    target_output_shape=img_shape,
                )
            )
            # print("dense_head_outputs", dense_head_outputs.decoded_channels.shape)
            dense_final_outputs = self.dense_adaptor(
                AdaptorInput(
                    adaptor_feature=dense_head_outputs.decoded_channels,
                    output_shape_hw=img_shape,
                )
            )
            # print("dense_final_outputs", dense_final_outputs.value.shape)
            # print("dense_final_outputs confidence", dense_final_outputs.confidence.shape)
            # print("dense_final_outputs mask", dense_final_outputs.mask.shape)

            if self.use_separate_dpt_heads:
                # 在 separate 模式下，为每个参数分别调用对应的 DPT head
                gs_param_outputs = {}
                for param_name in self.gs_params:
                    param_output = self.dense_heads_gs[param_name](
                        PredictionHeadLayeredInput(
                            list_features=dense_head_inputs,
                            target_output_shape=img_shape,
                        )
                    )
                    gs_param_outputs[param_name] = param_output.decoded_channels
                
                # Apply image adapters if enabled
                if self.use_image_adapter and self.image_adapters is not None and imgs_for_adapter is not None:
                    gs_param_outputs = self._apply_image_adapters(
                        gs_param_outputs, imgs_for_adapter
                    )
                
                # 将所有参数输出组合成一个字典，作为 dense_head_outputs_gs 返回
                # 创建一个包含所有参数的对象
                class GSParamOutputs:
                    def __init__(self, param_outputs_dict):
                        self.decoded_channels = param_outputs_dict
                
                dense_head_outputs_gs = GSParamOutputs(gs_param_outputs)
            else:
                dense_head_outputs_gs = self.dense_head_gs(
                    PredictionHeadLayeredInput(
                        list_features=dense_head_inputs,
                        target_output_shape=img_shape,
                    )
                )
            # print("dense_head_outputs_gs", dense_head_outputs_gs.decoded_channels.shape)
            
            # # 适配形状: [B, C, H, W] -> [B, H, W, C]
            # gs_features = dense_head_outputs_gs.decoded_channels.permute(0, 2, 3, 1).contiguous()
            # print("gs_features after permute", gs_features.shape)
            
            # # 应用 gs_adaptor
            # dense_final_outputs_gs = self.gs_adaptor(gs_features)
            # print("dense_final_outputs_gs after adaptor", dense_final_outputs_gs.shape)
            
            # # 还原形状: [B, H, W, C] -> [B, C, H, W]
            # dense_final_outputs_gs = dense_final_outputs_gs.permute(0, 3, 1, 2).contiguous()
            # print("dense_final_outputs_gs after permute back", dense_final_outputs_gs.shape)

        else:
            raise ValueError(
                f"Invalid pred_head_type: {self.pred_head_type}. Valid options: ['linear', 'dpt', 'dpt+pose']"
            )

        return dense_final_outputs, dense_head_outputs_gs
    
    def _apply_image_adapters(
        self,
        gs_param_outputs: dict,
        imgs: torch.Tensor,
    ) -> dict:
        """
        Apply image adapters to enhance Gaussian parameters with direct image features.
        
        Structure:
            if share_input_merger:
                shared_img_feat = shared_input_merger(imgs)  # 所有参数共享
                for each enabled param:
                    out = output_conv(gs_param_output + shared_img_feat)
            else:
                for each enabled param:
                    img_feat = per_param_input_merger[param](imgs)  # 每个参数独立
                    out = output_conv(gs_param_output + img_feat)
        
        Args:
            gs_param_outputs: Dictionary of Gaussian parameter outputs {param_name: tensor (B*T, C, H, W)}
            imgs: Input images of shape (B*T, C, H, W)
            
        Returns:
            Enhanced gs_param_outputs dictionary
        """
        # Map param names from gs_params to adapter keys
        param_name_mapping = {
            "opacity": "opacity",
            "scale": "scale", 
            "rotation": "rotation",
            "rgb": "rgb",
        }
        
        if self.share_input_merger:
            # 共享模式：使用共享的 input_merger 计算图像特征 (只计算一次！)
            shared_img_feat = self.shared_input_merger(imgs)  # (B*T, adapter_hidden_dim, H, W)
            
            for gs_param_name, adapter_key in param_name_mapping.items():
                if adapter_key in self.image_adapter_output_convs and gs_param_name in gs_param_outputs:
                    output_conv = self.image_adapter_output_convs[adapter_key]
                    
                    # Get current gs_param output (shape: B*T, adapter_hidden_dim, H, W)
                    gs_param_output = gs_param_outputs[gs_param_name]
                    
                    # 融合: gs_param_output + shared_img_feat
                    combined_feat = gs_param_output + shared_img_feat
                    
                    # Apply output_conv to get final dimension
                    enhanced_output = output_conv(combined_feat)
                    
                    # Replace with enhanced output
                    gs_param_outputs[gs_param_name] = enhanced_output
        else:
            # 独立模式：每个参数使用独立的 input_merger
            for gs_param_name, adapter_key in param_name_mapping.items():
                if adapter_key in self.image_adapter_output_convs and gs_param_name in gs_param_outputs:
                    # 获取该参数对应的 input_merger 和 output_conv
                    input_merger = self.per_param_input_mergers[adapter_key]
                    output_conv = self.image_adapter_output_convs[adapter_key]
                    
                    # 计算该参数专属的图像特征
                    img_feat = input_merger(imgs)  # (B*T, adapter_hidden_dim, H, W)
                    
                    # Get current gs_param output (shape: B*T, adapter_hidden_dim, H, W)
                    gs_param_output = gs_param_outputs[gs_param_name]
                    
                    # 融合: gs_param_output + img_feat
                    combined_feat = gs_param_output + img_feat
                    
                    # Apply output_conv to get final dimension
                    enhanced_output = output_conv(combined_feat)
                    
                    # Replace with enhanced output
                    gs_param_outputs[gs_param_name] = enhanced_output
        
        return gs_param_outputs

    def downstream_head(
        self,
        dense_head_inputs: Union[torch.Tensor, List[torch.Tensor]],
        scale_head_inputs: torch.Tensor,
        img_shape: Tuple[int, int],
        memory_efficient_inference: bool = False,
        imgs_for_adapter: torch.Tensor = None,
    ):
        """
        Run Prediction Heads & Post-Process Outputs
        
        Args:
            dense_head_inputs: Input features for dense head
            scale_head_inputs: Input features for scale head
            img_shape: Output image shape (H, W)
            memory_efficient_inference: Whether to use memory efficient inference
            imgs_for_adapter: Input images for image adapter, shape (B*T, C, H, W)
        """
        # Get device
        device = self.device

        # Use mini-batch inference to run the dense prediction head (the memory bottleneck)
        # This saves memory and is slower than running the dense prediction head in one go
        if memory_efficient_inference:
            # Obtain the batch size of the dense head inputs
            if self.pred_head_type == "linear":
                batch_size = dense_head_inputs.shape[0]
            elif self.pred_head_type in ["dpt", "dpt+pose"]:
                batch_size = dense_head_inputs[0].shape[0]
            else:
                raise ValueError(
                    f"Invalid pred_head_type: {self.pred_head_type}. Valid options: ['linear', 'dpt', 'dpt+pose']"
                )

            # Compute the mini batch size and number of mini batches adaptively based on available memory
            minibatch = self._compute_adaptive_minibatch_size()
            num_batches = (batch_size + minibatch - 1) // minibatch

            # Run prediction for each mini-batch
            dense_final_outputs_list = []
            dense_final_outputs_gs_list = []
            pose_final_outputs_list = [] if self.pred_head_type == "dpt+pose" else None
            print(f"[MemEfficient] Dense head mini-batch: size={minibatch}, "
                  f"num_batches={num_batches}, total_views={batch_size}")

            for batch_idx in range(num_batches):
                start_idx = batch_idx * minibatch
                end_idx = min((batch_idx + 1) * minibatch, batch_size)

                # Get the inputs for the current mini-batch
                if self.pred_head_type == "linear":
                    dense_head_inputs_batch = dense_head_inputs[start_idx:end_idx]
                elif self.pred_head_type in ["dpt", "dpt+pose"]:
                    dense_head_inputs_batch = [
                        x[start_idx:end_idx] for x in dense_head_inputs
                    ]
                else:
                    raise ValueError(
                        f"Invalid pred_head_type: {self.pred_head_type}. Valid options: ['linear', 'dpt', 'dpt+pose']"
                    )

                # Dense prediction (mini-batched)
                imgs_batch = imgs_for_adapter[start_idx:end_idx] if imgs_for_adapter is not None else None
                dense_final_outputs_batch, dense_final_outputs_gs_batch = self.downstream_dense_head(
                    dense_head_inputs_batch, img_shape, imgs_for_adapter=imgs_batch
                )
                dense_final_outputs_list.append(dense_final_outputs_batch)
                dense_final_outputs_gs_list.append(dense_final_outputs_gs_batch)

                # Pose prediction (mini-batched)
                if self.pred_head_type == "dpt+pose":
                    pose_head_inputs_batch = dense_head_inputs[-1][start_idx:end_idx]
                    pose_head_outputs_batch = self.pose_head(
                        PredictionHeadInput(last_feature=pose_head_inputs_batch)
                    )
                    pose_final_outputs_batch = self.pose_adaptor(
                        AdaptorInput(
                            adaptor_feature=pose_head_outputs_batch.decoded_channels,
                            output_shape_hw=img_shape,
                        )
                    )
                    pose_final_outputs_list.append(pose_final_outputs_batch)

                # Release intermediate memory between mini-batches
                if device.type == "cuda" and batch_idx < num_batches - 1:
                    torch.cuda.empty_cache()

            # Concatenate the dense prediction head outputs from all mini-batches
            available_keys = dense_final_outputs_batch.__dict__.keys()
            dense_pred_data_dict = {
                key: torch.cat(
                    [getattr(output, key) for output in dense_final_outputs_list], dim=0
                )
                for key in available_keys
            }
            dense_final_outputs = dense_final_outputs_batch.__class__(
                **dense_pred_data_dict
            )

            # Concatenate the GS prediction head outputs from all mini-batches
            first_gs_channels = dense_final_outputs_gs_list[0].decoded_channels
            if isinstance(first_gs_channels, dict):
                merged_gs_channels = {
                    k: torch.cat(
                        [gs.decoded_channels[k] for gs in dense_final_outputs_gs_list],
                        dim=0,
                    )
                    for k in first_gs_channels
                }

                class _MergedGSOutput:
                    def __init__(self, channels_dict):
                        self.decoded_channels = channels_dict

                dense_final_outputs_gs = _MergedGSOutput(merged_gs_channels)
            else:
                merged_gs_channels = torch.cat(
                    [gs.decoded_channels for gs in dense_final_outputs_gs_list],
                    dim=0,
                )
                dense_final_outputs_gs = dense_final_outputs_gs_list[0].__class__(
                    decoded_channels=merged_gs_channels
                )

            # Concatenate the pose prediction head outputs from all mini-batches
            pose_final_outputs = None
            if self.pred_head_type == "dpt+pose":
                available_keys = pose_final_outputs_batch.__dict__.keys()
                pose_pred_data_dict = {
                    key: torch.cat(
                        [getattr(output, key) for output in pose_final_outputs_list],
                        dim=0,
                    )
                    for key in available_keys
                }
                pose_final_outputs = pose_final_outputs_batch.__class__(
                    **pose_pred_data_dict
                )

            # Free per-batch lists and clear CUDA cache
            del dense_final_outputs_list, dense_final_outputs_gs_list
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            # Run prediction for all (batch_size * num_views) in one go
            # Dense prediction
            dense_final_outputs, dense_final_outputs_gs = self.downstream_dense_head(
                dense_head_inputs, img_shape, imgs_for_adapter=imgs_for_adapter
            )

            # Pose prediction
            pose_final_outputs = None
            if self.pred_head_type == "dpt+pose":
                pose_head_outputs = self.pose_head(
                    PredictionHeadInput(last_feature=dense_head_inputs[-1])
                )
                # print("pose_head_outputs", pose_head_outputs.decoded_channels.shape)
                pose_final_outputs = self.pose_adaptor(
                    AdaptorInput(
                        adaptor_feature=pose_head_outputs.decoded_channels,
                        output_shape_hw=img_shape,
                    )
                )
                # print("pose_final_outputs", pose_final_outputs.value.shape)

        # Scale prediction is lightweight, so we can run it in one go
        scale_head_output = self.scale_head(
            PredictionHeadTokenInput(last_feature=scale_head_inputs)
        )
        scale_final_output = self.scale_adaptor(
            AdaptorInput(
                adaptor_feature=scale_head_output.decoded_channels,
                output_shape_hw=img_shape,
            )
        )
        scale_final_output = scale_final_output.value.squeeze(-1)  # (B, 1, 1) -> (B, 1)

        # Clear CUDA cache for better memory efficiency
        if memory_efficient_inference and device.type == "cuda":
            torch.cuda.empty_cache()

        return dense_final_outputs, dense_final_outputs_gs, pose_final_outputs, scale_final_output

    def forward(self, views, memory_efficient_inference=False, use_gt_camera=True, use_gt_intrinsics=True):
        """
        Forward pass performing the following operations:
        1. Encodes the N input views (images).
        2. Encodes the optional geometric inputs (ray directions, depths, camera rotations, camera translations).
        3. Fuses the encoded features from the N input views and the optional geometric inputs using addition and normalization.
        4. Information sharing across the encoded features and a scale token using a multi-view attention transformer.
        5. Passes the final features from transformer through the prediction heads.
        6. Returns the processed final outputs for N views.

        Assumption:
        - All the input views and dense geometric inputs have the same image shape.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
                                Each dictionary should contain the following keys:
                                    "img" (tensor): Image tensor of shape (B, C, H, W). Input images must be normalized based on the data norm type of image encoder.
                                    "data_norm_type" (list): [model.encoder.data_norm_type]
                                Optionally, each dictionary can also contain the following keys for the respective optional geometric inputs:
                                    "ray_directions_cam" (tensor): Ray directions in the local camera frame. Tensor of shape (B, H, W, 3).
                                    "depth_along_ray" (tensor): Depth along the ray. Tensor of shape (B, H, W, 1).
                                    "camera_pose_quats" (tensor): Camera pose quaternions. Tensor of shape (B, 4). Camera pose is opencv (RDF) cam2world transformation.
                                    "camera_pose_trans" (tensor): Camera pose translations. Tensor of shape (B, 3). Camera pose is opencv (RDF) cam2world transformation.
                                    "is_metric_scale" (tensor): Boolean tensor indicating whether the geometric inputs are in metric scale or not. Tensor of shape (B, 1).
            memory_efficient_inference (bool): Whether to use memory efficient inference or not. This runs the dense prediction head (the memory bottleneck) in a memory efficient manner. Default is False.

        Returns:
            List[dict]: A list containing the final outputs for all N views.
        """
        # Get input shape of the images, number of views, and batch size per view
        batch_size_per_view, _, height, width = views[0]["img"].shape
        img_shape = (int(height), int(width))
        num_views = len(views)
        # print("batch_size_per_view", batch_size_per_view)
        # print("num_views", num_views)
        # print("image shape", img_shape)

        # Run the image encoder on all the input views
        all_encoder_features_across_views = self._encode_n_views(views)

        # Encode the optional geometric inputs and fuse with the encoded features from the N input views
        # Use high precision to prevent NaN values after layer norm in dense representation encoder (due to high variance in last dim of features)
        with torch.autocast("cuda", enabled=False):
            all_encoder_features_across_views = (
                self._encode_and_fuse_optional_geometric_inputs(
                    views, all_encoder_features_across_views
                )
            )

        # Expand the scale token to match the batch size
        input_scale_token = (
            self.scale_token.unsqueeze(0)
            .unsqueeze(-1)
            .repeat(batch_size_per_view, 1, 1)
        )  # (B, C, 1)

        # Combine all images into view-centric representation
        # Output is a list containing the encoded features for all N views after information sharing.
        info_sharing_input = MultiViewTransformerInput(
            features=all_encoder_features_across_views,
            additional_input_tokens=input_scale_token,
        )
        if self.info_sharing_return_type == "no_intermediate_features":
            final_info_sharing_multi_view_feat = self.info_sharing(info_sharing_input)
        elif self.info_sharing_return_type == "intermediate_features":
            (
                final_info_sharing_multi_view_feat,
                intermediate_info_sharing_multi_view_feat,
            ) = self.info_sharing(info_sharing_input)
        # print("final_info_sharing_multi_view_feat", final_info_sharing_multi_view_feat.features[0].shape)
        if self.pred_head_type == "linear":
            # Stack the features for all views
            dense_head_inputs = torch.cat(
                final_info_sharing_multi_view_feat.features, dim=0
            )
        elif self.pred_head_type in ["dpt", "dpt+pose"]:
            # Get the list of features for all views
            dense_head_inputs_list = []
            if self.use_encoder_features_for_dpt:
                # Stack all the image encoder features for all views
                stacked_encoder_features = torch.cat(
                    all_encoder_features_across_views, dim=0
                )
                # print("stacked_encoder_features", stacked_encoder_features.shape)
                dense_head_inputs_list.append(stacked_encoder_features)
                # Stack the first intermediate features for all views
                stacked_intermediate_features_1 = torch.cat(
                    intermediate_info_sharing_multi_view_feat[0].features, dim=0
                )
                # print("stacked_intermediate_features_1", stacked_intermediate_features_1.shape)
                dense_head_inputs_list.append(stacked_intermediate_features_1)
                # Stack the second intermediate features for all views
                stacked_intermediate_features_2 = torch.cat(
                    intermediate_info_sharing_multi_view_feat[1].features, dim=0
                )
                # print("stacked_intermediate_features_2", stacked_intermediate_features_2.shape)
                dense_head_inputs_list.append(stacked_intermediate_features_2)
                # Stack the last layer features for all views
                stacked_final_features = torch.cat(
                    final_info_sharing_multi_view_feat.features, dim=0
                )
                # print("stacked_final_features", stacked_final_features.shape)
                dense_head_inputs_list.append(stacked_final_features)
            else:
                # Stack the first intermediate features for all views
                stacked_intermediate_features_1 = torch.cat(
                    intermediate_info_sharing_multi_view_feat[0].features, dim=0
                )
                dense_head_inputs_list.append(stacked_intermediate_features_1)
                # Stack the second intermediate features for all views
                stacked_intermediate_features_2 = torch.cat(
                    intermediate_info_sharing_multi_view_feat[1].features, dim=0
                )
                dense_head_inputs_list.append(stacked_intermediate_features_2)
                # Stack the third intermediate features for all views
                stacked_intermediate_features_3 = torch.cat(
                    intermediate_info_sharing_multi_view_feat[2].features, dim=0
                )
                dense_head_inputs_list.append(stacked_intermediate_features_3)
                # Stack the last layer
                stacked_final_features = torch.cat(
                    final_info_sharing_multi_view_feat.features, dim=0
                )
                dense_head_inputs_list.append(stacked_final_features)
        else:
            raise ValueError(
                f"Invalid pred_head_type: {self.pred_head_type}. Valid options: ['linear', 'dpt', 'dpt+pose']"
            )

        with torch.autocast("cuda", enabled=False):
            # Prepare inputs for the downstream heads
            if self.pred_head_type == "linear":
                dense_head_inputs = dense_head_inputs
            elif self.pred_head_type in ["dpt", "dpt+pose"]:
                dense_head_inputs = dense_head_inputs_list
            scale_head_inputs = (
                final_info_sharing_multi_view_feat.additional_token_features
            )

            # Prepare images for image adapter (if enabled)
            imgs_for_adapter = None
            if self.use_image_adapter and self.image_adapters is not None:
                # Stack images from all views: (B*T, C, H, W)
                imgs_list = [view["img"] for view in views]
                imgs_for_adapter = torch.cat(imgs_list, dim=0)

            # Run the downstream heads
            dense_final_outputs, dense_final_outputs_gs, pose_final_outputs, scale_final_output = (
                self.downstream_head(
                    dense_head_inputs=dense_head_inputs,
                    scale_head_inputs=scale_head_inputs,
                    img_shape=img_shape,
                    memory_efficient_inference=memory_efficient_inference,
                    imgs_for_adapter=imgs_for_adapter,
                )
            )


            # Prepare the final scene representation for all views
            if self.scene_rep_type in [
                "pointmap",
                "pointmap+confidence",
                "pointmap+mask",
                "pointmap+confidence+mask",
            ]:
                output_pts3d = dense_final_outputs.value
                # Reshape final scene representation to (B * V, H, W, C)
                output_pts3d = output_pts3d.permute(0, 2, 3, 1).contiguous()
                # Split the predicted pointmaps back to their respective views
                output_pts3d_per_view = output_pts3d.chunk(num_views, dim=0)
                # Pack the output as a list of dictionaries
                res = []
                for i in range(num_views):
                    res.append(
                        {
                            "pts3d": output_pts3d_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "metric_scaling_factor": scale_final_output,
                        }
                    )
            elif self.scene_rep_type in [
                "raymap+depth",
                "raymap+depth+confidence",
                "raymap+depth+mask",
                "raymap+depth+confidence+mask",
            ]:
                # Reshape final scene representation to (B * V, H, W, C)
                output_scene_rep = dense_final_outputs.value.permute(
                    0, 2, 3, 1
                ).contiguous()
                # Get the predicted ray origins, directions, and depths along rays
                output_ray_origins, output_ray_directions, output_depth_along_ray = (
                    output_scene_rep.split([3, 3, 1], dim=-1)
                )
                # Get the predicted pointmaps
                output_pts3d = (
                    output_ray_origins + output_ray_directions * output_depth_along_ray
                )
                # Split the predicted quantities back to their respective views
                output_ray_origins_per_view = output_ray_origins.chunk(num_views, dim=0)
                output_ray_directions_per_view = output_ray_directions.chunk(
                    num_views, dim=0
                )
                output_depth_along_ray_per_view = output_depth_along_ray.chunk(
                    num_views, dim=0
                )
                output_pts3d_per_view = output_pts3d.chunk(num_views, dim=0)
                # Pack the output as a list of dictionaries
                res = []
                for i in range(num_views):
                    res.append(
                        {
                            "pts3d": output_pts3d_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "ray_origins": output_ray_origins_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "ray_directions": output_ray_directions_per_view[i],
                            "depth_along_ray": output_depth_along_ray_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "metric_scaling_factor": scale_final_output,
                        }
                    )
            elif self.scene_rep_type in [
                "raydirs+depth+pose",
                "raydirs+depth+pose+confidence",
                "raydirs+depth+pose+mask",
                "raydirs+depth+pose+confidence+mask",
            ]:
                # Reshape output dense rep to (B * V, H, W, C)
                output_dense_rep = dense_final_outputs.value.permute(
                    0, 2, 3, 1
                ).contiguous()
                # Get the predicted ray directions and depths along rays
                output_ray_directions, output_depth_along_ray = output_dense_rep.split(
                    [3, 1], dim=-1
                )
                # Get the predicted camera translations and quaternions
                output_cam_translations, output_cam_quats = (
                    pose_final_outputs.value.split([3, 4], dim=-1)
                )
                # Get the predicted pointmaps in world frame and camera frame
                output_pts3d = (
                    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                        output_ray_directions,
                        output_depth_along_ray,
                        output_cam_translations,
                        output_cam_quats,
                    )
                )
                output_pts3d_cam = output_ray_directions * output_depth_along_ray
                # Split the predicted quantities back to their respective views
                output_ray_directions_per_view = output_ray_directions.chunk(
                    num_views, dim=0
                )
                output_depth_along_ray_per_view = output_depth_along_ray.chunk(
                    num_views, dim=0
                )
                output_cam_translations_per_view = output_cam_translations.chunk(
                    num_views, dim=0
                )
                output_cam_quats_per_view = output_cam_quats.chunk(num_views, dim=0)
                output_pts3d_per_view = output_pts3d.chunk(num_views, dim=0)
                output_pts3d_cam_per_view = output_pts3d_cam.chunk(num_views, dim=0)
                # Pack the output as a list of dictionaries
                res = []
                for i in range(num_views):
                    # res.append(
                    #     {
                    #         "pts3d": output_pts3d_per_view[i]
                    #         * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                    #         "pts3d_cam": output_pts3d_cam_per_view[i]
                    #         * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                    #         "ray_directions": output_ray_directions_per_view[i],
                    #         "depth_along_ray": output_depth_along_ray_per_view[i]
                    #         * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                    #         "cam_trans": output_cam_translations_per_view[i]
                    #         * scale_final_output,
                    #         "cam_quats": output_cam_quats_per_view[i],
                    #         "metric_scaling_factor": scale_final_output,
                    #     }
                    # )

                    res.append(
                        {
                            "pts3d": output_pts3d_per_view[i],
                            "pts3d_cam": output_pts3d_cam_per_view[i],
                            "ray_directions": output_ray_directions_per_view[i],
                            "depth_along_ray": output_depth_along_ray_per_view[i],
                            "cam_trans": output_cam_translations_per_view[i],
                            "cam_quats": output_cam_quats_per_view[i],
                            # "metric_scaling_factor": scale_final_output,
                        }
                    )

                gs_params = self.decoded_gs_params_v2(dense_final_outputs_gs, views, res, batch_size_per_view, num_views, height, width, use_gt_camera, use_gt_intrinsics=use_gt_intrinsics)

            elif self.scene_rep_type in [
                "campointmap+pose",
                "campointmap+pose+confidence",
                "campointmap+pose+mask",
                "campointmap+pose+confidence+mask",
            ]:
                # Get the predicted camera frame pointmaps
                output_pts3d_cam = dense_final_outputs.value
                # Reshape final scene representation to (B * V, H, W, C)
                output_pts3d_cam = output_pts3d_cam.permute(0, 2, 3, 1).contiguous()
                # Get the predicted camera translations and quaternions
                output_cam_translations, output_cam_quats = (
                    pose_final_outputs.value.split([3, 4], dim=-1)
                )
                # Get the ray directions and depths along rays
                output_depth_along_ray = torch.norm(
                    output_pts3d_cam, dim=-1, keepdim=True
                )
                output_ray_directions = output_pts3d_cam / output_depth_along_ray
                # Get the predicted pointmaps in world frame
                output_pts3d = (
                    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                        output_ray_directions,
                        output_depth_along_ray,
                        output_cam_translations,
                        output_cam_quats,
                    )
                )
                # Split the predicted quantities back to their respective views
                output_ray_directions_per_view = output_ray_directions.chunk(
                    num_views, dim=0
                )
                output_depth_along_ray_per_view = output_depth_along_ray.chunk(
                    num_views, dim=0
                )
                output_cam_translations_per_view = output_cam_translations.chunk(
                    num_views, dim=0
                )
                output_cam_quats_per_view = output_cam_quats.chunk(num_views, dim=0)
                output_pts3d_per_view = output_pts3d.chunk(num_views, dim=0)
                output_pts3d_cam_per_view = output_pts3d_cam.chunk(num_views, dim=0)
                # Pack the output as a list of dictionaries
                res = []
                for i in range(num_views):
                    res.append(
                        {
                            "pts3d": output_pts3d_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "pts3d_cam": output_pts3d_cam_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "ray_directions": output_ray_directions_per_view[i],
                            "depth_along_ray": output_depth_along_ray_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "cam_trans": output_cam_translations_per_view[i]
                            * scale_final_output,
                            "cam_quats": output_cam_quats_per_view[i],
                            "metric_scaling_factor": scale_final_output,
                        }
                    )
            elif self.scene_rep_type in [
                "pointmap+raydirs+depth+pose",
                "pointmap+raydirs+depth+pose+confidence",
                "pointmap+raydirs+depth+pose+mask",
                "pointmap+raydirs+depth+pose+confidence+mask",
            ]:
                # Reshape final scene representation to (B * V, H, W, C)
                output_dense_rep = dense_final_outputs.value.permute(
                    0, 2, 3, 1
                ).contiguous()
                # Get the predicted pointmaps, ray directions and depths along rays
                output_pts3d, output_ray_directions, output_depth_along_ray = (
                    output_dense_rep.split([3, 3, 1], dim=-1)
                )
                # Get the predicted camera translations and quaternions
                output_cam_translations, output_cam_quats = (
                    pose_final_outputs.value.split([3, 4], dim=-1)
                )
                # Get the predicted pointmaps in camera frame
                output_pts3d_cam = output_ray_directions * output_depth_along_ray
                # Replace the predicted world-frame pointmaps if required
                if self.pred_head_config["adaptor_config"][
                    "use_factored_predictions_for_global_pointmaps"
                ]:
                    output_pts3d = (
                        convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                            output_ray_directions,
                            output_depth_along_ray,
                            output_cam_translations,
                            output_cam_quats,
                        )
                    )
                # Split the predicted quantities back to their respective views
                output_ray_directions_per_view = output_ray_directions.chunk(
                    num_views, dim=0
                )
                output_depth_along_ray_per_view = output_depth_along_ray.chunk(
                    num_views, dim=0
                )
                output_cam_translations_per_view = output_cam_translations.chunk(
                    num_views, dim=0
                )
                output_cam_quats_per_view = output_cam_quats.chunk(num_views, dim=0)
                output_pts3d_per_view = output_pts3d.chunk(num_views, dim=0)
                output_pts3d_cam_per_view = output_pts3d_cam.chunk(num_views, dim=0)
                # Pack the output as a list of dictionaries
                res = []
                for i in range(num_views):
                    res.append(
                        {
                            "pts3d": output_pts3d_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "pts3d_cam": output_pts3d_cam_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "ray_directions": output_ray_directions_per_view[i],
                            "depth_along_ray": output_depth_along_ray_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "cam_trans": output_cam_translations_per_view[i]
                            * scale_final_output,
                            "cam_quats": output_cam_quats_per_view[i],
                            "metric_scaling_factor": scale_final_output,
                        }
                    )
            else:
                raise ValueError(
                    f"Invalid scene_rep_type: {self.scene_rep_type}. \
                    Valid options: ['pointmap', 'raymap+depth', 'raydirs+depth+pose', 'campointmap+pose', 'pointmap+raydirs+depth+pose' \
                                    'pointmap+confidence', 'raymap+depth+confidence', 'raydirs+depth+pose+confidence', 'campointmap+pose+confidence', 'pointmap+raydirs+depth+pose+confidence' \
                                    'pointmap+mask', 'raymap+depth+mask', 'raydirs+depth+pose+mask', 'campointmap+pose+mask', 'pointmap+raydirs+depth+pose+mask' \
                                    'pointmap+confidence+mask', 'raymap+depth+confidence+mask', 'raydirs+depth+pose+confidence+mask', 'campointmap+pose+confidence+mask', 'pointmap+raydirs+depth+pose+confidence+mask']"
                )

            # Get the output confidences for all views (if available) and add them to the result
            if "confidence" in self.scene_rep_type:
                output_confidences = dense_final_outputs.confidence
                # Reshape confidences to (B * V, H, W)
                output_confidences = (
                    output_confidences.permute(0, 2, 3, 1).squeeze(-1).contiguous()
                )
                # Split the predicted confidences back to their respective views
                output_confidences_per_view = output_confidences.chunk(num_views, dim=0)
                # Add the confidences to the result
                for i in range(num_views):
                    res[i]["conf"] = output_confidences_per_view[i]

            # Get the output masks (and logits) for all views (if available) and add them to the result
            if "mask" in self.scene_rep_type:
                # Get the output masks
                output_masks = dense_final_outputs.mask
                # Reshape masks to (B * V, H, W)
                output_masks = output_masks.permute(0, 2, 3, 1).squeeze(-1).contiguous()
                # Threshold the masks at 0.5 to get binary masks (0: ambiguous, 1: non-ambiguous)
                output_masks = output_masks > 0.5
                # Split the predicted masks back to their respective views
                output_masks_per_view = output_masks.chunk(num_views, dim=0)
                # Get the output mask logits (for loss)
                output_mask_logits = dense_final_outputs.logits
                # Reshape mask logits to (B * V, H, W)
                output_mask_logits = (
                    output_mask_logits.permute(0, 2, 3, 1).squeeze(-1).contiguous()
                )
                # Split the predicted mask logits back to their respective views
                output_mask_logits_per_view = output_mask_logits.chunk(num_views, dim=0)
                # Add the masks and logits to the result
                for i in range(num_views):
                    res[i]["non_ambiguous_mask"] = output_masks_per_view[i]
                    res[i]["non_ambiguous_mask_logits"] = output_mask_logits_per_view[i]

        
        return res, gs_params

    def rescale_predictions(  # noqa: C901
        self,
        depth=None,
        opacity=None,
        scale=None,
        rotation=None,
        dxyzt=None,
        ms3=None,
        cov_t=None,
        omega=None,
        rgb=None,
    ):
        """
        Apply activation functions and rescale raw predictions to valid ranges.
        
        This function transforms raw network outputs into physically meaningful
        parameters for 4D Gaussian splatting. Each parameter is rescaled using
        sigmoid activation and learned biases to ensure valid ranges.
        
        Args:
            opacity: Raw opacity/alpha predictions
                Shape: [B, T, H, W, 1]
            scale: Raw 3D scale factors for Gaussians
                Shape: [B, T, H, W, 3]
            rotation: Raw rotation quaternions
                Shape: [B, T, H, W, 4]
            dxyzt: Raw spatial-temporal position residuals
                Shape: [B, T, H, W, 4] - (dx, dy, dz, dt)
            ms3: Raw marginal scale parameters for multi-scale modeling
                Shape: [B, T, H, W, 4*ms3_deg] - includes speed components
            cov_t: Raw temporal covariance
                Shape: [B, T, H, W, 1]
            omega: Raw angular velocity
                Shape: [B, T, H, W, 4*omega_deg]
            rgb: Raw RGB color values (optional)
                Shape: [B, T, H, W, 3]
                
        Returns:
            Tuple of rescaled parameters in same order as inputs,
            with physically valid ranges applied
        """ 
        # Rescale opacity to valid range [min, max] for transparency control
        if depth is not None:
            # print("depth", depth.min(), depth.max())
            # print("(depth + self.depth_bias).sigmoid().min()", (depth + self.depth_bias).sigmoid().min())
            # print("(depth + self.depth_bias).sigmoid().max()", (depth + self.depth_bias).sigmoid().max())
            depth = (depth + self.depth_bias).sigmoid() * (
                self.depth_far - self.depth_near
            ) + self.depth_near  # Map to [depth_near, depth_far] range
            
        if opacity is not None:
            # print("opacity", opacity.min(), opacity.max())
            # print("(opacity + self.opacity_bias).sigmoid().min()", (opacity + self.opacity_bias).sigmoid().min())
            # print("(opacity + self.opacity_bias).sigmoid().max()", (opacity + self.opacity_bias).sigmoid().max())
            opacity = (opacity + self.opacity_bias).sigmoid() * (
                self.opacity_max - self.opacity_min
            ) + self.opacity_min  # Map to [opacity_min, opacity_max] range
            
        # Rescale 3D scale factors and convert from radius to standard deviation
        if scale is not None:
            # print("scale", scale.min(), scale.max())
            # print("(scale + self.scale_bias).sigmoid().min()", (scale + self.scale_bias).sigmoid().min())
            # print("(scale + self.scale_bias).sigmoid().max()", (scale + self.scale_bias).sigmoid().max())
            scale = (scale + self.scale_bias).sigmoid() * (
                self.scale_max - self.scale_min
            ) + self.scale_min  # Map to [scale_min, scale_max] range
            scale = radius_to_sigma(scale)  # Convert radius to Gaussian sigma
            
        # Normalize rotation quaternions to unit length
        if rotation is not None:
            rotation = normalize(rotation)  # Ensure valid unit quaternion
            
        # Rescale spatial-temporal position residuals
        if dxyzt is not None:
            # print("dxyzt", dxyzt.min(), dxyzt.max())
            # print("(dxyzt + self.dxyzt_bias).sigmoid().min()", (dxyzt + self.dxyzt_bias).sigmoid().min())
            # print("(dxyzt + self.dxyzt_bias).sigmoid().max()", (dxyzt + self.dxyzt_bias).sigmoid().max())
            dxyzt = ((dxyzt + self.dxyzt_bias) * self.dxyzt_mult).sigmoid() * (
                self.dxyzt_max - self.dxyzt_min
            ) + self.dxyzt_min  # Map to small displacement range (e.g., 0.05m)
            
        # Process marginal scale parameters for multi-scale representation
        if ms3 is not None:
            # print("ms3", ms3.min(), ms3.max())
            # print("(ms3 + self.sigmoid_ms3_bias).sigmoid().min()", (ms3 + self.sigmoid_ms3_bias).sigmoid().min())
            # print("(ms3 + self.sigmoid_ms3_bias).sigmoid().max()", (ms3 + self.sigmoid_ms3_bias).sigmoid().max())
            # Extract degree of marginal scale (number of scale components)
            ms3_deg = ms3.shape[-1] // 4
            # Extract speed components (every 4th element starting from index 3)
            speed = ms3[..., 3::4, None]  # [B, T, H, W, ms3_deg, 1]
            # Reshape spatial components (first 3 of every 4 elements)
            ms3 = torch.cat(
                [ms3[..., None, i * 4 : i * 4 + 3] for i in range(ms3_deg)], dim=-2
            )  # [B, T, H, W, ms3_deg, 3]

            # Rescale speed with sigmoid and apply clamping threshold
            speed = (speed + self.sigmoid_ms3_bias).sigmoid() * (
                self.sigmoid_ms3_max - self.sigmoid_ms3_min
            ) + self.sigmoid_ms3_min
            speed = (speed - self.ms3_clamp).clamp(0)  # Zero out speeds below threshold

            # Apply decay factor to speed based on scale level  
            # Higher scale levels get progressively smaller speeds
            speed = torch.cat(
                [speed[..., i : i + 1, :] / self.ms3_deg_downmax_mult**i 
                 for i in range(ms3_deg)], dim=-2
            )  # [B, T, H, W, ms3_deg, 1]

            # Apply speed-modulated normalized marginal scales
            ms3 = speed * normalize(ms3[..., :3])  # Normalize and modulate by speed
            ms3 = ms3.reshape(ms3.shape[:-2] + (-1,))  # Flatten to [B, T, H, W, ms3_deg*3]
            
        # Process temporal covariance for time-varying Gaussians
        if cov_t is not None:
            # Rescale temporal covariance using sigmoid activation
            # print("cov_t", cov_t.min(), cov_t.max())
            # print("(cov_t + self.cov_t_bias).sigmoid().min()", (cov_t + self.cov_t_bias).sigmoid().min())
            # print("(cov_t + self.cov_t_bias).sigmoid().max()", (cov_t + self.cov_t_bias).sigmoid().max())
            
            cov_t = (cov_t + self.cov_t_bias).sigmoid() * (
                self.cov_t_max - self.cov_t_min
            ) + self.cov_t_min  # Map to [cov_t_min, cov_t_max] range
            cov_t = dt_to_cov_t(cov_t)  # Convert temporal difference to covariance
            
        # Process angular velocity for rotating Gaussians
        if omega is not None:
            # print("omega", omega.min(), omega.max())
            # print("(omega + self.sigmoid_omega_bias).sigmoid().min()", (omega + self.sigmoid_omega_bias).sigmoid().min())
            # print("(omega + self.sigmoid_omega_bias).sigmoid().max()", (omega + self.sigmoid_omega_bias).sigmoid().max())
            # Extract degree of angular velocity (number of rotation components)
            omega_deg = omega.shape[-1] // 4
            # Extract angular speed components (every 4th element)
            speed = omega[..., 3::4, None]  # [B, T, H, W, omega_deg, 1]
            # Reshape angular direction components (first 3 of every 4)
            omega = torch.cat(
                [omega[..., None, i * 4 : i * 4 + 3] for i in range(omega_deg)], dim=-2
            )  # [B, T, H, W, omega_deg, 3]

            # Rescale angular speed with sigmoid activation
            speed = (speed + self.sigmoid_omega_bias).sigmoid() * (
                self.sigmoid_omega_max - self.sigmoid_omega_min
            ) + self.sigmoid_omega_min
            speed = (speed - self.omega_clamp).clamp(0)  # Zero out small angular velocities

            # Apply decay factor to angular speed based on degree
            # Higher degrees get progressively smaller angular velocities
            speed = torch.cat(
                [speed[..., i : i + 1, :] / self.omega_deg_downmax_mult**i
                 for i in range(omega_deg)], dim=-2
            )  # [B, T, H, W, omega_deg, 1]

            # Apply speed-modulated normalized angular velocity
            omega = speed * normalize(omega[..., :3])  # Normalize axis and modulate by speed
            omega = omega.reshape(omega.shape[:-2] + (-1,))  # Flatten to [B, T, H, W, omega_deg*3]
            
        # Process RGB color values with learned bias and range
        if rgb is not None:
            # print("rgb", rgb.min(), rgb.max())
            # print("(rgb + self.rgb_bias).sigmoid().min()", (rgb + self.rgb_bias).sigmoid().min())
            # print("(rgb + self.rgb_bias).sigmoid().max()", (rgb + self.rgb_bias).sigmoid().max())
            # Rescale RGB values to valid color range
            rgb = (rgb + self.rgb_bias).sigmoid() * (
                self.rgb_max - self.rgb_min
            ) + self.rgb_min  # Map to [rgb_min, rgb_max] range (typically [0, 1])
        
        if depth is not None:
            return depth, opacity, scale, rotation, dxyzt, ms3, cov_t, omega, rgb
        else:
            return opacity, scale, rotation, dxyzt, ms3, cov_t, omega, rgb

    def decoded_gs_params_v2(self, dense_final_outputs_gs, views, res, batch_size_per_view, num_views, height, width, use_gt_camera, use_gt_intrinsics=True, grad_through_pred_camera=False):
        """
        使用 res 字典中的 depth 和预测出来的相机来转换成高斯参数
        
        Args:
            dense_final_outputs_gs: 高斯参数的密集输出
            res: 包含预测结果的列表，每个元素是一个字典，包含：
                - depth_along_ray: 预测的深度
                - cam_trans: 预测的相机平移
                - cam_quats: 预测的相机四元数
                - ray_directions: 射线方向
            batch_size_per_view: 每个视角的批次大小
            num_views: 视角数量
            height: 高度
            width: 宽度
            use_gt_camera: 是否使用GT相机外参
            use_gt_intrinsics: 是否使用GT射线方向(内参)，False时使用预测的ray_directions
        """
        # 处理高斯参数输出
        decoded_channels = dense_final_outputs_gs.decoded_channels
        if not self.use_separate_dpt_heads:
            dense_final_outputs_gs = decoded_channels.permute(0, 2, 3, 1).contiguous()
            dense_final_outputs_gs = dense_final_outputs_gs.view(num_views, batch_size_per_view, int(height), int(width), -1)
            dense_final_outputs_gs = dense_final_outputs_gs.permute(1, 0, 2, 3, 4)  # [B, T, H, W, C_out]
            opacity, scale, rotation, dxyzt, ms3, cov_t, omega, feat = (
                dense_final_outputs_gs.split(self.gs_params_split, dim=-1)
            )  # Split [B, T, H, W, C_out] into individual parameters based on gs_params_split
        else:
            # 在 separate 模式下，decoded_channels 是一个字典，包含每个参数的输出
            # 需要将每个参数从 [B*T, C, H, W] 转换为 [B, T, H, W, C] 格式
            gs_outputs = {}
            for param_name in self.gs_params:
                param_output = decoded_channels[param_name]  # [B*T, C, H, W]
                param_output = param_output.permute(0, 2, 3, 1).contiguous()  # [B*T, H, W, C]
                param_output = param_output.view(num_views, batch_size_per_view, int(height), int(width), -1)
                param_output = param_output.permute(1, 0, 2, 3, 4)  # [B, T, H, W, C]
                gs_outputs[param_name] = param_output

            opacity = gs_outputs["opacity"]
            scale = gs_outputs["scale"]
            rotation = gs_outputs["rotation"]
            dxyzt = gs_outputs["dxyzt"]
            ms3 = gs_outputs["ms3"]
            cov_t = gs_outputs["cov_t"]
            omega = gs_outputs["omega"]
            feat = gs_outputs["rgb"]
        
        gs_params = {}
        # gs_params["depth"] = depth  # [B, T, H, W, 1] depth along ray
        gs_params["feature"] = feat  # [B, T, H, W, C_feat] appearance features
        gs_params["opacity"] = opacity  # [B, T, H, W, 1] opacity/alpha
        gs_params["scaling"] = scale  # [B, T, H, W, 3] 3D scale factors
        gs_params["rotation"] = rotation  # [B, T, H, W, 4] rotation quaternion
        gs_params["ms3"] = ms3  # [B, T, H, W, 4*ms3_deg] marginal scale
        gs_params["cov_t"] = cov_t  # [B, T, H, W, 1] temporal covariance
        gs_params["omega"] = omega  # [B, T, H, W, 4*omega_deg] angular velocity
        gs_params["dxyzt"] = dxyzt  # [B, T, H, W, 4] position residuals

        # 从 res 中提取预测的深度
        depth_list = []
        for view_idx in range(num_views):
            depth_list.append(res[view_idx]["depth_along_ray"])
        output_depth_along_ray = torch.cat(depth_list, dim=0)  # [B*T, H, W, 1]
        output_depth_along_ray = output_depth_along_ray.reshape(num_views, batch_size_per_view, int(height), int(width), 1)
        output_depth_along_ray = output_depth_along_ray.permute(1, 0, 2, 3, 4)  # [B, T, H, W, 1]
        gs_params['depth'] = output_depth_along_ray

        # 提取射线方向：use_gt_intrinsics=True 时使用 GT，否则恢复 K 并生成 canonical pinhole rays
        ray_d_list = []
        _recovered_K_list = None
        if use_gt_intrinsics:
            for view_idx in range(num_views):
                ray_d_list.append(views[view_idx]["ray_directions_cam"])
        else:
            _recovered_K_list = []
            _pred_ray_d_list = []
            # ── Pass 1: 每帧独立恢复 K（训练默认行为，不聚合）────────────────
            _raw_K_list = []
            for view_idx in range(num_views):
                pred_rays = res[view_idx]["ray_directions"]
                _pred_ray_d_list.append(pred_rays)
                with torch.autocast("cuda", enabled=False):
                    K_raw = recover_pinhole_intrinsics_from_ray_directions(pred_rays.float(), use_geometric_calculation=False)
                    # 将 recovered K 限制在物理合理范围内，防止极端值导致梯度爆炸
                    # fx, fy: [W/4, W*4] 范围（对应 ~15°-150° FOV）
                    # cx: [W*0.2, W*0.8], cy: [H*0.2, H*0.8]
                    # NOTE: 用 stack 重建而非 K[:, i, j] = ... 的 __setitem__，
                    # 避免 inplace 索引赋值破坏 autograd 的 AsStrided view（否则
                    # backward 会报 "version N vs N+1" 错误）。
                    fx = K_raw[:, 0, 0].clamp(min=width / 4.0, max=width * 4.0)
                    fy = K_raw[:, 1, 1].clamp(min=height / 4.0, max=height * 4.0)
                    cx = K_raw[:, 0, 2].clamp(min=width * 0.2, max=width * 0.8)
                    cy = K_raw[:, 1, 2].clamp(min=height * 0.2, max=height * 0.8)
                    zero = torch.zeros_like(fx)
                    one = torch.ones_like(fx)
                    K = torch.stack([
                        torch.stack([fx,   zero, cx ], dim=-1),
                        torch.stack([zero, fy,   cy ], dim=-1),
                        torch.stack([zero, zero, one], dim=-1),
                    ], dim=-2)  # (B, 3, 3)
                _raw_K_list.append(K)

            # ── 推理专用：沿 T 维聚合/平滑 K，消除帧间 K 抖动导致的渲染呼吸/黑边 ──
            # 由 self.fix_decode_k_mode / self.fix_decode_k_sigma 控制（见 aeon4d.model.set_decode_k_smoothing），
            # 默认 "none" → 保持每帧独立 K。
            # median/mean/first: 聚合为单一 K 广播回每帧（极端档，丢失 per-frame 变焦能力）
            # gaussian: 沿 T 高斯低通，保留低频 zoom，压掉高频抖动（推荐，通过
            #           fix_decode_k_sigma 调 sigma，默认 3.0 帧）
            fix_decode_k_mode = str(getattr(self, "fix_decode_k_mode", "none")).lower()
            if fix_decode_k_mode != "none":
                K_stack = torch.stack(_raw_K_list, dim=0)  # [T, B, 3, 3]
                if fix_decode_k_mode == "median":
                    K_unified = K_stack.median(dim=0).values  # [B, 3, 3]
                    _raw_K_list = [K_unified for _ in range(num_views)]
                elif fix_decode_k_mode == "first":
                    K_unified = K_stack[0]
                    _raw_K_list = [K_unified for _ in range(num_views)]
                elif fix_decode_k_mode == "mean":
                    K_unified = K_stack.mean(dim=0)
                    _raw_K_list = [K_unified for _ in range(num_views)]
                elif fix_decode_k_mode == "gaussian":
                    from torch.nn.functional import conv1d, pad as _pad_fn
                    sigma = float(getattr(self, "fix_decode_k_sigma", 3.0))
                    T_k = K_stack.shape[0]
                    if T_k > 1 and sigma > 0:
                        radius = max(1, int(3 * sigma))
                        dev = K_stack.device
                        # 用 fp32 卷积防 bf16 数值损失；kernel 在 [-3σ, 3σ] 覆盖 99.7%
                        x = torch.arange(-radius, radius + 1, device=dev, dtype=torch.float32)
                        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
                        kernel = (kernel / kernel.sum()).view(1, 1, -1)  # [1,1,2R+1]
                        K32 = K_stack.float()  # [T, B, 3, 3]
                        B_k = K32.shape[1]
                        # [T, B, 3, 3] → [B*9, 1, T]
                        K_flat = K32.permute(1, 2, 3, 0).reshape(B_k * 9, 1, T_k)
                        K_pad = _pad_fn(K_flat, (radius, radius), mode="replicate")
                        K_smooth = conv1d(K_pad, kernel)  # [B*9, 1, T]
                        K_smooth = K_smooth.reshape(B_k, 3, 3, T_k).permute(3, 0, 1, 2).to(K_stack.dtype)
                        _raw_K_list = [K_smooth[t] for t in range(num_views)]
                    # T<=1 或 sigma<=0 时直接保留原 list，不动
                else:
                    raise ValueError(
                        f"Unknown fix_decode_k_mode={fix_decode_k_mode!r}; "
                        "expected one of: none, median, first, mean, gaussian"
                    )

            # ── Pass 2: 用（可能聚合后的）K 为每帧生成 canonical ray ────────
            for view_idx in range(num_views):
                K = _raw_K_list[view_idx]
                pred_rays = _pred_ray_d_list[view_idx]
                with torch.autocast("cuda", enabled=False):
                    _, canonical = get_rays_in_camera_frame(K, int(height), int(width), normalize_to_unit_sphere=True)
                ray_d_list.append(canonical.to(pred_rays.dtype))
                _recovered_K_list.append(K)

        ray_d = torch.stack(ray_d_list, dim=1)  # [B, T, H, W, 3]
        ray_d = ray_d.view(batch_size_per_view, num_views, int(height), int(width), 3)
        gs_params["ray_d"] = ray_d

        if not use_gt_intrinsics and _recovered_K_list is not None:
            pred_ray_d = torch.stack(_pred_ray_d_list, dim=1)
            pred_ray_d = pred_ray_d.view(batch_size_per_view, num_views, int(height), int(width), 3)
            _RAY_Z_EPS = 0.1  # unit-normalized rays: z=0.1 ≈ 84° off-axis, safe for any real camera
            # clamp pred z-component directly to prevent small-z amplification
            z_depth = (output_depth_along_ray * pred_ray_d[..., 2:3].clamp(min=_RAY_Z_EPS)).clamp(min=1e-4)
            # detach canonical z in denominator: gradients to pred_rays still flow through
            # canonical rays in gs_xyz = canonical * depth, avoiding double gradient path
            depth_along_canonical = z_depth / ray_d[..., 2:3].detach().clamp(min=_RAY_Z_EPS)
            gs_params['depth'] = depth_along_canonical

        # 生成时间戳 ts
        device = ray_d.device
        dtype = ray_d.dtype
        ts_values = torch.arange(num_views, device=device, dtype=dtype) / 24.0
        ts = ts_values.view(1, num_views, 1, 1, 1)
        ts = ts.expand(batch_size_per_view, num_views, int(height), int(width), 1)
        gs_params["t"] = ts
        
        
        if not self.high_resolution:
            gs_params = self.magic_filter(gs_params)

        depth, opacity, scale, rotation, dxyzt, ms3, cov_t, omega, feat = (
            gs_params["depth"],  # [B, T, H, W, 1] depth along ray
            gs_params["opacity"],  # [B, T, H, W, 1] opacity
            gs_params["scaling"],  # [B, T, H, W, 3] 3D scales
            gs_params["rotation"],  # [B, T, H, W, 4] quaternions
            gs_params["dxyzt"],  # [B, T, H, W, 4] position residuals
            gs_params["ms3"],  # [B, T, H, W, 1] marginal scale
            gs_params["cov_t"],  # [B, T, H, W, 1] temporal covariance
            gs_params["omega"],  # [B, T, H, W, 3] angular velocity
            gs_params["feature"],  # [B, T, H, W, C_feat] appearance
        )
        ray_d = gs_params["ray_d"]

        # 根据 use_gt_camera 选择使用 GT 相机还是预测相机
        device = ray_d.device
        dtype = ray_d.dtype
        
        if use_gt_camera:
            # 训练时使用 GT 相机参数
            cam_trans_list = []
            cam_quats_list = []
            for view_idx in range(num_views):
                if "camera_pose_in_view0_trans_norm" in views[view_idx]:
                    cam_trans_list.append(views[view_idx]["camera_pose_in_view0_trans_norm"])
                else:
                    # 如果没有提供 camera pose，使用零向量
                    print("warning: no camera pose in view0, use zero vector")
                    cam_trans_list.append(torch.zeros(batch_size_per_view, 3, device=device, dtype=dtype))
                if "camera_pose_in_view0_quats" in views[view_idx]:
                    cam_quats_list.append(views[view_idx]["camera_pose_in_view0_quats"])
                else:
                    # 如果没有提供 camera pose，使用零向量
                    print("warning: no camera pose in view0, use zero vector")
                    cam_quats_list.append(torch.zeros(batch_size_per_view, 4, device=device, dtype=dtype))
            
            # 拼接所有相机参数: [B, T, 3] 和 [B, T, 4]
            cam_trans = torch.stack(cam_trans_list, dim=1)  # [B, T, 3]
            cam_quats = torch.stack(cam_quats_list, dim=1)  # [B, T, 4]
            cam_trans = cam_trans.reshape(-1, 3)  # [B*T, 3]
            cam_quats = cam_quats.reshape(-1, 4)  # [B*T, 4]
        else:
            # 使用预测的相机参数。默认 detach 切断对 pose_head 的梯度；
            # grad_through_pred_camera=True 时保留梯度，让 gs_xyz 的 RGB/geometry
            # loss 通过 R(quat)·canonical·depth + trans 流回 pose_head。
            cam_trans_list = []
            cam_quats_list = []
            for view_idx in range(num_views):
                ct = res[view_idx]["cam_trans"]
                cq = res[view_idx]["cam_quats"]
                if not grad_through_pred_camera:
                    ct = ct.detach()
                    cq = cq.detach()
                cam_trans_list.append(ct)  # [B, 3]
                cam_quats_list.append(cq)  # [B, 4]
            
            # 先 stack 成 [B, T, 3] 和 [B, T, 4]，然后 reshape 成 [B*T, 3] 和 [B*T, 4]
            cam_trans = torch.stack(cam_trans_list, dim=1)  # [B, T, 3]
            cam_quats = torch.stack(cam_quats_list, dim=1)  # [B, T, 4]
            cam_trans = cam_trans.reshape(-1, 3)  # [B*T, 3]
            cam_quats = cam_quats.reshape(-1, 4)  # [B*T, 4]

        opacity, scale, rotation, dxyzt, ms3, cov_t, omega, feat = self.rescale_predictions(
            # depth=depth,
            opacity=opacity,
            scale=scale,
            rotation=rotation,
            dxyzt=dxyzt,
            ms3=ms3,
            cov_t=cov_t,
            omega=omega,
            rgb=feat,
        )

        B, T, H, W = depth.shape[:4]
        ray_d_flat = ray_d.reshape(B * T, H, W, 3)
        depth_flat = depth.squeeze(-1).reshape(B * T, H, W, 1)
        
        
        
        # 使用预测的相机参数计算世界坐标系中的高斯位置
        gs_xyz = convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
            ray_d_flat,
            depth_flat,
            cam_trans,  # 直接使用预测的相机平移，已经是 [B*T, 3]
            cam_quats,  # 直接使用预测的相机四元数，已经是 [B*T, 4]
        )

        B, H, W, _ = gs_xyz.shape
        ts = gs_params["t"]

        gs_params = {}
        gs_params["feature"] = feat  # [B, T, H, W, C_feat]
        gs_params["opacity"] = opacity  # [B, T, H, W, 1]
        gs_params["scaling"] = scale  # [B, T, H, W, 3]
        gs_params["rotation"] = rotation  # [B, T, H, W, 4]

        # 从深度和射线计算3D世界位置
        gs_params["xyz"] = gs_xyz.reshape(batch_size_per_view, num_views, H, W, 3)  # [B, T, H, W, 3] world positions
        gs_params["t"] = ts

        # 添加4D高斯时间参数
        gs_params["ms3"] = ms3  # [B, T, H, W, 1] marginal scale
        gs_params["cov_t"] = cov_t  # [B, T, H, W, 1] temporal covariance
        gs_params["omega"] = omega  # [B, T, H, W, 3] angular velocity

        # 应用空间-时间扰动
        gs_params["dxyzt"] = dxyzt  # [B, T, H, W, 4] position residuals

        if self.perturb_xyz:
            gs_params["xyz"] = gs_params["xyz"] + gs_params["dxyzt"][..., :3]  # [B, T, H, W, 3] perturbed positions
        if self.perturb_t:
            gs_params["t"] = gs_params["t"] + gs_params["dxyzt"][..., 3:]  # [B, T, H, W, 1] perturbed time

        for k in gs_params:
            gs_params[k] = gs_params[k].float()  # Convert to float32

        if _recovered_K_list is not None:
            gs_params["recovered_K"] = torch.stack(_recovered_K_list, dim=1).float()  # [B, T, 3, 3]

        return gs_params


    def magic_filter(self, gs_params: dict, magic_pattern=None):  # noqa: C901
        magic_pattern = (
            magic_pattern if magic_pattern is not None else self.magic_pattern
        )
        Mn = self.magic_num
        magic_pattern = [magic_pattern[0][:Mn], magic_pattern[1][:Mn]]

        # print("self.patch_size", self.patch_size)
        Hp, Wp = self.patch_size
        if not self.recalc_magic_pattern:
            # Downsample gs_params to get the number of points smaller for rendering
            if np.prod([len(m) for m in magic_pattern]) > 1:
                for key in list(gs_params.keys()):
                    # Perform magic pattern sampling on the values
                    val = gs_params[key]
                    val = rearrange(
                        val, "B T (Nh Hp) (Nw Wp) C -> B T Nh Nw Hp Wp C", Hp=Hp, Wp=Wp
                    )
                    val = val[
                        :, :, :, :, magic_pattern[0], magic_pattern[1], :
                    ]  # sample the 10 important points
                    val = rearrange(val, "B T Nh Nw X C -> B T (Nh X) Nw C")
                    gs_params[key] = val
        else:
            if self.recalc_method == "patch_sorting":
                # Recalculate magic pattern for each patch
                Hp, Wp = self.patch_size
                occ = gs_params["opacity"].float()
                occ = rearrange(
                    occ, "B T (Nh Hp) (Nw Wp) C -> B T Nh Nw (Hp Wp) C", Hp=Hp, Wp=Wp
                )

                val, ind = occ.topk(
                    len(magic_pattern[0]), dim=-2
                )  # B, T, Nh, Nw, 10, 1
                for key in list(gs_params.keys()):
                    # Perform magic pattern sampling on the values
                    val = gs_params[key]
                    val = rearrange(
                        val,
                        "B T (Nh Hp) (Nw Wp) C -> B T Nh Nw (Hp Wp) C",
                        Hp=Hp,
                        Wp=Wp,
                    )
                    val = multi_gather(val, ind, dim=-2)  # B, T, Nh, Nw, 10, C
                    val = rearrange(val, "B T Nh Nw X C -> B T (Nh X) Nw C")
                    gs_params[key] = val
            elif self.recalc_method == "patch_random":
                # Fixed patch random pattern
                # Recalculate magic pattern for each patch
                Hp, Wp = self.patch_size
                ind = self.random_pattern[None, None, None, None, :, None]  # 10, 1

                for key in list(gs_params.keys()):
                    # Perform magic pattern sampling on the values
                    val = gs_params[key]
                    val = rearrange(
                        val,
                        "B T (Nh Hp) (Nw Wp) C -> B T Nh Nw (Hp Wp) C",
                        Hp=Hp,
                        Wp=Wp,
                    )
                    val = multi_gather(val, ind, dim=-2)  # B, T, Nh, Nw, 10, C
                    val = rearrange(val, "B T Nh Nw X C -> B T (Nh X) Nw C")
                    gs_params[key] = val
            elif self.recalc_method == "global_random":
                # Fixed global random pattern
                # Recalculate magic pattern for each patch
                ind = self.random_pattern[None, None]  # N, 1

                for key in list(gs_params.keys()):
                    # Perform magic pattern sampling on the values
                    val = gs_params[key]
                    val = rearrange(
                        val,
                        "B T H W C -> B T (H W) C",
                    )
                    max_rand_val = val.shape[-2]
                    ind = ind % max_rand_val  # B, T, N, 1
                    val = multi_gather(val, ind, dim=-2)  # B, T, X, C
                    val = rearrange(val, "B T X C -> B T X 1 C")
                    gs_params[key] = val
            elif self.recalc_method == "global_sorting":
                # Recalculate magic pattern for each patch
                Hp, Wp = self.patch_size
                occ = gs_params["opacity"].float()
                occ = rearrange(occ, "B T H W C -> B T (H W) C")

                val, ind = occ.topk(len(self.random_pattern), dim=-2)  # B, T, X, 1
                for key in list(gs_params.keys()):
                    # Perform magic pattern sampling on the values
                    val = gs_params[key]
                    val = rearrange(
                        val,
                        "B T H W C -> B T (H W) C",
                    )
                    val = multi_gather(val, ind, dim=-2)  # B, T, X, C
                    val = rearrange(val, "B T X C -> B T X 1 C")
                    gs_params[key] = val
            elif self.recalc_method == "uniform":
                # Fixed global random pattern
                # Recalculate magic pattern for each patch
                for key in list(gs_params.keys()):
                    # Perform magic pattern sampling on the values
                    gs_params[key] = gs_params[key][..., ::4, ::5, :]  # 1 / 20
            else:
                raise NotImplementedError(
                    f"Recalc method {self.recalc_method} not implemented"
                )

        return gs_params
        