"""
StitchVAE4D: connects Wan video VAE latents to a 4D Gaussian feed-forward model
(MapAnythingGS) via a learnable stitching bridge and a modified DINOv2 encoder.

Supported bridge types:
  - "conv3d" (legacy): single Conv3D layer parsed from a spec string.
  - "perceiver": PerceiverBridge with cross-attention.
  - "dual_conv3d_perceiver": Conv3D + gated PerceiverBridge (used by the released checkpoint).
"""

import logging

import torch.nn as nn
import torch.nn.functional as F

from aeon4d.model.mapanything_stitched import MapAnythingGSStitched

logger = logging.getLogger(__name__)


class StitchVAE4D(nn.Module):
    """
    Flow: Wan latent (B, 16, T_vae, H/8, W/8) -> stitching bridge (with learned temporal upsampling)
    -> per-frame DINOv2 token grids -> MapAnythingGS downstream -> 4D Gaussians.
    """

    def __init__(
        self,
        feedforward_model: MapAnythingGSStitched,
        stitching_layer_config,
        vae_resolution: int = 512,
        ff_resolution: int = 518,
        bridge_type: str = "conv3d",
        bridge_config: dict = None,
    ):
        super().__init__()

        # Feed-forward 4D model (DINOv2 encoder with token injection)
        self.stitched_4d_model = feedforward_model

        # Dimensions derived from the feed-forward model
        self.enc_embed_dim = feedforward_model.enc_embed_dim
        self.patch_size = feedforward_model.patch_size
        self.ff_resolution = ff_resolution

        # Token injection mode flag (propagated from feedforward_model)
        self.token_injection = feedforward_model.stitch_skip_blocks > 0

        # VAE latent spatial size: vae_resolution // 8
        self.vae_spatial = vae_resolution // 8  # 64 for 512px

        # VAE channel count (Wan VAE uses 16 latent channels)
        self.vae_channels = 16

        self.bridge_type = bridge_type

        logger.info(
            f"StitchVAE4D: VAE spatial={self.vae_spatial}, "
            f"ff_resolution={self.ff_resolution}, "
            f"enc_embed_dim={self.enc_embed_dim}, "
            f"token_injection={self.token_injection}, "
            f"bridge_type={bridge_type}"
        )

        # Build stitching layer / bridge
        cfg = bridge_config or {}
        default_out_h = ff_resolution // feedforward_model.patch_size
        default_out_w = default_out_h
        if bridge_type == "perceiver":
            from aeon4d.model.perceiver_bridge import PerceiverBridge

            self.stitching_layer = PerceiverBridge(
                vae_channels=self.vae_channels,
                out_dim=cfg.get("out_dim", self.enc_embed_dim),
                out_h=cfg.get("out_h", default_out_h),
                out_w=cfg.get("out_w", default_out_w),
                num_layers=cfg.get("num_layers", 2),
                num_heads=cfg.get("num_heads", 16),
                cross_attn_freq=cfg.get("cross_attn_freq", 1),
                mlp_ratio=cfg.get("mlp_ratio", 4.0),
                temporal_kernel=cfg.get("temporal_kernel", 5),
                temporal_upsample_factor=cfg.get("temporal_upsample_factor", 1),
            )
        elif bridge_type == "dual_conv3d_perceiver":
            from aeon4d.model.dual_bridge import DualConv3DPerceiverBridge

            self.stitching_layer = DualConv3DPerceiverBridge(
                vae_channels=self.vae_channels,
                out_dim=cfg.get("out_dim", self.enc_embed_dim),
                out_h=cfg.get("out_h", default_out_h),
                out_w=cfg.get("out_w", default_out_w),
                # Conv3D path (weights come from the Aeon checkpoint; no ridge-init file at inference)
                conv3d_kernel=cfg.get("conv3d_kernel", (5, 3, 3)),
                conv3d_stride=cfg.get("conv3d_stride", (1, 2, 2)),
                conv3d_padding=cfg.get("conv3d_padding", (2, 1, 1)),
                conv3d_init_path=None,
                # Perceiver path
                perceiver_num_layers=cfg.get("perceiver_num_layers", 4),
                perceiver_num_heads=cfg.get("perceiver_num_heads", 16),
                perceiver_cross_attn_freq=cfg.get("perceiver_cross_attn_freq", 1),
                perceiver_mlp_ratio=cfg.get("perceiver_mlp_ratio", 8.0),
                perceiver_temporal_kernel=cfg.get("perceiver_temporal_kernel", 5),
                perceiver_temporal_upsample_factor=cfg.get("perceiver_temporal_upsample_factor", 4),
                # Fusion
                gate_init=cfg.get("gate_init", 0.0),
            )
        elif bridge_type == "conv3d":
            # Legacy: single Conv3D from spec string
            self.stitching_layer = stitching_layer_config.build(in_channels=self.vae_channels)
        else:
            raise ValueError(f"Unsupported bridge_type: {bridge_type}")

    def forward_with_latent(self, latent, views, num_frames, feedforward_image,
                            use_gt_camera=False, use_gt_intrinsics=False):
        """
        Forward from a Wan VAE latent.

        Args:
            latent: (B, 16, T_vae, H_vae/8, W_vae/8)
            views: list of view dicts (len = num_frames)
            num_frames: T
            feedforward_image: (B, 3, T, ff_res, ff_res) decoded pixel video in [-1, 1],
                used by the image adapters of the Gaussian heads.
            use_gt_camera / use_gt_intrinsics: inference uses predicted cameras (False).
        """
        T_vae = latent.shape[2]

        # 1) Temporal handling + stitching bridge
        bridge_handles_temporal = (
            self.bridge_type in ("perceiver", "dual_conv3d_perceiver")
            and getattr(self.stitching_layer, "temporal_upsample_factor", 1) > 1
        )

        if bridge_handles_temporal:
            # Bridge performs learned temporal upsampling internally
            stitched = self.stitching_layer(latent, num_frames=num_frames)
        else:
            # Legacy path: trilinear temporal interpolation before bridge
            if T_vae != num_frames:
                latent = F.interpolate(
                    latent,
                    size=(num_frames, latent.shape[3], latent.shape[4]),
                    mode="trilinear",
                    align_corners=False,
                )
            stitched = self.stitching_layer(latent)
        # stitched: (B, C_stitch, T, H_stitch, W_stitch)

        T = stitched.shape[2]

        if self.bridge_type in ("perceiver", "dual_conv3d_perceiver") or self.token_injection:
            # Bridge / token-injection output is already a per-frame DINOv2 token grid
            vae_latents_per_view = [stitched[:, :, t, :, :] for t in range(T)]
        else:
            # Legacy patch_embed mode: bilinear upsample each frame to the view resolution
            _, _, target_h, target_w = views[0]["img"].shape
            vae_latents_per_view = [
                F.interpolate(stitched[:, :, t, :, :], size=(target_h, target_w),
                              mode="bilinear", align_corners=False)
                for t in range(T)
            ]

        # 2) Stitched 4D model (modified DINOv2 + MapAnythingGS downstream heads)
        return self.stitched_4d_model(
            vae_latents_per_view=vae_latents_per_view,
            views=views,
            feedforward_image=feedforward_image,
            use_gt_camera=use_gt_camera,
            use_gt_intrinsics=use_gt_intrinsics,
            return_encoder_features=False,
            grad_through_pred_camera=False,
        )
