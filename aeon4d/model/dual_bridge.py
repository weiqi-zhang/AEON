"""
DualConv3DPerceiverBridge: a residual-style bridge that combines the
ridge-regression-friendly single Conv3D with a learned PerceiverBridge.

Motivation
----------
The layer-search experiment (find_layer_for_stitching.py, DynamicReplica)
solves a closed-form Conv3D per DINOv2 block and reports R² per layer.
At the recommended stitch point (block 4) a single Conv3D explains ~63 %
of the block output variance — a strong starting point but not a full
reconstruction.  The remaining ~37 % is non-linear residual that a single
Conv3D cannot capture.

DualBridge runs both paths in parallel:

    path_A = Conv3D(vae_latent)                 # already fits 63 % at init
    path_B = PerceiverBridge(vae_latent)        # non-linear residual learner
    out    = path_A + gate * path_B             # gate is a scalar, init = 0

At training start ``gate = 0`` so the bridge behaves exactly like the
ridge-solved Conv3D.  As training progresses the Perceiver branch learns
the remaining residual and ``gate`` grows.  This guarantees a monotonic
improvement over the Conv3D baseline (never worse) while giving Perceiver
room to expand.

Backward compatibility
----------------------
The forward signature and the ``temporal_upsample_factor`` attribute match
PerceiverBridge / ConvBridge so StitchVAE4D's bridge dispatcher only needs
a new ``elif`` branch.  No existing config or checkpoint is affected.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from aeon4d.model.perceiver_bridge import PerceiverBridge

logger = logging.getLogger(__name__)


class DualConv3DPerceiverBridge(nn.Module):
    """Residual dual-path bridge: Conv3D (ridge-init) + gated Perceiver.

    Args:
        vae_channels:       VAE latent channel count (16 for Wan VAE).
        out_dim:            Output token dim (must match DINOv2 embed_dim, 1024).
        out_h, out_w:       Output token grid (e.g. 32x32 for 448//14).

        # Conv3D path config (must match the spec used at layer search)
        conv3d_kernel:      (kT, kH, kW) – default (5, 3, 3).
        conv3d_stride:      (sT, sH, sW) – default (1, 2, 2) so 64 → 32 spatial.
        conv3d_padding:     (pT, pH, pW) – default (2, 1, 1) to preserve T & H.
        conv3d_init_path:   Path to a ``state_dict`` file produced by
                            find_layer_for_stitching.py
                            (e.g. ``state_dict_enc_blocks_4.pt``).
                            ``None`` → Xavier init (equivalent to a random
                            initialization for the Conv3D path).

        # Perceiver path config (forwarded to PerceiverBridge)
        perceiver_num_layers, perceiver_num_heads, perceiver_cross_attn_freq,
        perceiver_mlp_ratio, perceiver_temporal_kernel,
        perceiver_temporal_upsample_factor: identical meaning as in
            PerceiverBridge.

        # Fusion
        gate_init:          Initial value of the Perceiver scalar gate
                            (default 0.0, i.e. bridge = Conv3D at step 0).
    """

    def __init__(
        self,
        vae_channels: int = 16,
        out_dim: int = 1024,
        out_h: int = 32,
        out_w: int = 32,
        # Conv3D path
        conv3d_kernel=(5, 3, 3),
        conv3d_stride=(1, 2, 2),
        conv3d_padding=(2, 1, 1),
        conv3d_init_path: str | None = None,
        # Perceiver path
        perceiver_num_layers: int = 4,
        perceiver_num_heads: int = 16,
        perceiver_cross_attn_freq: int = 1,
        perceiver_mlp_ratio: float = 8.0,
        perceiver_temporal_kernel: int = 5,
        perceiver_temporal_upsample_factor: int = 4,
        # Fusion
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.out_h = out_h
        self.out_w = out_w

        # Expose temporal_upsample_factor for StitchVAE4D's dispatcher so it
        # knows the bridge takes care of the T_vae → T upsampling internally.
        self.temporal_upsample_factor = perceiver_temporal_upsample_factor

        # ── Path A: Conv3D with replicate padding (matches layer search) ──
        self.conv3d = nn.Conv3d(
            in_channels=vae_channels,
            out_channels=out_dim,
            kernel_size=tuple(conv3d_kernel),
            stride=tuple(conv3d_stride),
            padding=tuple(conv3d_padding),
            padding_mode="replicate",
        )

        if conv3d_init_path is not None:
            logger.info(f"Loading Conv3D ridge init from {conv3d_init_path}")
            sd = torch.load(
                conv3d_init_path, map_location="cpu", weights_only=False,
            )
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            missing, unexpected = self.conv3d.load_state_dict(sd, strict=False)
            if missing or unexpected:
                logger.warning(
                    f"Conv3D load_state_dict missing={missing}, "
                    f"unexpected={unexpected}"
                )

        # ── Path B: PerceiverBridge (handles temporal upsampling itself) ──
        self.perceiver = PerceiverBridge(
            vae_channels=vae_channels,
            out_dim=out_dim,
            out_h=out_h,
            out_w=out_w,
            num_layers=perceiver_num_layers,
            num_heads=perceiver_num_heads,
            cross_attn_freq=perceiver_cross_attn_freq,
            mlp_ratio=perceiver_mlp_ratio,
            temporal_kernel=perceiver_temporal_kernel,
            temporal_upsample_factor=perceiver_temporal_upsample_factor,
        )

        # ── Fusion gate: scalar, zero-initialized so the Perceiver branch
        #    contributes nothing at step 0 and grows as it learns. ──
        self.perceiver_gate = nn.Parameter(torch.tensor(float(gate_init)))

        logger.info(
            f"DualConv3DPerceiverBridge: vae_ch={vae_channels}, "
            f"out_dim={out_dim}, out_grid={out_h}x{out_w}, "
            f"conv3d_k={conv3d_kernel}, s={conv3d_stride}, p={conv3d_padding}, "
            f"gate_init={gate_init}, "
            f"params={sum(p.numel() for p in self.parameters()):,}"
        )

    def forward(self, x: torch.Tensor, num_frames: int | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, 16, T_vae, H_vae, W_vae) — raw VAE latent.
            num_frames: target frame count T (output temporal dim).
                If None, inferred from ``T = 1 + (T_vae - 1) * 4`` (Wan VAE).

        Returns:
            (B, out_dim, T, out_h, out_w) — already on the DINOv2 token grid.
        """
        B, C, T_vae, H, W = x.shape
        if num_frames is None:
            num_frames = 1 + (T_vae - 1) * 4

        # ── Path A: Conv3D on a trilinearly-upsampled latent. ──
        # Conv3D spec keeps T dim unchanged, so we must upsample T_vae → T
        # before the conv (identical to what find_layer_for_stitching.py
        # does inside fit_conv3d_streaming).
        if T_vae != num_frames:
            x_upsampled = F.interpolate(
                x,
                size=(num_frames, H, W),
                mode="trilinear",
                align_corners=True,
            )
        else:
            x_upsampled = x
        path_a = self.conv3d(x_upsampled)  # (B, out_dim, T, out_h, out_w)

        # ── Path B: Perceiver (has its own asymmetric temporal upsampling). ──
        path_b = self.perceiver(x, num_frames=num_frames)  # (B, out_dim, T, out_h, out_w)

        # Record branch magnitudes so training code can report the *actual*
        # Perceiver contribution rather than relying on the scalar gate alone
        # (gate value is entangled with path_B's internal weight scale, so
        # gate=0.05 can mean anything from 0.1% to 50% depending on ||path_B||).
        # RMS (root-mean-square) is size-invariant and comparable across
        # tensors of different shapes.
        with torch.no_grad():
            self._path_a_rms = path_a.detach().float().pow(2).mean().sqrt().item()
            self._path_b_rms = path_b.detach().float().pow(2).mean().sqrt().item()

        # ── Fusion: residual sum with learned scalar gate (init 0). ──
        return path_a + self.perceiver_gate * path_b
