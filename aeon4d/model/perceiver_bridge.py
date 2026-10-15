"""
PerceiverBridge: a cross-attention-based stitching bridge that maps
VAE latent tokens to DINOv2-compatible tokens.

Design rationale (from v5 training analysis):
  1. Single Conv3D is too weak — linear, no nonlinearity, 3×3 receptive field.
  2. Spatial downsampling (stride 2) + bilinear interpolation loses depth info.
  3. Cross-attention enables global receptive field + lossless resolution mapping.

Architecture:
  VAE latent (B, 16, T, H_vae, W_vae)
      → InputProj: Conv3D(16→out_dim, k=5×3×3, s=1×1×1) + LN + GELU
      → Per-frame cross-attention: learnable queries (H_out×W_out) attend to
        VAE tokens (H_in×W_in), interleaved with self-attention
      → Output LayerNorm
      → (B, out_dim, T, H_out, W_out)  ready for DINOv2 token injection

Assumes single square resolution training (e.g. 518×518 views → vae_res 520
→ 65×65 VAE latent → 37×37 output queries).
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class _CrossAttentionLayer(nn.Module):
    """Single layer: self-attention among queries + cross-attention to KV."""

    def __init__(self, dim: int, num_heads: int = 16, mlp_ratio: float = 4.0,
                 use_cross_attn: bool = True):
        super().__init__()
        self.use_cross_attn = use_cross_attn

        # Self-attention
        self.norm_sa = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True,
        )

        # Cross-attention (optional — some layers are self-attn only)
        if use_cross_attn:
            self.norm_ca_q = nn.LayerNorm(dim)
            self.norm_ca_kv = nn.LayerNorm(dim)
            self.cross_attn = nn.MultiheadAttention(
                dim, num_heads, batch_first=True,
            )

        # FFN
        self.norm_ffn = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, queries: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: (B, N_q, D)
            kv:      (B, N_kv, D)
        Returns:
            queries: (B, N_q, D)
        """
        # Self-attention
        q_norm = self.norm_sa(queries)
        queries = queries + self.self_attn(q_norm, q_norm, q_norm, need_weights=False)[0]

        # Cross-attention
        if self.use_cross_attn:
            q_norm = self.norm_ca_q(queries)
            kv_norm = self.norm_ca_kv(kv)
            queries = queries + self.cross_attn(
                q_norm, kv_norm, kv_norm, need_weights=False,
            )[0]

        # FFN
        queries = queries + self.ffn(self.norm_ffn(queries))
        return queries


class PerceiverBridge(nn.Module):
    """
    Cross-attention bridge from VAE latent space to DINOv2 token space.

    Replaces the single Conv3D stitching layer.  Backward-compatible: the
    ``forward()`` signature matches what ``StitchVAE4D`` expects from
    ``stitching_layer`` (input 5-D tensor, output 5-D tensor).

    Args:
        vae_channels:  VAE latent channel count (default 16).
        out_dim:       Output token dimension (must match DINOv2 embed_dim, 1024).
        out_h:         Output spatial height (e.g. 37 for 518//14).
        out_w:         Output spatial width.
        num_layers:    Number of cross-attention layers.
        num_heads:     Attention heads.
        cross_attn_freq: Insert cross-attention every N layers (Q-Former style).
        mlp_ratio:     FFN hidden dim = dim * mlp_ratio.
        temporal_kernel: Temporal kernel size for the input Conv3D projection.
    """

    def __init__(
        self,
        vae_channels: int = 16,
        out_dim: int = 1024,
        out_h: int = 37,
        out_w: int = 37,
        num_layers: int = 2,
        num_heads: int = 16,
        cross_attn_freq: int = 1,
        mlp_ratio: float = 4.0,
        temporal_kernel: int = 5,
        temporal_upsample_factor: int = 1,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.out_h = out_h
        self.out_w = out_w

        # --- Input projection: Conv3D on VAE latent (no spatial downsampling) ---
        t_pad = temporal_kernel // 2
        self.input_proj = nn.Sequential(
            nn.Conv3d(
                vae_channels, out_dim,
                kernel_size=(temporal_kernel, 3, 3),
                stride=(1, 1, 1),
                padding=(t_pad, 1, 1),
                padding_mode="replicate",
            ),
            # LayerNorm on channel dim — use GroupNorm(1, C) as a channel-last LN
            # for 5-D tensors
            nn.GroupNorm(1, out_dim),
            nn.GELU(),
        )

        # --- Learnable query grid ---
        num_queries = out_h * out_w
        self.queries = nn.Parameter(
            torch.randn(1, num_queries, out_dim) * 0.02
        )

        # --- 2D positional encoding for queries (fixed sinusoidal) ---
        self.register_buffer(
            "query_pos_embed",
            self._build_2d_sincos_pos_embed(out_dim, out_h, out_w),
        )

        # --- Cross-attention layers ---
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            use_cross = (i % cross_attn_freq == 0) if cross_attn_freq > 0 else False
            if cross_attn_freq == 1:
                use_cross = True
            self.layers.append(
                _CrossAttentionLayer(
                    dim=out_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_cross_attn=use_cross,
                )
            )

        # --- Output normalization (align distribution with DINOv2 tokens) ---
        self.output_norm = nn.LayerNorm(out_dim)

        # --- Learned temporal upsampling (T_vae → T) ---
        # Replaces trilinear interpolation with learned interleave upsampling
        # (same mechanism as WanDecoder3d's upsample3d).
        #
        # Asymmetric design matching Wan VAE encoding structure:
        #   Wan VAE encodes T frames as: L0 (1 frame, 1:1) + L1..Lk (4 frames each)
        #   So T_vae = 1 + k, and T = 1 + 4k.
        #   We only upsample L1..Lk by 4×, keeping L0 as-is:
        #     first_frame (1) + rest_frames * 4 = 1 + 4k = T  (exact match)
        #   This eliminates the lossy trilinear trim that was previously needed.
        #
        # When factor=1 (default), no temporal upsampling is performed.
        self.temporal_upsample_factor = temporal_upsample_factor
        if temporal_upsample_factor > 1:
            num_stages = 0
            f = temporal_upsample_factor
            while f > 1:
                assert f % 2 == 0, (
                    f"temporal_upsample_factor must be a power of 2, "
                    f"got {temporal_upsample_factor}"
                )
                f //= 2
                num_stages += 1
            self.temporal_ups = nn.ModuleList()
            for _ in range(num_stages):
                self.temporal_ups.append(nn.Sequential(
                    nn.Conv3d(
                        out_dim, out_dim * 2,
                        kernel_size=(3, 1, 1), padding=(1, 0, 0),
                    ),
                    nn.GroupNorm(1, out_dim * 2),
                    nn.GELU(),
                ))

        # KV positional encoding is built once on first forward (depends on
        # input spatial size which is fixed for single-resolution training).
        self._kv_pe: torch.Tensor | None = None

        self._init_weights()
        logger.info(
            f"PerceiverBridge: vae_ch={vae_channels}, out_dim={out_dim}, "
            f"out_grid={out_h}x{out_w}, layers={num_layers}, "
            f"heads={num_heads}, cross_freq={cross_attn_freq}, "
            f"temporal_upsample={temporal_upsample_factor}x, "
            f"params={sum(p.numel() for p in self.parameters()):,}"
        )

    def _init_weights(self):
        """Xavier init for linear layers, small init for output projections."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _build_2d_sincos_pos_embed(dim: int, h: int, w: int) -> torch.Tensor:
        """Build 2-D sinusoidal positional embedding (1, H*W, D)."""
        assert dim % 4 == 0, f"dim must be divisible by 4, got {dim}"
        half = dim // 2
        grid_y = torch.arange(h, dtype=torch.float32).unsqueeze(1).expand(h, w)
        grid_x = torch.arange(w, dtype=torch.float32).unsqueeze(0).expand(h, w)
        omega = 1.0 / (10000.0 ** (torch.arange(0, half, 2, dtype=torch.float32) / half))

        # y-dim
        pos_y = grid_y.flatten().unsqueeze(1) * omega.unsqueeze(0)
        # x-dim
        pos_x = grid_x.flatten().unsqueeze(1) * omega.unsqueeze(0)

        pe = torch.cat([pos_y.sin(), pos_y.cos(), pos_x.sin(), pos_x.cos()], dim=1)
        return pe.unsqueeze(0)  # (1, H*W, D)

    @staticmethod
    def _temporal_interleave(x: torch.Tensor) -> torch.Tensor:
        """Temporal sub-pixel shuffle: (B, 2C, T, H, W) → (B, C, 2T, H, W).

        Same mechanism as WanDecoder3d's upsample3d — splits channels into two
        groups and interleaves them as alternating frames.
        """
        B, C2, T, H, W = x.shape
        C = C2 // 2
        x = x.reshape(B, 2, C, T, H, W)
        x = torch.stack([x[:, 0], x[:, 1]], dim=3)
        return x.reshape(B, C, T * 2, H, W)

    def forward(self, x: torch.Tensor, num_frames: int | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, 16, T_in, H_vae, W_vae)  VAE latent tensor (square).
                T_in may be temporally compressed (e.g. T/4 by the VAE).
                For Wan VAE: T_in = 1 + (T-1)//4 where T is the original
                frame count.
            num_frames: Original frame count (unused after asymmetric
                upsampling fix, kept for API compatibility).
        Returns:
            (B, out_dim, T_out, out_h, out_w)  token tensor for DINOv2 injection.
                When temporal upsampling is active, T_out = 1 + (T_in-1)*factor,
                which exactly matches the original T for valid Wan VAE inputs
                (T = 1 + 4k).  Otherwise T_out = T_in.

        Backward-compatible: when ``temporal_upsample_factor == 1`` and
        ``num_frames`` is None, behaviour is identical to the original
        implementation.
        """
        B, C, T, H, W = x.shape

        # 1) Input projection: (B, 16, T, H, W) → (B, out_dim, T, H, W)
        x = self.input_proj(x)

        # 1.5) Asymmetric learned temporal upsampling: T_vae → T (exact)
        #
        # Wan VAE encodes T = 1+4k frames into T_vae = 1+k latents:
        #   L0 encodes frame 0 alone (1:1), L1..Lk each encode 4 frames.
        # We respect this structure: keep L0 as-is, only 4× upsample L1..Lk.
        #   Result: 1 + k*4 = 1 + 4k = T  (exact, no trilinear trim needed)
        if self.temporal_upsample_factor > 1:
            first_frame = x[:, :, :1, :, :]   # L0: represents 1 input frame
            rest_frames = x[:, :, 1:, :, :]   # L1..Lk: each represents 4 input frames

            for up_layer in self.temporal_ups:
                rest_frames = self._temporal_interleave(up_layer(rest_frames))

            x = torch.cat([first_frame, rest_frames], dim=2)  # (B, D, 1+4k, H, W)

        # 2) Build KV positional encoding (once, then cached)
        if self._kv_pe is None or self._kv_pe.shape[1] != H * W:
            self._kv_pe = self._build_2d_sincos_pos_embed(
                self.out_dim, H, W
            ).to(x.device, x.dtype)
        kv_pos_embed = self._kv_pe
        if kv_pos_embed.device != x.device or kv_pos_embed.dtype != x.dtype:
            kv_pos_embed = kv_pos_embed.to(x.device, x.dtype)
            self._kv_pe = kv_pos_embed

        # 3) Process each frame with cross-attention
        T = x.shape[2]  # may differ from input T after temporal upsampling
        queries_with_pe = self.queries + self.query_pos_embed  # (1, out_h*out_w, D)

        out_frames = []
        for t in range(T):
            frame = x[:, :, t, :, :]  # (B, out_dim, H, W)

            # Flatten spatial: (B, H*W, out_dim) — these are the KV tokens
            kv = frame.flatten(2).permute(0, 2, 1)  # (B, H*W, D)
            kv = kv + kv_pos_embed  # add spatial position info to KV

            # Expand queries for batch
            queries = queries_with_pe.expand(B, -1, -1)

            # Cross-attention layers (with gradient checkpointing in training)
            for layer in self.layers:
                if self.training and torch.is_grad_enabled():
                    queries = torch.utils.checkpoint.checkpoint(
                        layer, queries, kv, use_reentrant=False,
                    )
                else:
                    queries = layer(queries, kv)

            # Output norm
            queries = self.output_norm(queries)

            # Reshape to spatial grid
            frame_out = queries.permute(0, 2, 1).reshape(
                B, self.out_dim, self.out_h, self.out_w
            )
            out_frames.append(frame_out)

        return torch.stack(out_frames, dim=2)
