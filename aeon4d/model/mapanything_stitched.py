"""
MapAnythingGSStitched: a wrapper around MapAnythingGS that replaces the ViT
image encoder with pre-computed stitched features (projected from VAE latents).
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)
from aeon4d._vendor.mapanything.models.mapanything_gs.model import MapAnythingGS


class DepthImageAdapter(nn.Module):
    """Post-hoc image adapter for the depth/geometry DPT head.

    Extracts image features and fuses them with the DPT dense output
    (ray_directions 3ch + depth_along_ray 1ch = 4ch) via a residual
    connection.  The last conv is zero-initialized so the adapter starts
    as an identity — fully backward-compatible with pretrained checkpoints.
    """

    def __init__(self, hidden_dim: int = 64, dense_output_dim: int = 4):
        super().__init__()
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(dense_output_dim + hidden_dim, hidden_dim,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dense_output_dim,
                      kernel_size=1, stride=1, padding=0),
        )
        # Zero-init last conv → initial output is identity (adds zeros)
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, dense_output: torch.Tensor,
                images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dense_output: (B*T, 4, H, W) DPT output (ray_dirs + depth)
            images: (B*T, 3, H, W) ImageNet-normalised input images
        Returns:
            refined: (B*T, 4, H, W)
        """
        img_feat = self.img_encoder(images)
        combined = torch.cat([dense_output, img_feat], dim=1)
        residual = self.fusion(combined)
        return dense_output + residual


class MapAnythingGSStitched(nn.Module):
    """
    Wraps a MapAnythingGS model so that its DINOv2 encoder accepts
    VAE-latent-derived features instead of 3-channel RGB images.

    Two modes controlled by ``stitch_skip_blocks``:

    **Legacy mode** (``stitch_skip_blocks=0``):
        1. Replace the patch embedding Conv2d to accept ``stitch_channels``
           input channels instead of 3 (RGB).
        2. All DINOv2 blocks are kept.

    **Token injection mode** (``stitch_skip_blocks>0``):
        1. The first ``stitch_skip_blocks`` DINOv2 blocks are deleted.
        2. ``stitch_channels`` must equal ``enc_embed_dim`` (1024 for ViT-L).
        3. Conv3D output tokens are injected directly — patch_embed is skipped.
    """

    def __init__(self, model: MapAnythingGS, stitch_channels: int = 16,
                 stitch_skip_blocks: int = 0,
                 depth_image_adapter_config: dict = None):
        super().__init__()
        self.model = model
        self.stitch_skip_blocks = stitch_skip_blocks

        self._enc_embed_dim = model.encoder.enc_embed_dim
        self._patch_size = model.encoder.patch_size
        logger.info(
            f"MapAnythingGSStitched: enc_embed_dim={self._enc_embed_dim}, "
            f"patch_size={self._patch_size}, "
            f"stitch_skip_blocks={stitch_skip_blocks}"
        )

        # Default expected feature map size (set properly in forward())
        self._expected_feat_h = None  # set in forward() before encoder runs
        self._expected_feat_w = None

        if stitch_skip_blocks > 0:
            # ── Token injection mode ──
            assert stitch_channels == self._enc_embed_dim, (
                f"Token injection mode requires stitch_channels == enc_embed_dim "
                f"({self._enc_embed_dim}), got {stitch_channels}"
            )
            vit = self.model.encoder.model

            # DINOv2 may use BlockChunk (when block_chunks>0, default=1):
            # self.blocks is ModuleList of 1 BlockChunk containing all blocks
            # (with Identity() padding). Flatten to a plain ModuleList first.
            if getattr(vit, "chunked_blocks", False):
                flat_blocks = []
                for chunk in vit.blocks:
                    for blk in chunk:
                        if not isinstance(blk, nn.Identity):
                            flat_blocks.append(blk)
                vit.blocks = nn.ModuleList(flat_blocks)
                vit.chunked_blocks = False
                logger.info(
                    f"Flattened {len(flat_blocks)} DINOv2 blocks from BlockChunks"
                )

            # Delete early blocks (delete from front N times)
            for i in range(stitch_skip_blocks):
                logger.info(f"Removing DINOv2 encoder block {i}")
                del vit.blocks[0]
            logger.info(
                f"DINOv2 encoder blocks remaining: {len(vit.blocks)}"
            )
        else:
            # ── Legacy patch_embed mode ──
            old_proj = self.model.encoder.model.patch_embed.proj
            new_proj = nn.Conv2d(
                in_channels=stitch_channels,
                out_channels=old_proj.out_channels,
                kernel_size=old_proj.kernel_size,
                stride=old_proj.stride,
                bias=(old_proj.bias is not None),
            )
            nn.init.xavier_uniform_(new_proj.weight)
            if new_proj.bias is not None and old_proj.bias is not None:
                new_proj.bias.data.copy_(old_proj.bias.data)
            self.model.encoder.model.patch_embed.proj = new_proj
            logger.info(
                f"Replaced patch_embed.proj: Conv2d({stitch_channels}, "
                f"{old_proj.out_channels}, k={old_proj.kernel_size}, "
                f"s={old_proj.stride})"
            )

        # ── Optional depth image adapter ──
        cfg = depth_image_adapter_config or {}
        if cfg.get("enabled", False):
            hidden = cfg.get("hidden_dim", 64)
            self.depth_image_adapter = DepthImageAdapter(
                hidden_dim=hidden, dense_output_dim=4,
            )
            logger.info(
                f"DepthImageAdapter enabled: hidden_dim={hidden}"
            )
        else:
            self.depth_image_adapter = None

    @property
    def enc_embed_dim(self):
        return self._enc_embed_dim

    @property
    def patch_size(self):
        return self._patch_size

    def _encode_vae_latents(self, vae_latents_per_view):
        """
        Encode per-view feature maps through the modified DINOv2 encoder.

        **Legacy mode** (``stitch_skip_blocks == 0``):
            Input is (B, C_stitch, H, W) with H,W divisible by patch_size.
            Calls ``forward_features`` which routes through the replaced
            patch_embed.proj.

        **Token injection mode** (``stitch_skip_blocks > 0``):
            Input is (B, 1024, H_tok, W_tok) where each spatial position is
            already a 1024-dim token.  We manually prepend cls_token, add
            interpolated positional encoding, optionally add register tokens,
            and run through the *remaining* DINOv2 blocks + norm.

        Returns:
            tuple of (B, enc_embed_dim, H_out, W_out) feature tensors.
        """
        num_views = len(vae_latents_per_view)
        all_latents = torch.cat(vae_latents_per_view, dim=0)
        BN, C, H, W = all_latents.shape

        if self.stitch_skip_blocks > 0:
            features = self._encode_token_injection(all_latents, H, W)
        else:
            features = self._encode_patch_embed(all_latents, H, W)

        # In token injection mode, the feature grid (e.g. 32×32) may differ
        # from what downstream modules expect (img_H/patch_size × img_W/patch_size,
        # e.g. 37×37 for 518px images with patch_size=14).  Resize to match.
        if self.stitch_skip_blocks > 0 and len(vae_latents_per_view) > 0:
            # Infer expected feature size from view images (set during forward)
            expected_h = self._expected_feat_h
            expected_w = self._expected_feat_w
            if features.shape[2] != expected_h or features.shape[3] != expected_w:
                features = F.interpolate(
                    features,
                    size=(expected_h, expected_w),
                    mode="bilinear",
                    align_corners=False,
                )

        return features.chunk(num_views, dim=0)

    def _encode_patch_embed(self, all_latents, H, W):
        """Legacy path: forward through replaced patch_embed + all blocks."""
        BN = all_latents.shape[0]
        features = self.model.encoder.model.forward_features(
            all_latents
        )["x_norm_patchtokens"]
        features = features.permute(0, 2, 1)
        features = features.reshape(
            BN, self._enc_embed_dim,
            H // self._patch_size, W // self._patch_size,
        ).contiguous()
        return features

    def _encode_token_injection(self, all_latents, H, W):
        """
        Token injection path: skip patch_embed, inject tokens directly.

        all_latents: (BN, 1024, H_tok, W_tok) — already embed_dim channels.
        H, W: spatial dims of the token grid (e.g. 32, 32).
        """
        from torch.utils.checkpoint import checkpoint as ckpt_fn

        BN = all_latents.shape[0]
        vit = self.model.encoder.model

        # Flatten spatial → sequence: (BN, 1024, H, W) → (BN, H*W, 1024)
        x = all_latents.flatten(2).permute(0, 2, 1)  # (BN, H*W, D)

        # Prepend cls_token
        cls_tokens = vit.cls_token.expand(BN, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (BN, 1+H*W, D)

        # Add interpolated positional encoding
        # interpolate_pos_encoding expects pixel-level sizes (it divides by patch_size
        # internally), so multiply token grid dims back to pixel space.
        x = x + vit.interpolate_pos_encoding(x, H * self._patch_size, W * self._patch_size)

        # Add register tokens (if any)
        if hasattr(vit, "register_tokens") and vit.register_tokens is not None:
            reg = vit.register_tokens.expand(BN, -1, -1)
            x = torch.cat([x[:, :1], reg, x[:, 1:]], dim=1)

        # Run through remaining blocks
        for blk in vit.blocks:
            if self.training:
                x = ckpt_fn(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        x = vit.norm(x)

        # Extract patch tokens (skip cls + register tokens)
        num_reg = getattr(vit, "num_register_tokens", 0)
        patch_tokens = x[:, num_reg + 1:]  # (BN, H*W, D)

        features = patch_tokens.permute(0, 2, 1)  # (BN, D, H*W)
        features = features.reshape(BN, self._enc_embed_dim, H, W).contiguous()
        return features

    def forward(self, vae_latents_per_view, views, feedforward_image=None,
                use_gt_camera=True, use_gt_intrinsics=True,
                return_encoder_features=False,
                grad_through_pred_camera=False):
        """
        Forward pass: encode VAE-derived features through modified DINOv2,
        then run the rest of MapAnythingGS pipeline.

        Args:
            vae_latents_per_view: list of (B, C_stitch, H, W) tensors
                resized to encoder input resolution (e.g. 518x518).
            views: list of view dicts (camera params, img shape, etc.)
            feedforward_image: (B, 3, T, H, W) VAE-decoded pixel video in
                [-1, 1] for the image_adapter in downstream DPT heads.
                If None, falls back to view["img"] (ImageNet-normalised).
            use_gt_camera: whether to use ground truth camera
            use_gt_intrinsics: whether to use ground truth intrinsics
            return_encoder_features: if True, returns (res, gs_params, encoder_features)
                where encoder_features is a list of (B, D, H, W) per view.
        """
        batch_size_per_view, _, height, width = views[0]["img"].shape
        img_shape = (int(height), int(width))
        num_views = len(views)

        # Cache expected feature map size for token injection resizing
        if self.stitch_skip_blocks > 0:
            self._expected_feat_h = int(height) // self._patch_size
            self._expected_feat_w = int(width) // self._patch_size

        # Step 1: encode features through modified DINOv2
        all_encoder_features = list(self._encode_vae_latents(vae_latents_per_view))

        # Capture pre-fusion encoder features for feature alignment loss
        if return_encoder_features:
            pre_fusion_encoder_features = list(all_encoder_features)

        # Step 2: geometric fusion (identical to original)
        with torch.autocast("cuda", enabled=False):
            all_encoder_features = (
                self.model._encode_and_fuse_optional_geometric_inputs(
                    views, all_encoder_features
                )
            )

        # Step 3: scale token
        input_scale_token = (
            self.model.scale_token.unsqueeze(0)
            .unsqueeze(-1)
            .repeat(batch_size_per_view, 1, 1)
        )

        # Step 4: info sharing
        from aeon4d._vendor.uniception.models.info_sharing.base import MultiViewTransformerInput

        info_sharing_input = MultiViewTransformerInput(
            features=all_encoder_features,
            additional_input_tokens=input_scale_token,
        )
        if self.model.info_sharing_return_type == "no_intermediate_features":
            final_feat = self.model.info_sharing(info_sharing_input)
            intermediate_feat = None
        elif self.model.info_sharing_return_type == "intermediate_features":
            final_feat, intermediate_feat = self.model.info_sharing(info_sharing_input)

        # Step 5: prepare dense head inputs (DPT path)
        if self.model.pred_head_type in ["dpt", "dpt+pose"]:
            dense_head_inputs_list = []
            if self.model.use_encoder_features_for_dpt:
                stacked_encoder_features = torch.cat(all_encoder_features, dim=0)
                dense_head_inputs_list.append(stacked_encoder_features)
                dense_head_inputs_list.append(
                    torch.cat(intermediate_feat[0].features, dim=0)
                )
                dense_head_inputs_list.append(
                    torch.cat(intermediate_feat[1].features, dim=0)
                )
                dense_head_inputs_list.append(
                    torch.cat(final_feat.features, dim=0)
                )
            else:
                dense_head_inputs_list.append(
                    torch.cat(intermediate_feat[0].features, dim=0)
                )
                dense_head_inputs_list.append(
                    torch.cat(intermediate_feat[1].features, dim=0)
                )
                dense_head_inputs_list.append(
                    torch.cat(intermediate_feat[2].features, dim=0)
                )
                dense_head_inputs_list.append(
                    torch.cat(final_feat.features, dim=0)
                )
            dense_head_inputs = dense_head_inputs_list
        elif self.model.pred_head_type == "linear":
            dense_head_inputs = torch.cat(final_feat.features, dim=0)
        else:
            raise ValueError(f"Unsupported pred_head_type: {self.model.pred_head_type}")

        # Step 6: downstream heads (DPT forward in bf16 to save memory)
        # DPT heads upsample features to img_shape (e.g. 518×518), producing
        # large activation tensors. bf16 halves memory: ~52GB fp32 → ~26GB bf16.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scale_head_inputs = final_feat.additional_token_features

            imgs_for_adapter = None
            _need_imgs = (
                (self.model.use_image_adapter and self.model.image_adapters is not None)
                or self.depth_image_adapter is not None
            )
            if _need_imgs:
                if feedforward_image is not None:
                    # Use VAE-decoded pixels: (B, 3, T, H, W) → per-frame list
                    # Convert from [-1, 1] to ImageNet normalization to match
                    # what the adapter was pretrained on.
                    _mean = torch.tensor([0.485, 0.456, 0.406],
                                         device=feedforward_image.device).view(1, 3, 1, 1)
                    _std = torch.tensor([0.229, 0.224, 0.225],
                                        device=feedforward_image.device).view(1, 3, 1, 1)
                    ff_frames = []
                    for t in range(feedforward_image.shape[2]):
                        frame_01 = (feedforward_image[:, :, t] + 1.0) / 2.0  # [-1,1]→[0,1]
                        frame_01 = frame_01.clamp(0, 1)
                        # Resize to match view img resolution
                        if frame_01.shape[-1] != width or frame_01.shape[-2] != height:
                            frame_01 = F.interpolate(
                                frame_01, size=(height, width),
                                mode="bilinear", align_corners=False,
                            )
                        frame_norm = (frame_01 - _mean) / _std  # ImageNet normalize
                        ff_frames.append(frame_norm)
                    imgs_for_adapter = torch.cat(ff_frames, dim=0)  # (B*T, 3, H, W)
                else:
                    # Fallback: use GT images from view dicts
                    imgs_for_adapter = torch.cat(
                        [view["img"] for view in views], dim=0
                    )

            dense_final_outputs, dense_final_outputs_gs, pose_final_outputs, scale_final_output = (
                self.model.downstream_head(
                    dense_head_inputs=dense_head_inputs,
                    scale_head_inputs=scale_head_inputs,
                    img_shape=img_shape,
                    memory_efficient_inference=False,
                    imgs_for_adapter=imgs_for_adapter,
                )
            )

        # Step 7: decode scene representation in fp32
        # Geometric operations (quaternion→rotation matrix, ray×depth→3D points)
        # are precision-sensitive — keep in fp32.
        with torch.autocast("cuda", enabled=False):
            # Cast DPT outputs from bf16 → fp32 for geometric precision
            dense_final_outputs_value = dense_final_outputs.value.float()
            pose_final_outputs_value = pose_final_outputs.value.float()

            # Apply depth image adapter (residual refinement with image features)
            if self.depth_image_adapter is not None and imgs_for_adapter is not None:
                dense_final_outputs_value = self.depth_image_adapter(
                    dense_final_outputs_value, imgs_for_adapter.float(),
                )

            # Step 7: decode scene representation — reuse original logic
            # Only the raydirs+depth+pose branch (used by this model) is supported.
            if self.model.scene_rep_type in [
                "raydirs+depth+pose",
                "raydirs+depth+pose+confidence",
                "raydirs+depth+pose+mask",
                "raydirs+depth+pose+confidence+mask",
            ]:
                output_dense_rep = dense_final_outputs_value.permute(
                    0, 2, 3, 1
                ).contiguous()
                output_ray_directions, output_depth_along_ray = output_dense_rep.split(
                    [3, 1], dim=-1
                )

                output_cam_translations, output_cam_quats = (
                    pose_final_outputs_value.split([3, 4], dim=-1)
                )

                from aeon4d._vendor.mapanything.utils.geometry import (
                    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
                )

                output_pts3d = (
                    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                        output_ray_directions,
                        output_depth_along_ray,
                        output_cam_translations,
                        output_cam_quats,
                    )
                )
                output_pts3d_cam = output_ray_directions * output_depth_along_ray

                res = []
                for i in range(num_views):
                    s, e = i * batch_size_per_view, (i + 1) * batch_size_per_view
                    res.append({
                        "pts3d": output_pts3d[s:e],
                        "pts3d_cam": output_pts3d_cam[s:e],
                        "ray_directions": output_ray_directions[s:e],
                        "depth_along_ray": output_depth_along_ray[s:e],
                        "cam_trans": output_cam_translations[s:e],
                        "cam_quats": output_cam_quats[s:e],
                    })

                gs_params = self.model.decoded_gs_params_v2(
                    dense_final_outputs_gs,
                    views,
                    res,
                    batch_size_per_view,
                    num_views,
                    height,
                    width,
                    use_gt_camera,
                    use_gt_intrinsics=use_gt_intrinsics,
                    grad_through_pred_camera=grad_through_pred_camera,
                )
            else:
                raise NotImplementedError(
                    f"scene_rep_type '{self.model.scene_rep_type}' not yet "
                    "supported in the stitched wrapper. Add the branch from "
                    "MapAnythingGS.forward() here."
                )

        if return_encoder_features:
            return res, gs_params, pre_fusion_encoder_features
        return res, gs_params

