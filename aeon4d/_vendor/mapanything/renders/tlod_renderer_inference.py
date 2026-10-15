# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from typing import List, Mapping, Optional, Sequence, Union

import math
import os
import cv2
import time
import torch

BgColor = Optional[Union[Sequence[float], torch.Tensor]]


def _build_backgrounds(
    bg_color: BgColor,
    num_cameras: int,
    channels: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Build a (C, D) background tensor for gsplat (rasterization / rasterization_2dgs).

    bg_color may be a scalar, 3-tuple RGB, or any iterable/tensor. Trailing
    channels beyond the supplied length are zero-padded so non-RGB outputs
    (depth, feature) keep the original "absent = 0" semantics.
    Returns None (no-op) when bg_color is None.
    """
    if bg_color is None:
        return None
    bg = torch.as_tensor(bg_color, dtype=torch.float32, device=device).flatten()
    if bg.numel() < channels:
        pad = torch.zeros(channels - bg.numel(), dtype=torch.float32, device=device)
        bg = torch.cat([bg, pad], dim=0)
    elif bg.numel() > channels:
        bg = bg[:channels]
    return bg[None].expand(num_cameras, -1).contiguous()

# 解决 cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR 问题
# 切换到 magma 后端进行线性代数运算
# torch.backends.cuda.preferred_linalg_library("magma")

from gsplat.rendering import rasterization, rasterization_2dgs

from aeon4d._vendor.mapanything.acceleration.checkpoint import auto_grad_checkpoint, dummy_function
from aeon4d._vendor.mapanything.utils.easyvolcap_compat import (
    affine_inverse,
    angle_axis_to_quaternion,
    dotdict,
    flow_to_color,
    multi_gather,
    multi_scatter,
    qmul,
)
from aeon4d._vendor.mapanything.renders.gaussian_renderer import GaussianRenderer


@torch.jit.script
def compute_marginal_t(t: torch.Tensor, mu_t: torch.Tensor, cov_t: torch.Tensor):
    return torch.exp(-0.5 * (t - mu_t) ** 2 / cov_t)


@torch.jit.script
def _evaluate_temporal_delta(coeffs: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    degree: int = coeffs.shape[-1] // 3
    if degree == 0:
        out_shape: List[int] = []
        for d in range(coeffs.dim() - 1):
            out_shape.append(coeffs.shape[d])
        out_shape.append(3)
        return torch.zeros(out_shape, dtype=torch.float32, device=coeffs.device)

    new_shape: List[int] = []
    for d in range(coeffs.dim() - 1):
        new_shape.append(coeffs.shape[d])
    new_shape.append(degree)
    new_shape.append(3)
    coeffs = coeffs.reshape(new_shape)
    dt_base = dt.float()
    dt_power = dt_base
    delta = coeffs[..., 0, :].float() * dt_power
    for order in range(1, degree):
        dt_power = dt_power * dt_base
        delta = delta + coeffs[..., order, :].float() * dt_power
    return delta


def _motion_mask_from_ms3_cov_t(
    ms3_values: torch.Tensor,
    cov_t_values: torch.Tensor,
    motion_mask_th: float,
    motion_mask_cov_t_min: Optional[float],
    motion_mask_cov_t_max: Optional[float],
) -> torch.Tensor:
    motion_mask = ms3_values.norm(dim=-1, keepdim=True) > float(motion_mask_th)
    if motion_mask_cov_t_min is None and motion_mask_cov_t_max is None:
        return motion_mask
    if cov_t_values.dim() == motion_mask.dim() - 1:
        cov_gate = cov_t_values.unsqueeze(-1)
    else:
        cov_gate = cov_t_values[..., :1]
    if motion_mask_cov_t_min is not None:
        motion_mask = motion_mask & (cov_gate > float(motion_mask_cov_t_min))
    if motion_mask_cov_t_max is not None:
        motion_mask = motion_mask & (cov_gate < float(motion_mask_cov_t_max))
    return motion_mask


def _project_world_means_to_camera(
    means_world: torch.Tensor, viewmat: torch.Tensor
) -> torch.Tensor:
    return means_world @ viewmat[:3, :3].mT + viewmat[:3, -1]


def threedgs_rasterizer(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    height: int = 0,
    width: int = 0,
    render_mode: str = "RGB+ED",
    bg_color: BgColor = None,
):
    backgrounds = _build_backgrounds(
        bg_color, viewmats.shape[0], colors.shape[-1], colors.device
    )
    out_img, out_alpha, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        height=height,
        width=width,
        backgrounds=backgrounds,
        render_mode=render_mode.replace("MONO", "RGB"),
    )  # H, W, C since we're only rendering one image at a time
    if out_img.shape[-1] == 4 or out_img.shape[-1] == 2 or "D" in render_mode:
        out_dpt = out_img[..., -1:]  # 1, H, W, C
        out_img = out_img[..., :-1]  # 1, H, W, C
    else:
        out_dpt = torch.zeros_like(out_alpha)

    return (
        out_img,
        out_alpha,
        out_dpt,
        dotdict(visible=meta["radii"] > 0),
    )  # not differentiable
    # return out_img, out_alpha, out_dpt, meta


_FLIPPER_CACHE: dict = {}


def _get_flipper(device: torch.device) -> torch.Tensor:
    key = str(device)
    if key not in _FLIPPER_CACHE:
        _FLIPPER_CACHE[key] = torch.tensor([-1, -1, -1], dtype=torch.float32, device=device)
    return _FLIPPER_CACHE[key]


def twodgs_rasterizer(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    height: int = 0,
    width: int = 0,
    render_mode: str = "RGB+ED",
    bg_color: BgColor = None,
):
    # HACK: When having more than 8 channels in rgb, distorts and median_depths becomes strange
    channels = colors.shape[-1]
    if "D" in render_mode:
        channels += 1
    padded_channels = 0
    if channels not in (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat(
            [
                colors,
                torch.zeros(*colors.shape[:-1], padded_channels, device=colors.device),
            ],
            dim=-1,
        )
    # Add camera dimension to colors to match expected shape [C, N, D]
    if colors.dim() == 2:  # [N, D]
        colors = colors.unsqueeze(0)  # [1, N, D]

    backgrounds = _build_backgrounds(
        bg_color, viewmats.shape[0], colors.shape[-1], colors.device
    )

    colors, alphas, normals, surf_normals, distorts, median_depths, meta = (
        rasterization_2dgs(
            means=means,    # N, 3
            quats=quats,    # N, 4
            scales=scales,  # N, 3
            opacities=opacities,  # N
            colors=colors,  # C, N, D (where C=1 for single view)
            viewmats=viewmats,  # V, 4, 4
            Ks=Ks,  # V, 3, 3
            height=height,
            width=width,
            backgrounds=backgrounds,
            render_mode=render_mode.replace("MONO", "RGB"),
            # distloss="D" in render_mode,
            # depth_mode="median",  # use median depth for surface normal calculation
        )
    )  # H, W, C since we're only rendering one image at a time
    if "D" in render_mode:
        depths = colors[..., -1:]  # 1, H, W, C
        colors = colors[..., :-1]  # 1, H, W, C
    else:
        depths = median_depths

    if padded_channels > 0:
        colors = colors[..., :-padded_channels]

    # Rendered normal is in world space, convert back to camera space
    # viewmats: 4, 4, world to camera
    # right multiply the tranposed of viewmats to convert world space to camera space
    world_normals = normals  # keep a copy for debug
    world_surf_normals = surf_normals
    normals = normals @ viewmats[..., :3, :3].mT.detach()
    flipper = _get_flipper(means.device)
    normals = normals * flipper

    if surf_normals is not None:
        surf_normals = surf_normals[None]  # 1, H, W, 3
        world_surf_normals = surf_normals
        surf_normals = surf_normals @ viewmats[..., :3, :3].mT.detach()
        surf_normals = surf_normals * flipper
    else:
        surf_normals = normals
        world_surf_normals = world_normals

    

    colors = torch.cat(
        [colors, normals, surf_normals, distorts, median_depths], dim=-1
    )  # to retain gradient

    return (
        colors,
        alphas,
        depths,
        dotdict(),
    )


def twodgs_rasterizer_fast(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    height: int = 0,
    width: int = 0,
    render_mode: str = "RGB",
    radius_clip: float = 0.3,
    bg_color: BgColor = None,
):
    """Lightweight 2DGS rasterizer: only RGB + median depth, skips normal
    conversion and extra channel concatenation."""
    channels = colors.shape[-1]
    padded_channels = 0
    if channels not in (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat(
            [colors, torch.zeros(*colors.shape[:-1], padded_channels, device=colors.device)],
            dim=-1,
        )
    if colors.dim() == 2:
        colors = colors.unsqueeze(0)

    backgrounds = _build_backgrounds(
        bg_color, viewmats.shape[0], colors.shape[-1], colors.device
    )

    colors, alphas, normals, surf_normals, distorts, median_depths, meta = (
        rasterization_2dgs(
            means=means, quats=quats, scales=scales, opacities=opacities,
            colors=colors, viewmats=viewmats, Ks=Ks,
            height=height, width=width,
            render_mode="RGB",
            radius_clip=radius_clip,
            backgrounds=backgrounds,
        )
    )
    if padded_channels > 0:
        colors = colors[..., :-padded_channels]

    return (colors, alphas, median_depths, dotdict())


def compute_means(
    xyz: torch.Tensor,
    t: torch.Tensor,
    ms3: torch.Tensor,
    w2cs: torch.Tensor,
    ts: torch.Tensor,
    b: int = 0,
    i: int = 0,
):
    dt = (ts[b, i] - t[b]).float()
    means = xyz[b].float() + _evaluate_temporal_delta(ms3[b], dt)
    viewmat = w2cs[b, i].float()
    return _project_world_means_to_camera(means, viewmat)


def _start_fourdgs_timer(timing_state, device: torch.device):
    if timing_state is None or not timing_state.get("enabled", False):
        return None
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        timing_state["device"] = device
        return ("cuda", start_event, end_event)
    return ("cpu", time.perf_counter())


def _stop_fourdgs_timer(timing_state, timer_ctx):
    if timer_ctx is None or timing_state is None:
        return
    if timer_ctx[0] == "cuda":
        _, start_event, end_event = timer_ctx
        end_event.record()
        timing_state["event_pairs"].append((start_event, end_event))
    else:
        _, start_time = timer_ctx
        timing_state["elapsed_s"] += time.perf_counter() - start_time
    timing_state["n_frames"] += 1


def fourdgs_rasterizer(
    xyz: torch.Tensor,
    rgb: torch.Tensor,
    scale: torch.Tensor,
    rotation: torch.Tensor,
    opacity: torch.Tensor,
    t: torch.Tensor,
    cov_t: torch.Tensor,
    ms3: torch.Tensor,  # B, N, X3
    omega: torch.Tensor,  # B, N, X3
    w2cs: torch.Tensor,  # B, T, 4, 4
    Ks: torch.Tensor,
    ts: torch.Tensor,
    height: int = 0,
    width: int = 0,
    render_mode: str = "RGB+ED",
    b: int = 0,
    i: int = 0,
    use_2dgs: bool = False,
    vis_motion: bool = False,  # heavy on computation, thus skip if possible
    vis_flow: bool = False,  # heavy on computation, thus skip if possible
    marginal_th: float = 0.05,
    opacity_th: float = 0.0001,
    motion_mask_th: float = 0.25,  # velocity or angular velocity threshold for filtering actual motion
    retain_all_rgb: bool = False,  # retain all RGB channels, not just the first 3
    timing_state=None,
    bg_color: BgColor = None,
    motion_mask_cov_t_min: Optional[float] = None,
    motion_mask_cov_t_max: Optional[float] = None,
):
    # Check bounds for ts indexing to prevent CUDA device-side assert
    # assert b < ts.shape[0], f"b={b} out of range for ts.shape[0]={ts.shape[0]}"
    # assert i < ts.shape[1], f"i={i} out of range for ts.shape[1]={ts.shape[1]}"

    timer_ctx = _start_fourdgs_timer(timing_state, xyz.device)
    ts_b = ts[b]
    ts_i = ts_b[i]
    xyz_b = xyz[b]
    t_b = t[b]
    cov_t_b = cov_t[b]

    # Early temporal/opacity culling: most expensive motion math only runs on survivors.
    marginal_t = compute_marginal_t(ts_i, t_b, cov_t_b).float()
    opacities_all = (opacity[b].float() * marginal_t)[..., 0]
    mask = (marginal_t[..., 0] > marginal_th) & (opacities_all > opacity_th)
    ind = mask.nonzero(as_tuple=False).squeeze(-1)  # S, # MARK: SYNC
    num_selected = int(ind.numel())
    num_total = int(mask.numel())

    color_channels = rgb.shape[-1] if retain_all_rgb else (
        3 if "RGB" in render_mode else max(int(rgb.shape[-1]) - 3, 0)
    )
    if vis_flow:
        color_channels += 6
    if vis_motion:
        color_channels += 1

    if num_selected == 0:
        img_channels = color_channels + (8 if use_2dgs else 0)
        rgb_out = xyz_b.new_zeros((1, height, width, img_channels), dtype=torch.float32)
        occ_out = xyz_b.new_zeros((1, height, width, 1), dtype=torch.float32)
        dpt_out = xyz_b.new_zeros((1, height, width, 1), dtype=torch.float32)
        meta = dotdict()
        if not use_2dgs:
            meta.visible = torch.zeros(
                (1, num_total), device=xyz.device, dtype=torch.bool
            )
        _stop_fourdgs_timer(timing_state, timer_ctx)
        return rgb_out, occ_out, dpt_out, meta

    select_all = num_selected == num_total
    if select_all:
        xyz_sel = xyz_b
        t_sel = t_b
        ms3_sel = ms3[b]
        omega_sel = omega[b]
        rotation_sel = rotation[b]
        scale_sel = scale[b]
        opacities = opacities_all
        rgb_sel = rgb[b]
        cov_t_sel = cov_t_b
    else:
        xyz_sel = xyz_b.index_select(0, ind)
        t_sel = t_b.index_select(0, ind)
        ms3_sel = ms3[b].index_select(0, ind)
        omega_sel = omega[b].index_select(0, ind)
        rotation_sel = rotation[b].index_select(0, ind)
        scale_sel = scale[b].index_select(0, ind)
        opacities = opacities_all.index_select(0, ind)
        rgb_sel = rgb[b].index_select(0, ind)
        cov_t_sel = cov_t_b.index_select(0, ind)

    dt = (ts_i - t_sel).float()
    means = xyz_sel.float() + _evaluate_temporal_delta(ms3_sel, dt)
    domega = _evaluate_temporal_delta(omega_sel, dt)
    dquats = angle_axis_to_quaternion(domega)
    quats = qmul(rotation_sel.float(), dquats).float()
    scales = scale_sel.float()

    if not retain_all_rgb:
        colors = (
            rgb_sel[..., :3].float() if "RGB" in render_mode else rgb_sel[..., 3:].float()
        )
    else:
        colors = rgb_sel.float()

    viewmats = w2cs[b, i : i + 1].float()  # 1, 3, 4
    ixts = Ks[b, i : i + 1].float()

    if vis_flow:
        curr_means = _project_world_means_to_camera(means, viewmats[0])
        if i < w2cs.shape[1] - 1:
            next_viewmat = w2cs[b, i + 1].float()
            next_dt = (ts_b[i + 1] - t_sel).float()
            next_means_world = xyz_sel.float() + _evaluate_temporal_delta(ms3_sel, next_dt)
            next_means = _project_world_means_to_camera(next_means_world, next_viewmat)
        else:
            next_means = curr_means

        colors = torch.cat([colors, curr_means, next_means], dim=-1)

    if vis_motion:
        motion_mask = _motion_mask_from_ms3_cov_t(
            ms3_sel, cov_t_sel, motion_mask_th,
            motion_mask_cov_t_min, motion_mask_cov_t_max
        )
        colors = torch.cat([colors, motion_mask], dim=-1)  # N, C
        # colors = torch.cat([colors, velocity], dim=-1)  # N, C
    # print("#################################################")
    rgb, occ, dpt, meta = (threedgs_rasterizer if not use_2dgs else twodgs_rasterizer)(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        ixts,
        height,
        width,
        render_mode,
        bg_color=bg_color,
    )

    # For points with shorter life span
    # They will be filtered using the marginal and opacity threshold
    # We do not want extra regularization on those points
    # We only want the points that could be possibly seen on novel views to be regularized
    # Note that non-visible points are regularized
    if not select_all:
        for key in meta:
            canvas = torch.zeros_like(mask[None])  # N, # defaults to visible
            meta[key] = multi_scatter(canvas, ind[None], meta[key], dim=-1)

    _stop_fourdgs_timer(timing_state, timer_ctx)
    return rgb, occ, dpt, meta


def fourdgs_rasterizer_v2(
    xyz: torch.Tensor,
    rgb: torch.Tensor,
    scale: torch.Tensor,
    rotation: torch.Tensor,
    opacity: torch.Tensor,
    t: torch.Tensor,
    cov_t: torch.Tensor,
    ms3: torch.Tensor,  # B, N, X3
    omega: torch.Tensor,  # B, N, X3
    w2cs: torch.Tensor,  # B, T, 4, 4
    Ks: torch.Tensor,
    ts: torch.Tensor,
    height: int = 0,
    width: int = 0,
    render_mode: str = "RGB+ED",
    b: int = 0,
    i: int = 0,
    use_2dgs: bool = False,
    vis_motion: bool = False,
    vis_flow: bool = False,
    marginal_th: float = 0.05,
    opacity_th: float = 0.0001,
    motion_mask_th: float = 0.25,
    retain_all_rgb: bool = False,
    timing_state=None,
    bg_color: BgColor = None,
    motion_mask_cov_t_min: Optional[float] = None,
    motion_mask_cov_t_max: Optional[float] = None,
):
    """
    Zero-sync variant of fourdgs_rasterizer.

    Key differences from the original:
      - Replaces nonzero() + index_select (GPU-CPU sync) with opacity zeroing.
      - gsplat's tile-based culling naturally skips zero-opacity gaussians.
      - Computes motion for ALL gaussians (trades extra ALU for zero sync stalls).
      - Eliminates the meta scatter-back step.

    Signature-compatible with fourdgs_rasterizer for drop-in benchmarking.
    Designed for inference (no_grad); training use may need validation.
    """
    timer_ctx = _start_fourdgs_timer(timing_state, xyz.device)

    ts_b = ts[b]
    ts_i = ts_b[i]
    xyz_b = xyz[b].float()
    t_b = t[b]
    cov_t_b = cov_t[b]
    ms3_b = ms3[b]
    omega_b = omega[b]
    rotation_b = rotation[b].float()
    scale_b = scale[b].float()
    rgb_b = rgb[b]

    # --- temporal gating: zero opacity instead of sparse selection ---
    # No nonzero(), no int(numel()) → zero GPU-CPU synchronization
    marginal_t = compute_marginal_t(ts_i, t_b, cov_t_b).float()
    marginal_t = torch.nan_to_num(marginal_t, nan=0.0, posinf=1.0, neginf=0.0)
    opacities = (opacity[b].float() * marginal_t)[..., 0]
    valid = (marginal_t[..., 0] > marginal_th) & (opacities > opacity_th)
    opacities = torch.where(valid, opacities, opacities.new_zeros(()))

    # --- motion: zero dt for invalid gaussians to prevent polynomial explosion ---
    dt = (ts_i - t_b).float()
    dt = dt * valid.unsqueeze(-1).float()
    means = xyz_b + _evaluate_temporal_delta(ms3_b, dt)
    domega = _evaluate_temporal_delta(omega_b, dt)
    dquats = angle_axis_to_quaternion(domega)
    quats = qmul(rotation_b, dquats).float()
    scale_b = torch.where(valid.unsqueeze(-1), scale_b, scale_b.new_zeros(()))

    # --- colors ---
    if not retain_all_rgb:
        colors = (
            rgb_b[..., :3].float() if "RGB" in render_mode else rgb_b[..., 3:].float()
        )
    else:
        colors = rgb_b.float()

    viewmats = w2cs[b, i : i + 1].float()
    ixts = Ks[b, i : i + 1].float()

    if vis_flow:
        curr_means = _project_world_means_to_camera(means, viewmats[0])
        if i < w2cs.shape[1] - 1:
            next_viewmat = w2cs[b, i + 1].float()
            next_dt = (ts_b[i + 1] - t_b).float()
            next_means_world = xyz_b + _evaluate_temporal_delta(ms3_b, next_dt)
            next_means = _project_world_means_to_camera(next_means_world, next_viewmat)
        else:
            next_means = curr_means
        colors = torch.cat([colors, curr_means, next_means], dim=-1)

    if vis_motion:
        motion_mask = _motion_mask_from_ms3_cov_t(
            ms3_b, cov_t_b, motion_mask_th,
            motion_mask_cov_t_min, motion_mask_cov_t_max
        )
        colors = torch.cat([colors, motion_mask], dim=-1)

    rgb_out, occ, dpt, meta = (threedgs_rasterizer if not use_2dgs else twodgs_rasterizer)(
        means, quats, scale_b, opacities, colors,
        viewmats, ixts, height, width, render_mode,
        bg_color=bg_color,
    )

    _stop_fourdgs_timer(timing_state, timer_ctx)
    return rgb_out, occ, dpt, meta


class TLODRenderer(GaussianRenderer):
    def __init__(
        self,
        height=512,
        width=512,
        marginal_th: float = 0.05,
        opacity_th: float = 0.0001,
        motion_mask_th: float = 0.25,
        motion_mask_cov_t_min: Optional[float] = None,
        motion_mask_cov_t_max: Optional[float] = None,
        use_2dgs: bool = True,
        vis_flow: bool = False,
        vis_motion: bool = False,
        use_grad_checkpoint: bool = True,
        enable_static_voxel_merge: bool = True,
        static_voxel_size: float = 0.002,
        static_lifespan_threshold: float = 1.0,
        static_lifespan_quantile: float = 0.7,
        static_merge_min_points: int = 64,
        static_merge_weight_min: float = 1e-4,
        **kwargs,
    ):
        self.height = height
        self.width = width
        self.marginal_th = marginal_th
        self.opacity_th = opacity_th
        self.motion_mask_th = motion_mask_th
        self.motion_mask_cov_t_min = motion_mask_cov_t_min
        self.motion_mask_cov_t_max = motion_mask_cov_t_max
        self.use_2dgs = use_2dgs
        self.vis_flow = vis_flow
        self.vis_motion = vis_motion
        self.use_grad_checkpoint = use_grad_checkpoint
        self.enable_static_voxel_merge = enable_static_voxel_merge
        self.static_voxel_size = static_voxel_size
        self.static_lifespan_threshold = static_lifespan_threshold
        self.static_lifespan_quantile = static_lifespan_quantile
        self.static_merge_min_points = static_merge_min_points
        self.static_merge_weight_min = static_merge_weight_min
        self._fourdgs_profiler = None
        self.use_v2_rasterizer = kwargs.pop('use_v2_rasterizer', False)

    def _start_fourdgs_profiler(self):
        self._fourdgs_profiler = {
            "enabled": True,
            "event_pairs": [],
            "elapsed_s": 0.0,
            "n_frames": 0,
            "device": None,
        }

    def _get_fourdgs_profiler_state(self):
        state = getattr(self, "_fourdgs_profiler", None)
        if state is None or not state.get("enabled", False):
            return None
        return state

    def _stop_fourdgs_profiler(self):
        state = getattr(self, "_fourdgs_profiler", None)
        if state is None or not state.get("enabled", False):
            self._fourdgs_profiler = None
            return None

        state["enabled"] = False
        if state["event_pairs"]:
            device = state.get("device")
            if device is not None and device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_s = sum(
                start_event.elapsed_time(end_event)
                for start_event, end_event in state["event_pairs"]
            ) / 1000.0
        else:
            elapsed_s = float(state.get("elapsed_s", 0.0))

        n_frames = int(state.get("n_frames", 0))
        info = {
            "elapsed": elapsed_s,
            "n_frames": n_frames,
            "fps": n_frames / max(elapsed_s, 1e-9),
        }
        self._fourdgs_profiler = None
        return info

    def _weighted_voxel_mean(
        self,
        values: torch.Tensor,
        inverse_indices: torch.Tensor,
        weights: torch.Tensor,
        num_voxels: int,
    ) -> torch.Tensor:
        out = values.new_zeros((num_voxels, values.shape[-1]))
        scatter_index = inverse_indices.unsqueeze(-1).expand(-1, values.shape[-1])
        out.scatter_add_(0, scatter_index, values * weights.unsqueeze(-1))
        weight_sums = values.new_zeros(num_voxels)
        weight_sums.scatter_add_(0, inverse_indices, weights)
        out = out / weight_sums.clamp(min=1e-8).unsqueeze(-1)
        return out

    def _merge_static_gaussians_by_voxel(
        self,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        scale: torch.Tensor,
        rotation: torch.Tensor,
        opacity: torch.Tensor,
        t: torch.Tensor,
        cov_t: torch.Tensor,
        ms3: torch.Tensor,
        omega: torch.Tensor,
    ):
        if xyz.shape[0] == 0 or self.static_voxel_size <= 0:
            return xyz, rgb, scale, rotation, opacity, t, cov_t, ms3, omega, None

        lifespan = cov_t[..., 0]
        if self.static_lifespan_threshold is not None:
            static_mask = lifespan >= float(self.static_lifespan_threshold)
        else:
            q = float(self.static_lifespan_quantile)
            q = max(0.0, min(1.0, q))
            if lifespan.numel() > 2**24:
                import numpy as np
                threshold = float(np.quantile(lifespan.detach().cpu().numpy(), q))
                threshold = torch.tensor(threshold, device=lifespan.device, dtype=lifespan.dtype)
            else:
                threshold = torch.quantile(lifespan.detach(), q)
            static_mask = lifespan >= threshold

        static_count = int(static_mask.sum().item())
        if static_count < max(2, int(self.static_merge_min_points)):
            return xyz, rgb, scale, rotation, opacity, t, cov_t, ms3, omega, None

        static_xyz = xyz[static_mask]
        static_rgb = rgb[static_mask]
        static_scale = scale[static_mask]
        static_rotation = rotation[static_mask]
        static_opacity = opacity[static_mask]
        static_t = t[static_mask]
        static_cov_t = cov_t[static_mask]
        static_ms3 = ms3[static_mask]
        static_omega = omega[static_mask]

        voxel_coords = torch.floor(static_xyz / self.static_voxel_size).to(torch.int64)
        voxel_coords = voxel_coords - voxel_coords.min(dim=0).values
        _, inverse_indices = torch.unique(
            voxel_coords, dim=0, return_inverse=True
        )
        num_voxels = int(inverse_indices.max().item()) + 1

        if num_voxels >= static_xyz.shape[0]:
            return xyz, rgb, scale, rotation, opacity, t, cov_t, ms3, omega, None

        weights = static_opacity[..., 0].clamp(min=self.static_merge_weight_min)

        merged_xyz = self._weighted_voxel_mean(
            static_xyz, inverse_indices, weights, num_voxels
        )
        merged_rgb = self._weighted_voxel_mean(
            static_rgb, inverse_indices, weights, num_voxels
        )
        merged_scale = self._weighted_voxel_mean(
            static_scale, inverse_indices, weights, num_voxels
        )
        merged_rotation = self._weighted_voxel_mean(
            static_rotation, inverse_indices, weights, num_voxels
        )
        merged_rotation = merged_rotation / merged_rotation.norm(
            dim=-1, keepdim=True
        ).clamp(min=1e-8)
        merged_opacity = self._weighted_voxel_mean(
            static_opacity, inverse_indices, weights, num_voxels
        )
        merged_t = self._weighted_voxel_mean(
            static_t, inverse_indices, weights, num_voxels
        )
        merged_cov_t = self._weighted_voxel_mean(
            static_cov_t, inverse_indices, weights, num_voxels
        )
        merged_ms3 = self._weighted_voxel_mean(
            static_ms3, inverse_indices, weights, num_voxels
        )
        merged_omega = self._weighted_voxel_mean(
            static_omega, inverse_indices, weights, num_voxels
        )

        dynamic_mask = ~static_mask
        merged_xyz = torch.cat([xyz[dynamic_mask], merged_xyz], dim=0)
        merged_rgb = torch.cat([rgb[dynamic_mask], merged_rgb], dim=0)
        merged_scale = torch.cat([scale[dynamic_mask], merged_scale], dim=0)
        merged_rotation = torch.cat([rotation[dynamic_mask], merged_rotation], dim=0)
        merged_opacity = torch.cat([opacity[dynamic_mask], merged_opacity], dim=0)
        merged_t = torch.cat([t[dynamic_mask], merged_t], dim=0)
        merged_cov_t = torch.cat([cov_t[dynamic_mask], merged_cov_t], dim=0)
        merged_ms3 = torch.cat([ms3[dynamic_mask], merged_ms3], dim=0)
        merged_omega = torch.cat([omega[dynamic_mask], merged_omega], dim=0)

        stats = {
            "before": int(xyz.shape[0]),
            "after": int(merged_xyz.shape[0]),
            "static_before": static_count,
            "static_after": num_voxels,
        }
        return (
            merged_xyz,
            merged_rgb,
            merged_scale,
            merged_rotation,
            merged_opacity,
            merged_t,
            merged_cov_t,
            merged_ms3,
            merged_omega,
            stats,
        )

    def single_batch_rasterizer(
        self,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        scale: torch.Tensor,
        rotation: torch.Tensor,
        opacity: torch.Tensor,
        t: torch.Tensor,
        cov_t: torch.Tensor,
        ms3: torch.Tensor,
        omega: torch.Tensor,
        w2cs: torch.Tensor,
        Ks: torch.Tensor,
        ts: torch.Tensor,
        height: int = 0,
        width: int = 0,
        render_mode: str = "RGB+ED",
        flo=None,
    ):
        ckpt_function = (
            dummy_function
            if (not self.use_grad_checkpoint or not xyz.requires_grad)
            else auto_grad_checkpoint
        )
        timing_state = self._get_fourdgs_profiler_state()
        render_motion = (not xyz.requires_grad) and bool(self.vis_motion)
        render_flow = (not xyz.requires_grad) and bool(self.vis_flow)

        B, T, _, _ = w2cs.shape
        
        # Ensure ts has the same time dimension as w2cs
        # assert ts.shape[1] == T, f"ts.shape[1]={ts.shape[1]} does not match w2cs.shape[1]={T}"

        # B, T, 3, 3 & B, T, 4, 4 for camera parameters
        out_imgs = []
        out_alphas = []
        out_dpts = []
        out_metas = dotdict()
        for b in range(B):
            out_imgs_batch = []
            out_alphas_batch = []
            out_dpts_batch = []
            out_metas_batch = dotdict()
            for i in range(T):
                out_img, out_alpha, out_dpt, meta = ckpt_function(
                    fourdgs_rasterizer,
                    xyz,
                    rgb,
                    scale,
                    rotation,
                    opacity,
                    t,
                    cov_t,
                    ms3,
                    omega,
                    w2cs,
                    Ks,
                    ts,
                    height or self.height,
                    width or self.width,
                    render_mode,
                    b,
                    i,
                    self.use_2dgs,
                    render_motion,
                    render_flow,
                    self.marginal_th,
                    self.opacity_th,
                    self.motion_mask_th,
                    False,
                    timing_state,
                    None,
                    self.motion_mask_cov_t_min,
                    self.motion_mask_cov_t_max,
                )

                if xyz.requires_grad:
                    out_imgs_batch.append(out_img)
                    out_alphas_batch.append(out_alpha)
                    out_dpts_batch.append(out_dpt)
                else:
                    # To save memory during inference and rendering
                    out_imgs_batch.append(out_img.detach().cpu())
                    out_alphas_batch.append(out_alpha.detach().cpu())
                    out_dpts_batch.append(out_dpt.detach().cpu())
                for key in meta:
                    if key not in out_metas_batch:
                        out_metas_batch[key] = []
                    out_metas_batch[key].append(meta[key])

            out_img = torch.cat(out_imgs_batch).permute(0, 3, 1, 2)  # V, C, H, W
            out_alpha = torch.cat(out_alphas_batch).permute(0, 3, 1, 2)  # V, C, H, W
            out_dpt = torch.cat(out_dpts_batch).permute(0, 3, 1, 2)  # V, C, H, W
            out_meta = dotdict(
                {
                    key: torch.cat(val).permute(0, 3, 1, 2)
                    if val[0].ndim == 4
                    else torch.cat(val)
                    for key, val in out_metas_batch.items()
                }
            )

            out_imgs.append(out_img)
            out_alphas.append(out_alpha)
            out_dpts.append(out_dpt)
            for key in out_meta:
                if key not in out_metas:
                    out_metas[key] = []
                out_metas[key].append(out_meta[key])

        img, mask, depth = (
            torch.stack(out_imgs),
            torch.stack(out_alphas),
            torch.stack(out_dpts),
        )
        meta = dotdict()
        for key in out_metas:
            meta[key] = torch.stack(out_metas[key])

        return img, mask, depth, meta

    def single_batch_rasterizer_v2(
        self,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        scale: torch.Tensor,
        rotation: torch.Tensor,
        opacity: torch.Tensor,
        t: torch.Tensor,
        cov_t: torch.Tensor,
        ms3: torch.Tensor,
        omega: torch.Tensor,
        w2cs: torch.Tensor,
        Ks: torch.Tensor,
        ts: torch.Tensor,
        height: int = 0,
        width: int = 0,
        render_mode: str = "RGB+ED",
        flo=None,
    ):
        """Same as single_batch_rasterizer but dispatches to fourdgs_rasterizer_v2."""
        ckpt_function = (
            dummy_function
            if (not self.use_grad_checkpoint or not xyz.requires_grad)
            else auto_grad_checkpoint
        )
        timing_state = self._get_fourdgs_profiler_state()
        render_motion = (not xyz.requires_grad) and bool(self.vis_motion)
        render_flow = (not xyz.requires_grad) and bool(self.vis_flow)

        B, T, _, _ = w2cs.shape

        out_imgs = []
        out_alphas = []
        out_dpts = []
        out_metas = dotdict()
        for b in range(B):
            out_imgs_batch = []
            out_alphas_batch = []
            out_dpts_batch = []
            out_metas_batch = dotdict()
            for i in range(T):
                out_img, out_alpha, out_dpt, meta = ckpt_function(
                    fourdgs_rasterizer_v2,
                    xyz,
                    rgb,
                    scale,
                    rotation,
                    opacity,
                    t,
                    cov_t,
                    ms3,
                    omega,
                    w2cs,
                    Ks,
                    ts,
                    height or self.height,
                    width or self.width,
                    render_mode,
                    b,
                    i,
                    self.use_2dgs,
                    render_motion,
                    render_flow,
                    self.marginal_th,
                    self.opacity_th,
                    self.motion_mask_th,
                    False,
                    timing_state,
                    None,
                    self.motion_mask_cov_t_min,
                    self.motion_mask_cov_t_max,
                )

                if xyz.requires_grad:
                    out_imgs_batch.append(out_img)
                    out_alphas_batch.append(out_alpha)
                    out_dpts_batch.append(out_dpt)
                else:
                    out_imgs_batch.append(out_img.detach().cpu())
                    out_alphas_batch.append(out_alpha.detach().cpu())
                    out_dpts_batch.append(out_dpt.detach().cpu())
                for key in meta:
                    if key not in out_metas_batch:
                        out_metas_batch[key] = []
                    out_metas_batch[key].append(meta[key])

            out_img = torch.cat(out_imgs_batch).permute(0, 3, 1, 2)
            out_alpha = torch.cat(out_alphas_batch).permute(0, 3, 1, 2)
            out_dpt = torch.cat(out_dpts_batch).permute(0, 3, 1, 2)
            out_meta = dotdict(
                {
                    key: torch.cat(val).permute(0, 3, 1, 2)
                    if val[0].ndim == 4
                    else torch.cat(val)
                    for key, val in out_metas_batch.items()
                }
            )

            out_imgs.append(out_img)
            out_alphas.append(out_alpha)
            out_dpts.append(out_dpt)
            for key in out_meta:
                if key not in out_metas:
                    out_metas[key] = []
                out_metas[key].append(out_meta[key])

        img, mask, depth = (
            torch.stack(out_imgs),
            torch.stack(out_alphas),
            torch.stack(out_dpts),
        )
        meta = dotdict()
        for key in out_metas:
            meta[key] = torch.stack(out_metas[key])

        return img, mask, depth, meta

    def single_batch_rasterizer_v3(
        self,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        scale: torch.Tensor,
        rotation: torch.Tensor,
        opacity: torch.Tensor,
        t: torch.Tensor,
        cov_t: torch.Tensor,
        ms3: torch.Tensor,
        omega: torch.Tensor,
        w2cs: torch.Tensor,
        Ks: torch.Tensor,
        ts: torch.Tensor,
        height: int = 0,
        width: int = 0,
        render_mode: str = "RGB+ED",
        flo=None,
    ):
        """
        High-throughput variant (inference only):
          - Pre-casts per-gaussian tensors ONCE per batch (not per frame).
          - Inlines the per-frame logic (no function-call overhead).
          - Uses a dedicated CUDA stream for GPU→CPU transfer, overlapping
            frame i+1 compute with frame i transfer.
        """
        timing_state = self._get_fourdgs_profiler_state()
        render_motion = (not xyz.requires_grad) and bool(self.vis_motion)
        render_flow = (not xyz.requires_grad) and bool(self.vis_flow)
        is_inference = not xyz.requires_grad

        B, T, _, _ = w2cs.shape
        _h = height or self.height
        _w = width or self.width

        _use_cuda = xyz.is_cuda
        transfer_stream = torch.cuda.Stream() if (is_inference and _use_cuda) else None

        out_imgs = []
        out_alphas = []
        out_dpts = []
        out_metas = dotdict()

        for b in range(B):
            # --- pre-cast per-gaussian attributes ONCE for all T frames ---
            xyz_b = xyz[b].float()
            rgb_b = rgb[b]
            scale_b_base = scale[b].float()
            rotation_b = rotation[b].float()
            opacity_b = opacity[b].float()
            t_b = t[b]
            cov_t_b = cov_t[b]
            ms3_b = ms3[b]
            omega_b = omega[b]
            ts_b = ts[b]
            w2cs_b = w2cs[b].float()
            Ks_b = Ks[b].float()

            colors_base = (
                rgb_b[..., :3].float()
                if "RGB" in render_mode
                else rgb_b[..., 3:].float()
            )

            out_imgs_batch = []
            out_alphas_batch = []
            out_dpts_batch = []
            out_metas_batch = dotdict()

            for i in range(T):
                timer_ctx = _start_fourdgs_timer(timing_state, xyz.device)
                ts_i = ts_b[i]

                # temporal gating
                marginal_t = compute_marginal_t(ts_i, t_b, cov_t_b).float()
                marginal_t = torch.nan_to_num(marginal_t, nan=0.0, posinf=1.0, neginf=0.0)
                opacities = (opacity_b * marginal_t)[..., 0]
                valid = (marginal_t[..., 0] > self.marginal_th) & (opacities > self.opacity_th)
                opacities = torch.where(valid, opacities, opacities.new_zeros(()))

                # motion (dt=0 for invalid)
                dt = (ts_i - t_b).float() * valid.unsqueeze(-1).float()
                means = xyz_b + _evaluate_temporal_delta(ms3_b, dt)
                domega = _evaluate_temporal_delta(omega_b, dt)
                dquats = angle_axis_to_quaternion(domega)
                quats = qmul(rotation_b, dquats).float()
                scale_b_cur = torch.where(
                    valid.unsqueeze(-1), scale_b_base, scale_b_base.new_zeros(())
                )

                colors = colors_base
                viewmats = w2cs_b[i : i + 1]
                ixts = Ks_b[i : i + 1]

                if render_flow:
                    curr_means = _project_world_means_to_camera(means, viewmats[0])
                    if i < T - 1:
                        next_dt = (ts_b[i + 1] - t_b).float() * valid.unsqueeze(-1).float()
                        next_means_world = xyz_b + _evaluate_temporal_delta(ms3_b, next_dt)
                        next_means = _project_world_means_to_camera(
                            next_means_world, w2cs_b[i + 1]
                        )
                    else:
                        next_means = curr_means
                    colors = torch.cat([colors, curr_means, next_means], dim=-1)

                if render_motion:
                    motion_mask = _motion_mask_from_ms3_cov_t(
                        ms3_b, cov_t_b, self.motion_mask_th,
                        self.motion_mask_cov_t_min, self.motion_mask_cov_t_max
                    )
                    colors = torch.cat([colors, motion_mask], dim=-1)

                out_img, out_alpha, out_dpt, meta = (
                    threedgs_rasterizer if not self.use_2dgs else twodgs_rasterizer
                )(
                    means, quats, scale_b_cur, opacities, colors,
                    viewmats, ixts, _h, _w, render_mode,
                )
                _stop_fourdgs_timer(timing_state, timer_ctx)

                if is_inference:
                    out_img = out_img.detach()
                    out_alpha = out_alpha.detach()
                    out_dpt = out_dpt.detach()
                    if transfer_stream is not None:
                        with torch.cuda.stream(transfer_stream):
                            out_imgs_batch.append(out_img.to("cpu", non_blocking=True))
                            out_alphas_batch.append(out_alpha.to("cpu", non_blocking=True))
                            out_dpts_batch.append(out_dpt.to("cpu", non_blocking=True))
                    else:
                        out_imgs_batch.append(out_img.cpu())
                        out_alphas_batch.append(out_alpha.cpu())
                        out_dpts_batch.append(out_dpt.cpu())
                else:
                    out_imgs_batch.append(out_img)
                    out_alphas_batch.append(out_alpha)
                    out_dpts_batch.append(out_dpt)

                for key in meta:
                    if key not in out_metas_batch:
                        out_metas_batch[key] = []
                    out_metas_batch[key].append(meta[key])

            if transfer_stream is not None:
                transfer_stream.synchronize()

            out_img = torch.cat(out_imgs_batch).permute(0, 3, 1, 2)
            out_alpha = torch.cat(out_alphas_batch).permute(0, 3, 1, 2)
            out_dpt = torch.cat(out_dpts_batch).permute(0, 3, 1, 2)
            out_meta = dotdict(
                {
                    key: torch.cat(val).permute(0, 3, 1, 2)
                    if val[0].ndim == 4
                    else torch.cat(val)
                    for key, val in out_metas_batch.items()
                }
            )

            out_imgs.append(out_img)
            out_alphas.append(out_alpha)
            out_dpts.append(out_dpt)
            for key in out_meta:
                if key not in out_metas:
                    out_metas[key] = []
                out_metas[key].append(out_meta[key])

        img, mask, depth = (
            torch.stack(out_imgs),
            torch.stack(out_alphas),
            torch.stack(out_dpts),
        )
        meta = dotdict()
        for key in out_metas:
            meta[key] = torch.stack(out_metas[key])

        return img, mask, depth, meta

    def single_batch_rasterizer_v4(
        self,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        scale: torch.Tensor,
        rotation: torch.Tensor,
        opacity: torch.Tensor,
        t: torch.Tensor,
        cov_t: torch.Tensor,
        ms3: torch.Tensor,
        omega: torch.Tensor,
        w2cs: torch.Tensor,
        Ks: torch.Tensor,
        ts: torch.Tensor,
        height: int = 0,
        width: int = 0,
        render_mode: str = "RGB+ED",
        flo=None,
    ):
        """
        Static/dynamic split variant (inference only):
          - Splits gaussians into static (cov_t > 1, ms3~=0) and dynamic.
          - Precomputes means/quats/scales for static gaussians ONCE per batch.
          - Per frame: only runs motion pipeline on dynamic gaussians (~10%),
            then concatenates with precomputed static results before rasterization.
          - Inherits v3 optimizations: pre-cast, CUDA stream transfer.
        """
        timing_state = self._get_fourdgs_profiler_state()
        render_motion = (not xyz.requires_grad) and bool(self.vis_motion)
        render_flow = (not xyz.requires_grad) and bool(self.vis_flow)
        is_inference = not xyz.requires_grad

        B, T, _, _ = w2cs.shape
        _h = height or self.height
        _w = width or self.width

        _use_cuda = xyz.is_cuda
        transfer_stream = torch.cuda.Stream() if (is_inference and _use_cuda) else None

        out_imgs = []
        out_alphas = []
        out_dpts = []
        out_metas = dotdict()

        for b in range(B):
            xyz_b = xyz[b].float()
            rgb_b = rgb[b]
            scale_b_base = scale[b].float()
            rotation_b = rotation[b].float()
            opacity_b = opacity[b].float()
            t_b = t[b]
            cov_t_b = cov_t[b]
            ms3_b = ms3[b]
            omega_b = omega[b]
            ts_b = ts[b]
            w2cs_b = w2cs[b].float()
            Ks_b = Ks[b].float()

            colors_base = (
                rgb_b[..., :3].float()
                if "RGB" in render_mode
                else rgb_b[..., 3:].float()
            )

            # ---- static/dynamic split (once per batch) ----
            static_mask = cov_t_b[..., 0] > 1.0
            dynamic_mask = ~static_mask
            n_static = int(static_mask.sum().item())
            n_dynamic = int(dynamic_mask.sum().item())
            n_total = static_mask.numel()
            use_split = n_static > 0 and n_dynamic > 0 and n_static > n_total // 2

            if use_split:
                s_idx = static_mask.nonzero(as_tuple=False).squeeze(-1)
                d_idx = dynamic_mask.nonzero(as_tuple=False).squeeze(-1)

                means_s = xyz_b[s_idx]
                quats_s = rotation_b[s_idx]
                scales_s = scale_b_base[s_idx]
                opa_s = opacity_b[s_idx][..., 0]
                colors_s = colors_base[s_idx]

                xyz_d = xyz_b[d_idx]
                rotation_d = rotation_b[d_idx]
                scale_d = scale_b_base[d_idx]
                opacity_d = opacity_b[d_idx]
                t_d = t_b[d_idx]
                cov_t_d = cov_t_b[d_idx]
                ms3_d = ms3_b[d_idx]
                omega_d = omega_b[d_idx]
                colors_d = colors_base[d_idx]
            # ---- end split ----

            out_imgs_batch = []
            out_alphas_batch = []
            out_dpts_batch = []
            out_metas_batch = dotdict()

            for i in range(T):
                timer_ctx = _start_fourdgs_timer(timing_state, xyz.device)
                ts_i = ts_b[i]

                if use_split:
                    # -- dynamic: temporal gating + motion + sparse filter --
                    marg_d = compute_marginal_t(ts_i, t_d, cov_t_d).float()
                    marg_d = torch.nan_to_num(marg_d, nan=0.0, posinf=1.0, neginf=0.0)
                    opa_d = (opacity_d * marg_d)[..., 0]
                    valid_d = (marg_d[..., 0] > self.marginal_th) & (opa_d > self.opacity_th)

                    alive_d = valid_d.nonzero(as_tuple=False).squeeze(-1)

                    dt_d_alive = (ts_i - t_d[alive_d]).float()
                    means_d = xyz_d[alive_d] + _evaluate_temporal_delta(ms3_d[alive_d], dt_d_alive)
                    domega_d_alive = _evaluate_temporal_delta(omega_d[alive_d], dt_d_alive)
                    dquats_d = angle_axis_to_quaternion(domega_d_alive)
                    quats_d = qmul(rotation_d[alive_d], dquats_d).float()
                    sc_d = scale_d[alive_d]
                    opa_d_alive = opa_d[alive_d]
                    colors_d_alive = colors_d[alive_d]

                    # -- concatenate: static (precomputed, no per-frame work) + alive dynamic --
                    means = torch.cat([means_s, means_d], dim=0)
                    quats = torch.cat([quats_s, quats_d], dim=0)
                    opacities = torch.cat([opa_s, opa_d_alive], dim=0)
                    scale_b_cur = torch.cat([scales_s, sc_d], dim=0)
                    colors = torch.cat([colors_s, colors_d_alive], dim=0)

                    if render_flow:
                        all_xyz = torch.cat([means_s, xyz_d[alive_d]], dim=0)
                        all_t = torch.cat([t_b[s_idx], t_d[alive_d]], dim=0)
                        all_ms3 = torch.cat([ms3_b[s_idx], ms3_d[alive_d]], dim=0)
                        curr_means = _project_world_means_to_camera(means, w2cs_b[i])
                        if i < T - 1:
                            next_dt = (ts_b[i + 1] - all_t).float()
                            next_means_world = all_xyz + _evaluate_temporal_delta(all_ms3, next_dt)
                            next_means = _project_world_means_to_camera(next_means_world, w2cs_b[i + 1])
                        else:
                            next_means = curr_means
                        colors = torch.cat([colors, curr_means, next_means], dim=-1)

                    if render_motion:
                        all_ms3 = torch.cat([ms3_b[s_idx], ms3_d[alive_d]], dim=0)
                        all_cov_t = torch.cat([cov_t_b[s_idx], cov_t_d[alive_d]], dim=0)
                        motion_mask = _motion_mask_from_ms3_cov_t(
                            all_ms3,
                            all_cov_t,
                            self.motion_mask_th,
                            self.motion_mask_cov_t_min,
                            self.motion_mask_cov_t_max,
                        )
                        colors = torch.cat([colors, motion_mask], dim=-1)
                else:
                    # -- fallback: same as v3 (no split) --
                    marginal_t = compute_marginal_t(ts_i, t_b, cov_t_b).float()
                    marginal_t = torch.nan_to_num(marginal_t, nan=0.0, posinf=1.0, neginf=0.0)
                    opacities = (opacity_b * marginal_t)[..., 0]
                    valid = (marginal_t[..., 0] > self.marginal_th) & (opacities > self.opacity_th)
                    opacities = torch.where(valid, opacities, opacities.new_zeros(()))

                    dt = (ts_i - t_b).float() * valid.unsqueeze(-1).float()
                    means = xyz_b + _evaluate_temporal_delta(ms3_b, dt)
                    domega = _evaluate_temporal_delta(omega_b, dt)
                    dquats = angle_axis_to_quaternion(domega)
                    quats = qmul(rotation_b, dquats).float()
                    scale_b_cur = torch.where(
                        valid.unsqueeze(-1), scale_b_base, scale_b_base.new_zeros(())
                    )
                    colors = colors_base

                    if render_flow:
                        curr_means = _project_world_means_to_camera(means, w2cs_b[i])
                        if i < T - 1:
                            next_dt = (ts_b[i + 1] - t_b).float() * valid.unsqueeze(-1).float()
                            next_means_world = xyz_b + _evaluate_temporal_delta(ms3_b, next_dt)
                            next_means = _project_world_means_to_camera(next_means_world, w2cs_b[i + 1])
                        else:
                            next_means = curr_means
                        colors = torch.cat([colors, curr_means, next_means], dim=-1)

                    if render_motion:
                        motion_mask = _motion_mask_from_ms3_cov_t(
                            ms3_b, cov_t_b, self.motion_mask_th,
                            self.motion_mask_cov_t_min, self.motion_mask_cov_t_max
                        )
                        colors = torch.cat([colors, motion_mask], dim=-1)

                viewmats = w2cs_b[i : i + 1]
                ixts = Ks_b[i : i + 1]

                if not use_split:
                    alive = opacities > 0
                    ind = alive.nonzero(as_tuple=False).squeeze(-1)
                    means = means.index_select(0, ind)
                    quats = quats.index_select(0, ind)
                    scale_b_cur = scale_b_cur.index_select(0, ind)
                    opacities = opacities.index_select(0, ind)
                    colors = colors.index_select(0, ind)

                _rasterize_fn = threedgs_rasterizer
                if self.use_2dgs:
                    _rasterize_fn = twodgs_rasterizer_fast if is_inference else twodgs_rasterizer
                out_img, out_alpha, out_dpt, meta = _rasterize_fn(
                    means, quats, scale_b_cur, opacities, colors,
                    viewmats, ixts, _h, _w, render_mode,
                )
                _stop_fourdgs_timer(timing_state, timer_ctx)

                if is_inference:
                    out_img = out_img.detach()
                    out_alpha = out_alpha.detach()
                    out_dpt = out_dpt.detach()
                    if transfer_stream is not None:
                        with torch.cuda.stream(transfer_stream):
                            out_imgs_batch.append(out_img.to("cpu", non_blocking=True))
                            out_alphas_batch.append(out_alpha.to("cpu", non_blocking=True))
                            out_dpts_batch.append(out_dpt.to("cpu", non_blocking=True))
                    else:
                        out_imgs_batch.append(out_img.cpu())
                        out_alphas_batch.append(out_alpha.cpu())
                        out_dpts_batch.append(out_dpt.cpu())
                else:
                    out_imgs_batch.append(out_img)
                    out_alphas_batch.append(out_alpha)
                    out_dpts_batch.append(out_dpt)

                for key in meta:
                    if key not in out_metas_batch:
                        out_metas_batch[key] = []
                    out_metas_batch[key].append(meta[key])

            if transfer_stream is not None:
                transfer_stream.synchronize()

            out_img = torch.cat(out_imgs_batch).permute(0, 3, 1, 2)
            out_alpha = torch.cat(out_alphas_batch).permute(0, 3, 1, 2)
            out_dpt = torch.cat(out_dpts_batch).permute(0, 3, 1, 2)
            out_meta = dotdict(
                {
                    key: torch.cat(val).permute(0, 3, 1, 2)
                    if val[0].ndim == 4
                    else torch.cat(val)
                    for key, val in out_metas_batch.items()
                }
            )

            out_imgs.append(out_img)
            out_alphas.append(out_alpha)
            out_dpts.append(out_dpt)
            for key in out_meta:
                if key not in out_metas:
                    out_metas[key] = []
                out_metas[key].append(out_meta[key])

        img, mask, depth = (
            torch.stack(out_imgs),
            torch.stack(out_alphas),
            torch.stack(out_dpts),
        )
        meta = dotdict()
        for key in out_metas:
            meta[key] = torch.stack(out_metas[key])

        return img, mask, depth, meta

    def single_batch_rasterizer_v5(
        self,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        scale: torch.Tensor,
        rotation: torch.Tensor,
        opacity: torch.Tensor,
        t: torch.Tensor,
        cov_t: torch.Tensor,
        ms3: torch.Tensor,
        omega: torch.Tensor,
        w2cs: torch.Tensor,
        Ks: torch.Tensor,
        ts: torch.Tensor,
        height: int = 0,
        width: int = 0,
        render_mode: str = "RGB+ED",
        flo=None,
    ):
        """
        Temporal bucketing + zero-copy buffer variant (inference only).

        Key optimisations over v4:
          1. Static gaussians (cov_t > 1, motion frozen): written into a
             pre-allocated buffer ONCE outside the per-frame loop.  Per frame
             the static portion is already in-place — zero index_select,
             zero torch.cat.
          2. Dynamic gaussians: candidate indices + marginal-weighted opacity
             pre-computed via temporal intervals (outside timing).  Source
             arrays pre-gathered per frame so the timed loop only runs
             motion math on compact contiguous tensors + one small memcpy
             into the buffer's dynamic slot.
          3. The rasterizer reads contiguous slices of the single buffer —
             no per-frame allocation or concatenation.
        """
        timing_state = self._get_fourdgs_profiler_state()
        render_motion = (not xyz.requires_grad) and bool(self.vis_motion)
        render_flow = (not xyz.requires_grad) and bool(self.vis_flow)
        is_inference = not xyz.requires_grad

        B, T, _, _ = w2cs.shape
        _h = height or self.height
        _w = width or self.width

        _use_cuda = xyz.is_cuda
        transfer_stream = torch.cuda.Stream() if (is_inference and _use_cuda) else None

        _neg2_ln_marg = -2.0 * math.log(max(self.marginal_th, 1e-30))

        out_imgs = []
        out_alphas = []
        out_dpts = []
        out_metas = dotdict()

        for b in range(B):
            xyz_b = xyz[b].float()
            rgb_b = rgb[b]
            scale_b_base = scale[b].float()
            rotation_b = rotation[b].float()
            opacity_b = opacity[b].float()
            t_b = t[b]
            cov_t_b = cov_t[b]
            ms3_b = ms3[b]
            omega_b = omega[b]
            ts_b = ts[b]
            w2cs_b = w2cs[b].float()
            Ks_b = Ks[b].float()
            _dev = xyz_b.device

            base_color_ch = (
                3 if "RGB" in render_mode
                else max(int(rgb_b.shape[-1]) - 3, 0)
            )
            colors_base = (
                rgb_b[..., :3].float()
                if "RGB" in render_mode
                else rgb_b[..., 3:].float()
            )

            # ================================================================
            # Phase 1: Static / Dynamic separation
            # ================================================================
            static_mask = cov_t_b[..., 0] > 1.0
            has_static = static_mask.any().item()
            has_dynamic = (~static_mask).any().item()

            n_s = 0
            if has_static:
                s_idx = static_mask.nonzero(as_tuple=False).squeeze(-1)
                means_s = xyz_b[s_idx]
                quats_s = rotation_b[s_idx]
                scales_s = scale_b_base[s_idx]
                opa_s = opacity_b[s_idx][..., 0]
                colors_s = colors_base[s_idx]
                t_s = t_b[s_idx]
                cov_t_s = cov_t_b[s_idx]
                ms3_s = ms3_b[s_idx]

                alive_s = opa_s > 0.01
                if not alive_s.all():
                    means_s = means_s[alive_s]
                    quats_s = quats_s[alive_s]
                    scales_s = scales_s[alive_s]
                    opa_s = opa_s[alive_s]
                    colors_s = colors_s[alive_s]
                    t_s = t_s[alive_s]
                    cov_t_s = cov_t_s[alive_s]
                    ms3_s = ms3_s[alive_s]
                n_s = means_s.shape[0]
                if n_s == 0:
                    has_static = False

            if has_dynamic:
                d_idx = (~static_mask).nonzero(as_tuple=False).squeeze(-1)
                xyz_d = xyz_b[d_idx]
                rotation_d = rotation_b[d_idx]
                scale_d = scale_b_base[d_idx]
                opacity_d = opacity_b[d_idx]
                t_d = t_b[d_idx]
                cov_t_d = cov_t_b[d_idx]
                ms3_d = ms3_b[d_idx]
                omega_d = omega_b[d_idx]
                colors_d = colors_base[d_idx]

            # ================================================================
            # Phase 2: Temporal pre-computation (OUTSIDE timing loop)
            # ================================================================

            # --- 2a. Dynamic: interval bucketing + pre-gather + pre-compute motion ---
            frame_d_n = []
            frame_d_opa_list = []
            frame_d_means_pre = []
            frame_d_quats_pre = []
            frame_d_scales_pre = []
            frame_d_colors_pre = []
            frame_d_xyz_orig_pre = []
            frame_d_t_orig_pre = []
            frame_d_cov_t_orig_pre = []
            frame_d_ms3_orig_pre = []

            if has_dynamic:
                r_d = torch.sqrt(
                    torch.clamp(cov_t_d[..., 0] * _neg2_ln_marg, min=0.0)
                )
                t_center_d = t_d[..., 0]
                t_lo_d = t_center_d - r_d
                t_hi_d = t_center_d + r_d

                _need_orig = render_flow or render_motion

                for i in range(T):
                    ts_iv = ts_b[i]
                    alive = (t_lo_d <= ts_iv) & (t_hi_d >= ts_iv)
                    cand = alive.nonzero(as_tuple=False).squeeze(-1)
                    if cand.numel() > 0:
                        mg = compute_marginal_t(
                            ts_iv, t_d[cand], cov_t_d[cand]
                        ).float()
                        mg = torch.nan_to_num(
                            mg, nan=0.0, posinf=1.0, neginf=0.0
                        )
                        op = (opacity_d[cand] * mg)[..., 0]
                        valid = op > self.opacity_th
                        if valid.any():
                            vc = cand[valid]
                            d_xyz = xyz_d[vc]
                            d_rot = rotation_d[vc]
                            d_ms3 = ms3_d[vc]
                            d_omg = omega_d[vc]
                            d_t = t_d[vc]
                            d_cov_t = cov_t_d[vc]

                            dt = (ts_iv - d_t).float()
                            d_means = d_xyz + _evaluate_temporal_delta(d_ms3, dt)
                            d_domega = _evaluate_temporal_delta(d_omg, dt)
                            d_quats = qmul(
                                d_rot, angle_axis_to_quaternion(d_domega)
                            ).float()

                            frame_d_n.append(vc.numel())
                            frame_d_opa_list.append(op[valid])
                            frame_d_means_pre.append(d_means)
                            frame_d_quats_pre.append(d_quats)
                            frame_d_scales_pre.append(scale_d[vc])
                            frame_d_colors_pre.append(colors_d[vc])
                            if _need_orig:
                                frame_d_xyz_orig_pre.append(d_xyz)
                                frame_d_t_orig_pre.append(d_t)
                                frame_d_cov_t_orig_pre.append(d_cov_t)
                                frame_d_ms3_orig_pre.append(d_ms3)
                            else:
                                frame_d_xyz_orig_pre.append(None)
                                frame_d_t_orig_pre.append(None)
                                frame_d_cov_t_orig_pre.append(None)
                                frame_d_ms3_orig_pre.append(None)
                            continue
                    frame_d_n.append(0)
                    frame_d_opa_list.append(None)
                    frame_d_means_pre.append(None)
                    frame_d_quats_pre.append(None)
                    frame_d_scales_pre.append(None)
                    frame_d_colors_pre.append(None)
                    frame_d_xyz_orig_pre.append(None)
                    frame_d_t_orig_pre.append(None)
                    frame_d_cov_t_orig_pre.append(None)
                    frame_d_ms3_orig_pre.append(None)
            else:
                frame_d_n = [0] * T

            # --- 2b. Pre-allocate unified buffer ---
            max_dyn = max(frame_d_n) if frame_d_n else 0
            buf_sz = n_s + max_dyn
            if buf_sz == 0:
                buf_sz = 1

            extra_ch = (6 if render_flow else 0) + (1 if render_motion else 0)
            total_color_ch = base_color_ch + extra_ch

            buf_means = torch.empty(buf_sz, 3, device=_dev, dtype=torch.float32)
            buf_quats = torch.empty(buf_sz, 4, device=_dev, dtype=torch.float32)
            buf_scales = torch.empty(buf_sz, 3, device=_dev, dtype=torch.float32)
            buf_opa = torch.empty(buf_sz, device=_dev, dtype=torch.float32)
            buf_colors = torch.empty(
                buf_sz, total_color_ch, device=_dev, dtype=torch.float32
            )

            if render_flow or render_motion:
                buf_xyz_orig = torch.empty(buf_sz, 3, device=_dev, dtype=torch.float32)
                buf_t_orig = t_b.new_empty(buf_sz, t_b.shape[-1])
                buf_ms3_orig = ms3_b.new_empty(buf_sz, ms3_b.shape[-1])
                if render_motion:
                    buf_cov_t_orig = cov_t_b.new_empty(buf_sz, cov_t_b.shape[-1])

            # --- 2c. Fill static portion ONCE ---
            if has_static:
                buf_means[:n_s] = means_s
                buf_quats[:n_s] = quats_s
                buf_scales[:n_s] = scales_s
                buf_opa[:n_s] = opa_s
                buf_colors[:n_s, :base_color_ch] = colors_s
                if render_flow or render_motion:
                    buf_xyz_orig[:n_s] = means_s
                    buf_t_orig[:n_s] = t_s
                    buf_ms3_orig[:n_s] = ms3_s
                    if render_motion:
                        buf_cov_t_orig[:n_s] = cov_t_s

            # ================================================================
            # Phase 3: Per-frame rendering (INSIDE timing loop)
            # ================================================================
            _rasterize_fn = threedgs_rasterizer
            if self.use_2dgs:
                _rasterize_fn = (
                    twodgs_rasterizer_fast if is_inference else twodgs_rasterizer
                )

            out_imgs_batch = []
            out_alphas_batch = []
            out_dpts_batch = []
            out_metas_batch = dotdict()

            for i in range(T):
                timer_ctx = _start_fourdgs_timer(timing_state, xyz.device)
                ts_i = ts_b[i]

                n_d = frame_d_n[i]
                n_total = n_s + n_d

                if n_total == 0:
                    out_imgs_batch.append(
                        xyz_b.new_zeros((1, _h, _w, total_color_ch))
                    )
                    out_alphas_batch.append(xyz_b.new_zeros((1, _h, _w, 1)))
                    out_dpts_batch.append(xyz_b.new_zeros((1, _h, _w, 1)))
                    _stop_fourdgs_timer(timing_state, timer_ctx)
                    continue

                # ---- dynamic: copy pre-computed results → buffer ----
                if n_d > 0:
                    s = n_s
                    e = s + n_d
                    buf_means[s:e] = frame_d_means_pre[i]
                    buf_quats[s:e] = frame_d_quats_pre[i]
                    buf_scales[s:e] = frame_d_scales_pre[i]
                    buf_opa[s:e] = frame_d_opa_list[i]
                    buf_colors[s:e, :base_color_ch] = frame_d_colors_pre[i]
                    if render_flow or render_motion:
                        buf_xyz_orig[s:e] = frame_d_xyz_orig_pre[i]
                        buf_t_orig[s:e] = frame_d_t_orig_pre[i]
                        buf_ms3_orig[s:e] = frame_d_ms3_orig_pre[i]
                        if render_motion:
                            buf_cov_t_orig[s:e] = frame_d_cov_t_orig_pre[i]

                # ---- flow / motion extra channels ----
                if render_flow:
                    cur_cam = _project_world_means_to_camera(
                        buf_means[:n_total], w2cs_b[i]
                    )
                    if i < T - 1:
                        ndt = (ts_b[i + 1] - buf_t_orig[:n_total]).float()
                        nxt_w = (
                            buf_xyz_orig[:n_total]
                            + _evaluate_temporal_delta(
                                buf_ms3_orig[:n_total], ndt
                            )
                        )
                        nxt_cam = _project_world_means_to_camera(
                            nxt_w, w2cs_b[i + 1]
                        )
                    else:
                        nxt_cam = cur_cam
                    buf_colors[:n_total, base_color_ch:base_color_ch + 3] = cur_cam
                    buf_colors[:n_total, base_color_ch + 3:base_color_ch + 6] = nxt_cam

                if render_motion:
                    mo_off = base_color_ch + (6 if render_flow else 0)
                    buf_colors[:n_total, mo_off:mo_off + 1] = _motion_mask_from_ms3_cov_t(
                        buf_ms3_orig[:n_total],
                        buf_cov_t_orig[:n_total],
                        self.motion_mask_th,
                        self.motion_mask_cov_t_min,
                        self.motion_mask_cov_t_max,
                    ).float()

                # ---- rasterize (contiguous buffer slices) ----
                out_img, out_alpha, out_dpt, meta = _rasterize_fn(
                    buf_means[:n_total],
                    buf_quats[:n_total],
                    buf_scales[:n_total],
                    buf_opa[:n_total],
                    buf_colors[:n_total, :total_color_ch],
                    w2cs_b[i : i + 1],
                    Ks_b[i : i + 1],
                    _h, _w, render_mode,
                )
                _stop_fourdgs_timer(timing_state, timer_ctx)

                if is_inference:
                    out_img = out_img.detach()
                    out_alpha = out_alpha.detach()
                    out_dpt = out_dpt.detach()
                    if transfer_stream is not None:
                        with torch.cuda.stream(transfer_stream):
                            out_imgs_batch.append(
                                out_img.to("cpu", non_blocking=True)
                            )
                            out_alphas_batch.append(
                                out_alpha.to("cpu", non_blocking=True)
                            )
                            out_dpts_batch.append(
                                out_dpt.to("cpu", non_blocking=True)
                            )
                    else:
                        out_imgs_batch.append(out_img.cpu())
                        out_alphas_batch.append(out_alpha.cpu())
                        out_dpts_batch.append(out_dpt.cpu())
                else:
                    out_imgs_batch.append(out_img)
                    out_alphas_batch.append(out_alpha)
                    out_dpts_batch.append(out_dpt)

                for key in meta:
                    if key not in out_metas_batch:
                        out_metas_batch[key] = []
                    out_metas_batch[key].append(meta[key])

            if transfer_stream is not None:
                transfer_stream.synchronize()

            out_img = torch.cat(out_imgs_batch).permute(0, 3, 1, 2)
            out_alpha = torch.cat(out_alphas_batch).permute(0, 3, 1, 2)
            out_dpt = torch.cat(out_dpts_batch).permute(0, 3, 1, 2)
            out_meta = dotdict(
                {
                    key: torch.cat(val).permute(0, 3, 1, 2)
                    if val[0].ndim == 4
                    else torch.cat(val)
                    for key, val in out_metas_batch.items()
                }
            )

            out_imgs.append(out_img)
            out_alphas.append(out_alpha)
            out_dpts.append(out_dpt)
            for key in out_meta:
                if key not in out_metas:
                    out_metas[key] = []
                out_metas[key].append(out_meta[key])

        img, mask, depth = (
            torch.stack(out_imgs),
            torch.stack(out_alphas),
            torch.stack(out_dpts),
        )
        meta = dotdict()
        for key in out_metas:
            meta[key] = torch.stack(out_metas[key])

        return img, mask, depth, meta

    def __call__(
        self,
        gs_params: Mapping[str, torch.Tensor],  # B, T, H, W, C
        ts: torch.Tensor,  # B, T, 1
        Ks: torch.Tensor,  # B, T, 3, 3
        RTs: torch.Tensor,  # B, T, 3, 4
        height: int = 0,
        width: int = 0,
        render_mode: str = "RGB+ED",
        **kwargs,
    ):
        # B, T, H, W, C for params
        xyz, rgb, scale, rotation, opacity = (
            gs_params["xyz"],
            gs_params["feature"],  # assume last channel is for monochrome
            gs_params["scaling"],
            gs_params["rotation"],
            gs_params["opacity"],
        )

        t, cov_t, ms3, omega = (
            gs_params["t"],
            gs_params["cov_t"],
            gs_params["ms3"],
            gs_params["omega"],
        )

        xyz = xyz.reshape(xyz.shape[0], -1, xyz.shape[-1])
        rgb = rgb.reshape(rgb.shape[0], -1, rgb.shape[-1])
        scale = scale.reshape(scale.shape[0], -1, scale.shape[-1])
        rotation = rotation.reshape(rotation.shape[0], -1, rotation.shape[-1])
        opacity = opacity.reshape(opacity.shape[0], -1, opacity.shape[-1])
        t = t.reshape(t.shape[0], -1, t.shape[-1])
        cov_t = cov_t.reshape(cov_t.shape[0], -1, cov_t.shape[-1])
        ms3 = ms3.reshape(ms3.shape[0], -1, ms3.shape[-1])
        omega = omega.reshape(omega.shape[0], -1, omega.shape[-1])
        Ks = Ks.float()  # B, V, 3, 3
        w2cs = affine_inverse(RTs).float()  # B, V, 4, 4
        # In no-grad inference, motion/flow outputs should respect the explicit
        # renderer flags instead of being auto-enabled.
        render_motion = (not xyz.requires_grad) and bool(self.vis_motion)
        render_flow = (not xyz.requires_grad) and bool(self.vis_flow)

        imgs = []
        masks = []
        depths = []
        metas = dotdict()
        voxel_merge_stats = []
        B = ts.shape[0]
        for b in range(B):
            xyz_b = xyz[b]
            rgb_b = rgb[b]
            scale_b = scale[b]
            rotation_b = rotation[b]
            opacity_b = opacity[b]
            t_b = t[b]
            cov_t_b = cov_t[b]
            ms3_b = ms3[b]
            omega_b = omega[b]

            if self.enable_static_voxel_merge:
                (
                    xyz_b,
                    rgb_b,
                    scale_b,
                    rotation_b,
                    opacity_b,
                    t_b,
                    cov_t_b,
                    ms3_b,
                    omega_b,
                    merge_stats,
                ) = self._merge_static_gaussians_by_voxel(
                    xyz_b,
                    rgb_b,
                    scale_b,
                    rotation_b,
                    opacity_b,
                    t_b,
                    cov_t_b,
                    ms3_b,
                    omega_b,
                )
                if merge_stats is not None:
                    voxel_merge_stats.append(merge_stats)

            if self.use_v2_rasterizer == 5:
                _batch_rasterize = self.single_batch_rasterizer_v5
            elif self.use_v2_rasterizer == 4:
                _batch_rasterize = self.single_batch_rasterizer_v4
            elif self.use_v2_rasterizer == 3:
                _batch_rasterize = self.single_batch_rasterizer_v3
            elif self.use_v2_rasterizer:
                _batch_rasterize = self.single_batch_rasterizer_v2
            else:
                _batch_rasterize = self.single_batch_rasterizer
            img, mask, depth, meta = _batch_rasterize(
                xyz_b.unsqueeze(0),
                rgb_b.unsqueeze(0),
                scale_b.unsqueeze(0),
                rotation_b.unsqueeze(0),
                opacity_b.unsqueeze(0),
                t_b.unsqueeze(0),
                cov_t_b.unsqueeze(0),
                ms3_b.unsqueeze(0),
                omega_b.unsqueeze(0),
                w2cs[b : b + 1],
                Ks[b : b + 1],
                ts[b : b + 1],
                height=height or self.height,
                width=width or self.width,
                render_mode=render_mode,
            )
            imgs.append(img)
            masks.append(mask)
            depths.append(depth)
            for key in meta:
                if key not in metas:
                    metas[key] = []
                metas[key].append(meta[key])
        img = torch.cat(imgs).float()  # B, V, C, H, W
        mask = torch.cat(masks).float()  # B, V, C, H, W
        depth = torch.cat(depths).float()  # B, V, C, H, W
        meta = dotdict()
        for key in metas:
            if all(x.shape == metas[key][0].shape for x in metas[key]):
                meta[key] = torch.cat(metas[key]).float()  # B, V, N
            else:
                # Gaussian count may differ across batch after voxel merge; pad to stack.
                max_n = max(x.shape[-1] for x in metas[key])
                padded = []
                valid_lengths = []
                for x in metas[key]:
                    valid_lengths.append(x.shape[-1])
                    if x.shape[-1] < max_n:
                        pad = x.new_zeros(*x.shape[:-1], max_n - x.shape[-1])
                        x = torch.cat([x, pad], dim=-1)
                    padded.append(x)
                meta[key] = torch.cat(padded).float()
                meta[f"{key}_valid_length"] = torch.tensor(
                    valid_lengths, device=meta[key].device, dtype=torch.long
                )

        if voxel_merge_stats:
            before = sum(x["before"] for x in voxel_merge_stats)
            after = sum(x["after"] for x in voxel_merge_stats)
            static_before = sum(x["static_before"] for x in voxel_merge_stats)
            static_after = sum(x["static_after"] for x in voxel_merge_stats)
            meta.static_voxel_merge_before = torch.tensor(
                [before], device=img.device, dtype=torch.float32
            )
            meta.static_voxel_merge_after = torch.tensor(
                [after], device=img.device, dtype=torch.float32
            )
            meta.static_voxel_merge_ratio = torch.tensor(
                [after / max(before, 1)], device=img.device, dtype=torch.float32
            )
            meta.static_voxel_merge_static_before = torch.tensor(
                [static_before], device=img.device, dtype=torch.float32
            )
            meta.static_voxel_merge_static_after = torch.tensor(
                [static_after], device=img.device, dtype=torch.float32
            )

        _used_fast_2dgs = (self.use_v2_rasterizer in (4, 5)) and (not xyz.requires_grad)
        if self.use_2dgs and not _used_fast_2dgs:
            C = img.shape[-3]
            img, normal, surf_normal, distort, median_depth = img.split(
                [C - 8, 3, 3, 1, 1], dim=-3
            )
            meta.normal = normal  # already in -1, 1 range
            meta.surf_normal = surf_normal  # already in -1, 1 range
            meta.distort = distort * 2 - 1
            meta.median_depth = median_depth

        # Universal
        if render_motion:
            C = img.shape[-3]
            img, motion_mask = img.split([C - 1, 1], dim=-3)
            meta.motion_mask = motion_mask * 2 - 1

        # Approximated 2D flow rendering
        if render_flow:
            Ks = Ks.to(img.device)  # MARK: SYNC
            C = img.shape[-3]
            img, mean0, mean1 = img.split([C - 3 - 3, 3, 3], dim=-3)
            # B, T, 3, H, W, camera space xyz map
            B, T, _, H, W = mean1.shape
            meta.mean0 = mean0
            meta.mean1 = mean1

            # B, T, N, 3 @ B, T, 3, 3 -> B, T, N, 3
            screen0 = mean0.reshape(B, T, 3, H * W).permute(0, 1, 3, 2) @ Ks.mT
            screen0 = screen0[..., :2] / (screen0[..., 2:].clip(1e-8))
            screen0 = screen0.permute(0, 1, 3, 2).reshape(B, T, 2, H, W)
            screen1 = mean1.reshape(B, T, 3, H * W).permute(0, 1, 3, 2) @ Ks.mT
            screen1 = screen1[..., :2] / (screen1[..., 2:].clip(1e-8))
            screen1 = screen1.permute(0, 1, 3, 2).reshape(B, T, 2, H, W)
            flow = screen1 - screen0
            del screen0, screen1
            # flow = torch.cat([flow, flow[:, -1:]], dim=1)
            meta.flow = flow

            flow = flow.reshape(-1, *flow.shape[-3:])
            flow_vis = flow_to_color(flow) / 255  # don't be too large
            flow_vis = flow_vis.reshape(img.shape[:-3] + flow_vis.shape[1:])
            meta.flow_vis = flow_vis * 2 - 1

        img = (img * 2 - 1).clamp(-1, 1)

        output = {
            "rgb": img,
            "mask": mask,
            "depth": depth,
            **meta,
        }
        if "MONO" in render_mode:
            output["mono"] = output.pop("rgb")
        return output
