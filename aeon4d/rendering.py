"""4DGS rendering: dummy views for the decoder, moving-anchor spiral novel-view cameras, rasterization."""

import logging
import numpy as np
import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def _apply_motion_mask_output_gates(
    render_out: dict,
    flow_residual_th=None,
    alpha_th=None,
    flow_visible_alpha_th: float = 0.05,
) -> None:
    """Optional pixel-space gates for rendered motion_mask.

    The base motion_mask is rendered from per-Gaussian attributes. These gates
    intersect it with visual evidence in the rendered frame: residual screen
    flow and final alpha coverage.
    """
    if "motion_mask" not in render_out:
        return

    motion_mask = render_out["motion_mask"].float()
    before = float((motion_mask > 0.5).float().mean().item() * 100.0)
    applied = []

    if flow_residual_th is not None:
        if "flow" not in render_out:
            logger.warning("motion_mask flow-residual gate requested, but render_out has no 'flow'; skipping.")
        else:
            flow_raw = render_out["flow"].float()
            valid = torch.isfinite(flow_raw).all(dim=2, keepdim=True)
            flow = torch.nan_to_num(flow_raw, nan=0.0, posinf=0.0, neginf=0.0)

            visible = valid
            if "mask" in render_out and render_out["mask"] is not None:
                visible = visible & (render_out["mask"].float() > float(flow_visible_alpha_th))

            residual = flow.clone()
            B, T = residual.shape[:2]
            for b in range(B):
                for t in range(T):
                    pix = visible[b, t, 0]
                    if int(pix.sum().item()) < 32:
                        continue
                    median_flow = residual[b, t, :, pix].median(dim=1).values.view(2, 1, 1)
                    residual[b, t] = residual[b, t] - median_flow

            mag = residual.norm(dim=2, keepdim=True)
            flow_gate = (mag > float(flow_residual_th)) & visible
            # The renderer's forward flow is zero on the last frame by construction.
            # Do not erase the last motion mask just because there is no t+1 frame.
            if flow_gate.shape[1] > 1:
                flow_gate[:, -1:] = visible[:, -1:]
            motion_mask = motion_mask * flow_gate.float()
            render_out["flow_residual_magnitude"] = mag
            applied.append(f"flow_residual>{float(flow_residual_th):g}px")

    if alpha_th is not None:
        if "mask" not in render_out or render_out["mask"] is None:
            logger.warning("motion_mask alpha gate requested, but render_out has no alpha 'mask'; skipping.")
        else:
            alpha_gate = render_out["mask"].float() > float(alpha_th)
            motion_mask = motion_mask * alpha_gate.float()
            applied.append(f"alpha>{float(alpha_th):g}")

    render_out["motion_mask"] = motion_mask.clamp(0.0, 1.0)
    if applied:
        after = float((render_out["motion_mask"] > 0.5).float().mean().item() * 100.0)
        logger.info(
            "motion_mask output gates applied: %s; active %.2f%% -> %.2f%%",
            ", ".join(applied),
            before,
            after,
        )


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])


IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def build_dummy_views(feedforward_image, num_frames, device):
    """
    Build dummy view dicts from decoded pixel video for forward pass.

    The model's forward needs views[t]["img"] for shape info and geometric
    fusion. We construct minimal view dicts from the decoded pixels.

    Args:
        feedforward_image: (B, 3, T, H, W) in [-1, 1]
        num_frames: T
        device: torch device
    Returns:
        views: list of T view dicts
    """
    B, C, T, H, W = feedforward_image.shape
    mean = IMAGENET_MEAN.to(device).view(1, 3, 1, 1)
    std = IMAGENET_STD.to(device).view(1, 3, 1, 1)

    views = []
    for t in range(num_frames):
        frame_01 = (feedforward_image[:, :, t] + 1.0) / 2.0  # [-1,1] → [0,1]
        frame_01 = frame_01.clamp(0, 1)
        frame_norm = (frame_01 - mean) / std  # ImageNet normalize

        view = {
            "img": frame_norm,  # (B, 3, H, W) ImageNet normalized
            "data_norm_type": ["dinov2"] * B,
            "true_shape": torch.tensor([[H, W]], device=device).repeat(B, 1),
        }
        views.append(view)

    return views


def _to_tensor(x, device):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, device=device)


def build_moving_anchor_spiral_c2ws(
    res,
    n_rots,
    zrate,
    radii,
    rotation_mode,
    look_forward_focal,
    device,
):
    """Generate one novel pose per original frame, anchored to that frame's pose.

    ``rotation_mode='look_forward'`` is scheme A: each perturbed camera center
    re-orients toward the current frame's forward focal point.
    ``rotation_mode='keep'`` is scheme B: only translate around the original
    camera path and keep each frame's original predicted rotation.
    """
    from aeon4d._vendor.mapanything.utils.geometry import quaternion_to_rotation_matrix

    cam_trans = torch.stack([_to_tensor(r["cam_trans"], device).float() for r in res], dim=1)
    cam_quats = torch.stack([_to_tensor(r["cam_quats"], device).float() for r in res], dim=1)
    B, T = cam_trans.shape[:2]
    rot_mat = quaternion_to_rotation_matrix(cam_quats.reshape(-1, 4)).reshape(B, T, 3, 3)

    rx, ry, rz = [float(x) for x in radii]
    theta = torch.arange(T, device=device, dtype=torch.float32) / max(T, 1)
    theta = theta * (2.0 * np.pi * float(n_rots))
    local_offsets = torch.stack(
        [
            torch.cos(theta) * rx,
            torch.sin(theta) * ry,
            torch.sin(theta * float(zrate)) * rz,
        ],
        dim=-1,
    )
    local_offsets = local_offsets.view(1, T, 3, 1).expand(B, -1, -1, -1)
    centers = cam_trans + torch.matmul(rot_mat, local_offsets).squeeze(-1)

    if rotation_mode == "keep":
        novel_rot = rot_mat
    elif rotation_mode == "look_forward":
        focal_local = torch.zeros(B, T, 3, 1, device=device, dtype=torch.float32)
        focal_local[:, :, 2, 0] = float(look_forward_focal)
        lookat = cam_trans + torch.matmul(rot_mat, focal_local).squeeze(-1)
        v_front = F.normalize(lookat - centers, dim=-1, eps=1e-6)
        v_up = F.normalize(-rot_mat[:, :, :, 1], dim=-1, eps=1e-6)
        v_right = F.normalize(torch.cross(v_front, v_up, dim=-1), dim=-1, eps=1e-6)
        v_down = torch.cross(v_front, v_right, dim=-1)
        novel_rot = torch.stack([v_right, v_down, v_front], dim=-1)
    else:
        raise ValueError(f"Unknown moving-anchor rotation_mode: {rotation_mode}")

    pose_mat = torch.eye(4, device=device, dtype=torch.float32).view(1, 1, 4, 4).repeat(B, T, 1, 1)
    pose_mat[:, :, :3, :3] = novel_rot
    pose_mat[:, :, :3, 3] = centers
    return pose_mat.contiguous()


def render_4dgs(gs_params, res, device, render_h=518, render_w=518, fix_k_mode="median",
                bg_color=(0.0, 0.0, 0.0), override_c2ws=None, focal_zoom=1.0,
                motion_mask=False, motion_mask_th=0.25,
                motion_mask_cov_t_min=None, motion_mask_cov_t_max=None,
                motion_mask_use_ms3=True, motion_mask_flow_residual_th=None,
                motion_mask_alpha_th=None):
    """Render 4D Gaussians into RGB + depth videos.

    fix_k_mode controls how per-frame recovered K is aggregated along T to
    stabilize rendering (pose_head rays are unsupervised at training time, so
    per-frame K fluctuates 1-10% →画面呼吸):
        "none"   — 保持每帧独立 K（原行为）
        "median" — 沿 T 取中位数（默认，对离群帧 robust）
        "first"  — 用第 0 帧 K
        "mean"   — 沿 T 取均值

    bg_color: length-3 sequence RGB ∈ [0,1] for alpha compositing the rendered
        image onto a solid background.

        **Default (0,0,0) = black**, which matches training: the model is
        trained on black-background renders (no background compositing or
        randomization).

        Passing (1,1,1) or decoded-video per-frame bg makes the output
        prettier on OOD prompts (Wan T2V distributions where opacity is
        under-fit) but NO LONGER matches what the model was trained to
        output; use at visual-only risk.

        None disables compositing entirely (identical to black here, since
        gsplat's implicit bg is also 0 — kept for API parity).
    """
    from aeon4d._vendor.mapanything.renders.tlod_renderer_inference import TLODRenderer
    from aeon4d._vendor.mapanything.utils.geometry import quaternion_to_rotation_matrix

    need_motion_flow = bool(motion_mask) and motion_mask_flow_residual_th is not None
    renderer = TLODRenderer(
        vis_motion=bool(motion_mask) and bool(motion_mask_use_ms3),
        vis_flow=need_motion_flow,
        motion_mask_th=float(motion_mask_th),
        motion_mask_cov_t_min=motion_mask_cov_t_min,
        motion_mask_cov_t_max=motion_mask_cov_t_max,
    )

    B = res[0]["cam_trans"].shape[0]
    T = len(res)

    # Build camera-to-world extrinsics from predicted per-frame poses.
    cam_trans = torch.stack([_to_tensor(r["cam_trans"], device).float() for r in res], dim=1)
    cam_quats = torch.stack([_to_tensor(r["cam_quats"], device).float() for r in res], dim=1)

    rot_mat = quaternion_to_rotation_matrix(cam_quats.reshape(-1, 4)).reshape(B, T, 3, 3)
    pose_mat = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)
    pose_mat[:, :, :3, :3] = rot_mat
    pose_mat[:, :, :3, 3] = cam_trans

    # Intrinsics (built from the original T predicted poses; later broadcast/
    # resampled to the rendering view count M when override_c2ws is given).
    if "recovered_K" in gs_params:
        Ks = _to_tensor(gs_params["recovered_K"], device).float()  # [B, T, 3, 3]
        logger.info(
            f"[render_4dgs] recovered_K BEFORE scale: shape={tuple(Ks.shape)} "
            f"fx0={Ks[0,0,0,0].item():.2f} fy0={Ks[0,0,1,1].item():.2f} "
            f"cx0={Ks[0,0,0,2].item():.2f} cy0={Ks[0,0,1,2].item():.2f}"
        )
        # recovered_K 是从 ray_directions (B, H_ref, W_ref, 3) 反推的, 像素坐标系
        # 是 H_ref × W_ref = ff_resolution × ff_resolution. 当渲染画布 (render_h ×
        # render_w) 与之不同时, 必须按比例缩放 K, 否则投影像素只落在 [0, H_ref] /
        # [0, W_ref] 子区间, 画布其余位置 (右半/下半) 整片黑 + GS 视锥外伪影.
        rd = res[0].get("ray_directions")
        if rd is not None and rd.dim() >= 3:
            ref_h = int(rd.shape[-3])
            ref_w = int(rd.shape[-2])
            logger.info(
                f"[render_4dgs] ref_from_ray_directions={ref_h}x{ref_w}, "
                f"render_target={render_h}x{render_w}"
            )
            if ref_h != render_h or ref_w != render_w:
                scale_w = float(render_w) / float(ref_w)
                scale_h = float(render_h) / float(ref_h)
                Ks = Ks.clone()
                Ks[..., 0, 0] *= scale_w   # fx
                Ks[..., 1, 1] *= scale_h   # fy
                Ks[..., 0, 2] *= scale_w   # cx
                Ks[..., 1, 2] *= scale_h   # cy
                logger.info(
                    f"[render_4dgs] K scaled by (sw={scale_w:.3f}, sh={scale_h:.3f}); "
                    f"AFTER scale: fx0={Ks[0,0,0,0].item():.2f} fy0={Ks[0,0,1,1].item():.2f} "
                    f"cx0={Ks[0,0,0,2].item():.2f} cy0={Ks[0,0,1,2].item():.2f}"
                )
        else:
            logger.warning(
                f"[render_4dgs] could not infer K reference resolution: "
                f"rd is None? {rd is None}; using K as-is — render may have black borders!"
            )
        if fix_k_mode == "median":
            Ks = Ks.median(dim=1, keepdim=True).values.expand_as(Ks).contiguous()
        elif fix_k_mode == "first":
            Ks = Ks[:, 0:1].expand_as(Ks).contiguous()
        elif fix_k_mode == "mean":
            Ks = Ks.mean(dim=1, keepdim=True).expand_as(Ks).contiguous()
        elif fix_k_mode == "none":
            pass  # 保持每帧独立
        else:
            raise ValueError(f"Unknown fix_k_mode: {fix_k_mode}")
    else:
        Ks = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)
        Ks[:, :, 0, 0] = Ks[:, :, 1, 1] = render_w
        Ks[:, :, 0, 2] = render_w / 2.0
        Ks[:, :, 1, 2] = render_h / 2.0

    # Optional novel-view path: replace the predicted T poses with sampled
    # spiral poses while keeping the same 4D Gaussians. The caller is expected
    # to pass one override pose per output time frame, matching map-anything's
    # novel_path_time_video convention: camera path advances and timestamps use
    # the original frame cadence.
    if override_c2ws is not None:
        pose_mat = override_c2ws.to(device=device, dtype=torch.float32)
        if pose_mat.dim() == 3:
            pose_mat = pose_mat.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        M = pose_mat.shape[1]
        # Linearly resample Ks across T -> M so per-frame K stays smooth even
        # with fix_k_mode='none'. For median/first/mean the rows are already
        # identical, so this is a no-op.
        idx = torch.linspace(0, T - 1, M, device=device).round().long()
        Ks = Ks.index_select(1, idx).contiguous()
        if float(focal_zoom) <= 0:
            raise ValueError(f"focal_zoom must be > 0, got {focal_zoom}")
        if abs(float(focal_zoom) - 1.0) > 1e-6:
            Ks = Ks.clone()
            Ks[:, :, 0, 0] *= float(focal_zoom)
            Ks[:, :, 1, 1] *= float(focal_zoom)
        ts = (torch.arange(M, device=device, dtype=torch.float32) / 24.0).view(1, M, 1).expand(B, -1, -1)
    else:
        ts = (torch.arange(T, device=device, dtype=torch.float32) / 24.0).view(1, T, 1).expand(B, -1, -1)

    gs_clean = {k: _to_tensor(v, device).float() for k, v in gs_params.items() if k != "recovered_K"}

    with torch.no_grad():
        out = renderer(gs_params=gs_clean, ts=ts, Ks=Ks, RTs=pose_mat,
                       height=render_h, width=render_w, render_mode="RGB+ED")

    if "rgb" in out:
        out["rgb"] = (out["rgb"] + 1.0) / 2.0

        # Alpha-composite onto solid background. Rasterizer uses implicit
        # bg=0 (black), so rgb01 ≈ α·c; we add (1-α)·bg_color to replace the
        # implicit black with the requested color. Without this, opacity<1
        # regions look dimmer than the matching decoded video.
        if bg_color is not None and "mask" in out:
            bg = torch.as_tensor(bg_color, device=out["rgb"].device,
                                 dtype=out["rgb"].dtype).view(1, 1, 3, 1, 1)
            mask = out["mask"]  # (B, T, 1, H, W) in [0, 1]
            # Broadcast mask across RGB channels.
            out["rgb"] = (out["rgb"] + (1.0 - mask) * bg).clamp(0.0, 1.0)
    if "motion_mask" in out:
        out["motion_mask"] = (out["motion_mask"] + 1.0) / 2.0
    elif motion_mask and "mask" in out:
        # Allows flow/alpha-only masks when the ms3 gate is explicitly disabled.
        out["motion_mask"] = torch.ones_like(out["mask"])
    if motion_mask:
        _apply_motion_mask_output_gates(
            out,
            flow_residual_th=motion_mask_flow_residual_th,
            alpha_th=motion_mask_alpha_th,
        )
    return out

