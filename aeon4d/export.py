"""Saving results: videos/frames, depth & normal visualizations, 4DGS / static 3DGS checkpoints."""

import logging
import os
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils


logger = logging.getLogger(__name__)


def _apply_depth_colormap(depth_norm):
    """Apply turbo colormap to a normalized depth map [H, W] in [0,1] -> [3, H, W] tensor."""
    import matplotlib.cm as cm
    d_np = depth_norm.detach().cpu().clamp(0, 1).numpy()
    colored = cm.get_cmap('turbo')(d_np)[:, :, :3]  # (H, W, 3) float64
    return torch.from_numpy(colored).permute(2, 0, 1).float().to(depth_norm.device)


def save_cov_t_distribution(gs_params, save_dir, tag, batch_idx=0):
    """Save cov_t distribution of all Gaussians (across all frames) as a histogram."""
    if "cov_t" not in gs_params:
        logger.warning(f"gs_params has no 'cov_t' key, skip cov_t distribution for tag={tag}")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    data = gs_params["cov_t"][batch_idx].detach().float().cpu().numpy().ravel()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data, bins=200, alpha=0.7, color="tab:blue", density=True)
    ax.set_xlabel("cov_t value")
    ax.set_ylabel("density")
    ax.set_title(f"cov_t distribution — {tag} (all Gaussians, all frames)")
    fig.tight_layout()
    save_path = os.path.join(save_dir, f"b{batch_idx}_cov_t_dist_{tag}.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"cov_t distribution ({tag}) saved → {save_path}")


def save_4dgs_checkpoint(save_dir, gs_params, res, views, ff_resolution, batch_idx=0):
    """Save stitched 4DGS in world4d_js-compatible format (gs_params.pt + scene_info.json)."""
    import json
    from aeon4d._vendor.mapanything.utils.geometry import quaternion_to_rotation_matrix

    os.makedirs(save_dir, exist_ok=True)

    gs_cpu = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
              for k, v in gs_params.items()}
    torch.save(gs_cpu, os.path.join(save_dir, "gs_params.pt"))

    T = len(res)
    b = batch_idx

    def _qt_to_c2w(quat, trans):
        q = quat.detach().float().cpu().reshape(1, 4)
        R = quaternion_to_rotation_matrix(q)[0].numpy()
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R
        c2w[:3, 3] = trans.detach().float().cpu().numpy()
        return c2w.tolist()

    def _K_to_list(K):
        if isinstance(K, torch.Tensor):
            return K.detach().float().cpu().numpy().tolist()
        if isinstance(K, np.ndarray):
            return K.tolist()
        return K

    rec_K = gs_params.get("recovered_K")
    viewer_c2w, viewer_K = [], []
    for t in range(T):
        viewer_c2w.append(_qt_to_c2w(res[t]["cam_quats"][b], res[t]["cam_trans"][b]))
        if rec_K is not None:
            viewer_K.append(_K_to_list(rec_K[b, t]))
        elif "camera_intrinsics" in views[t]:
            viewer_K.append(_K_to_list(views[t]["camera_intrinsics"][b]))
        else:
            f = float(ff_resolution)
            viewer_K.append([[f, 0.0, f / 2.0], [0.0, f, f / 2.0], [0.0, 0.0, 1.0]])

    has_gt = all("camera_pose_quats" in v and "camera_pose_trans" in v for v in views)
    if has_gt:
        input_c2w, input_K = [], []
        for t in range(T):
            input_c2w.append(_qt_to_c2w(views[t]["camera_pose_quats"][b],
                                        views[t]["camera_pose_trans"][b]))
            if "camera_intrinsics" in views[t]:
                input_K.append(_K_to_list(views[t]["camera_intrinsics"][b]))
            else:
                input_K.append(viewer_K[t])
    else:
        input_c2w = [list(c) for c in viewer_c2w]
        input_K = [list(k) for k in viewer_K]

    scene_meta = {"frames": []}
    for t in range(T):
        K = viewer_K[t]
        scene_meta["frames"].append({
            "transform_matrix": viewer_c2w[t],
            "fl_x": float(K[0][0]),
            "fl_y": float(K[1][1]),
            "cx": float(K[0][2]),
            "cy": float(K[1][2]),
            "file_path": f"b{b}_t{t:03d}.png",
        })

    meta = {
        "gs_height": int(ff_resolution),
        "gs_width": int(ff_resolution),
        "scene_meta": scene_meta,
        "data_root": ".",
        "input_c2w": input_c2w,
        "input_K": input_K,
        "viewer_c2w": viewer_c2w,
        "viewer_K": viewer_K,
        "camera_source": "predicted_pose_only",
    }
    with open(os.path.join(save_dir, "scene_info.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved 4DGS checkpoint → {save_dir}")


def save_static_3dgs_checkpoint(save_dir, gs_params, res, views, ff_resolution, batch_idx=0):
    """Save a static GraphDECO/Inria-compatible 3DGS PLY.

    The source model predicts 4DGS fields shaped [B, T, H, W, C].  For static
    scenes we freeze every Gaussian at its current xyz/scale/rotation/opacity
    and ignore temporal fields (t/cov_t/ms3/omega), equivalent to zero velocity.
    """
    import json
    from plyfile import PlyData, PlyElement
    from aeon4d._vendor.mapanything.utils.geometry import quaternion_to_rotation_matrix

    required = ("xyz", "feature", "scaling", "rotation", "opacity")
    missing = [k for k in required if k not in gs_params]
    if missing:
        raise KeyError(f"Cannot save static 3DGS; missing gs_params keys: {missing}")

    os.makedirs(save_dir, exist_ok=True)
    b = batch_idx

    xyz = gs_params["xyz"][b].detach().float().reshape(-1, 3)
    colors = gs_params["feature"][b].detach().float().reshape(-1, gs_params["feature"].shape[-1])
    colors = colors[..., :3].clamp(0.0, 1.0)
    scales = gs_params["scaling"][b].detach().float().reshape(-1, 3).clamp_min(1e-8)
    rotations = F.normalize(
        gs_params["rotation"][b].detach().float().reshape(-1, 4),
        dim=-1,
        eps=1e-6,
    )
    opacities = gs_params["opacity"][b].detach().float().reshape(-1, 1).clamp(1e-6, 1.0 - 1e-6)

    finite_mask = (
        torch.isfinite(xyz).all(dim=-1)
        & torch.isfinite(colors).all(dim=-1)
        & torch.isfinite(scales).all(dim=-1)
        & torch.isfinite(rotations).all(dim=-1)
        & torch.isfinite(opacities).all(dim=-1)
    )
    if not bool(finite_mask.all()):
        dropped = int((~finite_mask).sum().item())
        logger.warning(f"[static_3dgs] dropped {dropped} non-finite Gaussian(s) before PLY export.")
        xyz = xyz[finite_mask]
        colors = colors[finite_mask]
        scales = scales[finite_mask]
        rotations = rotations[finite_mask]
        opacities = opacities[finite_mask]

    ply_path = os.path.join(save_dir, "point_cloud.ply")

    xyz_np = xyz.cpu().numpy()
    normals = np.zeros_like(xyz_np)
    c0 = 0.28209479177387814
    features_dc = ((colors.cpu().numpy() - 0.5) / c0)[:, :, None]
    features_rest = np.zeros((xyz_np.shape[0], 3, 0), dtype=np.float32)
    f_dc = features_dc.transpose(0, 2, 1).reshape(xyz_np.shape[0], -1)
    f_rest = features_rest.transpose(0, 2, 1).reshape(xyz_np.shape[0], -1)
    opacity_np = opacities.cpu().numpy()
    opacity_np = np.log(opacity_np / (1.0 - opacity_np))
    scale_np = np.log(scales.cpu().numpy())
    rotation_np = rotations.cpu().numpy()

    attribute_names = ["x", "y", "z", "nx", "ny", "nz"]
    attribute_names += [f"f_dc_{i}" for i in range(f_dc.shape[1])]
    attribute_names += [f"f_rest_{i}" for i in range(f_rest.shape[1])]
    attribute_names += ["opacity"]
    attribute_names += [f"scale_{i}" for i in range(scale_np.shape[1])]
    attribute_names += [f"rot_{i}" for i in range(rotation_np.shape[1])]

    elements = np.empty(xyz_np.shape[0], dtype=[(name, "f4") for name in attribute_names])
    for i, name in enumerate(("x", "y", "z")):
        elements[name] = xyz_np[:, i]
    for i, name in enumerate(("nx", "ny", "nz")):
        elements[name] = normals[:, i]
    for i in range(f_dc.shape[1]):
        elements[f"f_dc_{i}"] = f_dc[:, i]
    for i in range(f_rest.shape[1]):
        elements[f"f_rest_{i}"] = f_rest[:, i]
    elements["opacity"] = opacity_np[:, 0]
    for i in range(scale_np.shape[1]):
        elements[f"scale_{i}"] = scale_np[:, i]
    for i in range(rotation_np.shape[1]):
        elements[f"rot_{i}"] = rotation_np[:, i]
    PlyData([PlyElement.describe(elements, "vertex")]).write(ply_path)

    def _qt_to_c2w(quat, trans):
        q = quat.detach().float().cpu().reshape(1, 4)
        R = quaternion_to_rotation_matrix(q)[0].numpy()
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R
        c2w[:3, 3] = trans.detach().float().cpu().numpy()
        return c2w.tolist()

    def _K_to_list(K):
        if isinstance(K, torch.Tensor):
            return K.detach().float().cpu().numpy().tolist()
        if isinstance(K, np.ndarray):
            return K.tolist()
        return K

    T = len(res)
    rec_K = gs_params.get("recovered_K")
    frames = []
    for t in range(T):
        if rec_K is not None:
            K = _K_to_list(rec_K[b, t])
        elif "camera_intrinsics" in views[t]:
            K = _K_to_list(views[t]["camera_intrinsics"][b])
        else:
            f = float(ff_resolution)
            K = [[f, 0.0, f / 2.0], [0.0, f, f / 2.0], [0.0, 0.0, 1.0]]
        frames.append({
            "transform_matrix": _qt_to_c2w(res[t]["cam_quats"][b], res[t]["cam_trans"][b]),
            "fl_x": float(K[0][0]),
            "fl_y": float(K[1][1]),
            "cx": float(K[0][2]),
            "cy": float(K[1][2]),
            "file_path": f"b{b}_t{t:03d}.png",
        })

    meta = {
        "format": "graphdeco_static_3dgs",
        "point_cloud": "point_cloud.ply",
        "point_count": int(xyz.shape[0]),
        "source_shape": list(gs_params["xyz"][b].shape),
        "source_temporal_fields_ignored": ["t", "cov_t", "ms3", "omega", "dxyzt"],
        "velocity_assumption": "zero",
        "sh_degree": 0,
        "gs_height": int(ff_resolution),
        "gs_width": int(ff_resolution),
        "scene_meta": {"frames": frames},
        "camera_source": "predicted_pose_only",
    }
    with open(os.path.join(save_dir, "scene_info.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved static 3DGS checkpoint → {save_dir}")


def save_spiral_render(output_dir, render_out, fps=8):
    """Save a spiral-render output as compact MP4s (RGB + depth + normal).

    Mirrors the per-modality writes from save_generation but skips per-frame
    PNG dumps and the side-by-side video — spiral renders are a derivative
    visualization, the canonical comparison artifact stays the predicted-pose
    render produced by save_generation.
    """
    os.makedirs(output_dir, exist_ok=True)
    if render_out is None or "rgb" not in render_out:
        logger.warning(f"spiral render is empty, nothing to save in {output_dir}")
        return
    rgb = render_out["rgb"]  # (B, M, 3, H, W)
    B, M = rgb.shape[:2]

    def _save_video(frames, path):
        imgs = []
        for f in frames:
            img = f.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            imgs.append((img * 255).astype(np.uint8))
        imageio.mimsave(path, imgs, fps=fps, macro_block_size=1)

    for b in range(B):
        rgb_frames = [rgb[b, t] for t in range(M)]
        _save_video(rgb_frames, os.path.join(output_dir, f"b{b}_spiral_rgb.mp4"))

        if "depth" in render_out:
            depth = render_out["depth"]
            d_all = depth[b, :, 0].float()
            inv_all = 1.0 / d_all.clamp(min=1e-6)
            inv_flat = inv_all.flatten()
            n = inv_flat.numel()
            inv_lo = inv_flat.kthvalue(max(1, int(0.02 * n))).values.item()
            inv_hi = inv_flat.kthvalue(max(1, int(0.98 * n))).values.item()
            d_frames = []
            for t in range(M):
                d = depth[b, t, 0].float()
                inv = 1.0 / d.clamp(min=1e-6)
                d_vis = ((inv - inv_lo) / (inv_hi - inv_lo + 1e-8)).clamp(0, 1)
                d_frames.append(_apply_depth_colormap(d_vis))
            _save_video(d_frames, os.path.join(output_dir, f"b{b}_spiral_depth.mp4"))

        if "normal" in render_out:
            gs_normal = render_out["normal"]
            gn_frames = [(gs_normal[b, t] + 1.0) / 2.0 for t in range(M)]
            _save_video(gn_frames, os.path.join(output_dir, f"b{b}_spiral_normal_gs.mp4"))

        if "motion_mask" in render_out:
            motion_mask = render_out["motion_mask"]
            mm_frames = [
                motion_mask[b, t, 0].clamp(0, 1).unsqueeze(0).repeat(3, 1, 1)
                for t in range(motion_mask.shape[1])
            ]
            _save_video(mm_frames, os.path.join(output_dir, f"b{b}_spiral_motion_mask.mp4"))


def save_generation(output_dir, prompt, decoded_video, res, gs_params,
                    render_out, ff_resolution, fps=8):
    """Save all generation outputs: video, renders, PLY."""
    os.makedirs(output_dir, exist_ok=True)

    # Save prompt
    with open(os.path.join(output_dir, "prompt.txt"), "w") as f:
        f.write(prompt)

    B = decoded_video.shape[0]
    T = decoded_video.shape[2]

    def _save_video(frames, path):
        imgs = []
        for f in frames:
            img = f.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            imgs.append((img * 255).astype(np.uint8))
        imageio.mimsave(path, imgs, fps=fps, macro_block_size=1)

    for b in range(B):
        # 1) Generated video (from Wan diffusion)
        gen_frames = [(decoded_video[b, :, t] + 1) / 2 for t in range(T)]
        _save_video(gen_frames, os.path.join(output_dir, f"b{b}_generated_video.mp4"))

        # 2) Rendered RGB video
        if render_out is not None and "rgb" in render_out:
            rgb = render_out["rgb"]
            render_frames = [rgb[b, t] for t in range(rgb.shape[1])]
            _save_video(render_frames, os.path.join(output_dir, f"b{b}_render_rgb.mp4"))

            # Side-by-side: generated | rendered
            sbs_frames = []
            for t in range(min(len(gen_frames), len(render_frames))):
                g = F.interpolate(gen_frames[t].unsqueeze(0).to(render_frames[t].device),
                                  size=render_frames[t].shape[1:],
                                  mode="bilinear", align_corners=False)[0]
                sbs = torch.cat([g.clamp(0, 1), render_frames[t].clamp(0, 1)], dim=2)
                sbs_frames.append(sbs)
            _save_video(sbs_frames, os.path.join(output_dir, f"b{b}_side_by_side.mp4"))

        # 3) Rendered depth video — 用 inverse depth (1/d) + 全局 2%/98% percentile 归一化:
        #    - inverse: 近处 1/d 大、远处 1/d 小,避免远景大深度把近处动态范围压扁
        #    - 全局范围: 跨帧共享颜色映射,保持帧间可比
        #    - percentile: 裁掉极端 outlier (极近处 1/d 爆掉)
        #    turbo colormap 下: 近处→红 (高值), 远处→蓝 (低值), 和之前正好反过来
        if render_out is not None and "depth" in render_out:
            depth = render_out["depth"]
            d_all = depth[b, :, 0].float()
            inv_all = 1.0 / d_all.clamp(min=1e-6)
            inv_flat = inv_all.flatten()
            n = inv_flat.numel()
            inv_lo = inv_flat.kthvalue(max(1, int(0.02 * n))).values.item()
            inv_hi = inv_flat.kthvalue(max(1, int(0.98 * n))).values.item()
            d_frames = []
            for t in range(depth.shape[1]):
                d = depth[b, t, 0].float()
                inv = 1.0 / d.clamp(min=1e-6)
                d_vis = ((inv - inv_lo) / (inv_hi - inv_lo + 1e-8)).clamp(0, 1)
                d_frames.append(_apply_depth_colormap(d_vis))
            _save_video(d_frames, os.path.join(output_dir, f"b{b}_render_depth.mp4"))

        # 3b) Gaussian-rendered normal video
        if render_out is not None and "normal" in render_out:
            gs_normal = render_out["normal"]  # (B, T, 3, H, W) in [-1, 1]
            gn_frames = [(gs_normal[b, t] + 1.0) / 2.0 for t in range(gs_normal.shape[1])]
            _save_video(gn_frames, os.path.join(output_dir, f"b{b}_render_normal_gs.mp4"))

        # 3c) Depth-derived surface normal video
        if render_out is not None and "surf_normal" in render_out:
            surf_normal = render_out["surf_normal"]  # (B, T, 3, H, W) in [-1, 1]
            sn_frames = [(surf_normal[b, t] + 1.0) / 2.0 for t in range(surf_normal.shape[1])]
            _save_video(sn_frames, os.path.join(output_dir, f"b{b}_render_normal_surf.mp4"))

        # 3d) Motion mask video (renderer vis_motion)
        if render_out is not None and "motion_mask" in render_out:
            motion_mask = render_out["motion_mask"]  # (B, T, 1, H, W) in [0, 1]
            mm_frames = []
            for t in range(motion_mask.shape[1]):
                m = motion_mask[b, t, 0].clamp(0, 1)
                mm_frames.append(m.unsqueeze(0).repeat(3, 1, 1))
            _save_video(mm_frames, os.path.join(output_dir, f"b{b}_motion_mask_raw.mp4"))

        # 4) Save individual rendered frames
        frame_dir = os.path.join(output_dir, "frames")
        os.makedirs(frame_dir, exist_ok=True)
        for t in range(T):
            vutils.save_image(gen_frames[t].clamp(0, 1),
                              os.path.join(frame_dir, f"b{b}_gen_t{t:03d}.png"))
        if render_out is not None and "rgb" in render_out:
            for t in range(render_out["rgb"].shape[1]):
                vutils.save_image(render_out["rgb"][b, t].clamp(0, 1),
                                  os.path.join(frame_dir, f"b{b}_render_t{t:03d}.png"))

        # Save Gaussian-rendered normal frames
        if render_out is not None and "normal" in render_out:
            gs_normal_dir = os.path.join(output_dir, "render_normal_gs")
            os.makedirs(gs_normal_dir, exist_ok=True)
            gs_normal = render_out["normal"]
            for t in range(gs_normal.shape[1]):
                n_vis = (gs_normal[b, t] + 1.0) / 2.0
                vutils.save_image(n_vis.clamp(0, 1), os.path.join(gs_normal_dir, f"b{b}_t{t:03d}.png"))

        # Save depth-derived surface normal frames
        if render_out is not None and "surf_normal" in render_out:
            surf_normal_dir = os.path.join(output_dir, "render_normal_surf")
            os.makedirs(surf_normal_dir, exist_ok=True)
            surf_normal = render_out["surf_normal"]
            for t in range(surf_normal.shape[1]):
                n_vis = (surf_normal[b, t] + 1.0) / 2.0
                vutils.save_image(n_vis.clamp(0, 1), os.path.join(surf_normal_dir, f"b{b}_t{t:03d}.png"))

        # Save motion mask frames
        if render_out is not None and "motion_mask" in render_out:
            motion_mask_dir = os.path.join(output_dir, "motion_mask")
            os.makedirs(motion_mask_dir, exist_ok=True)
            motion_mask = render_out["motion_mask"]
            for t in range(motion_mask.shape[1]):
                m = motion_mask[b, t, 0].clamp(0, 1)
                vutils.save_image(
                    m.unsqueeze(0).repeat(3, 1, 1),
                    os.path.join(motion_mask_dir, f"b{b}_t{t:03d}.png"),
                )

    logger.info(f"Outputs saved to {output_dir}")

