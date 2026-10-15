"""
Utilities for geometry operations.

References: DUSt3R, MoGe
"""
import einops as ein
import torch
from aeon4d._vendor.mapanything.utils.misc import invalid_to_zeros


def get_rays_in_camera_frame(intrinsics, height, width, normalize_to_unit_sphere):
    """
    Convert camera intrinsics to a raymap (ray origins + directions) in camera frame.
    Note: Currently only supports pinhole camera model.

    Args:
        - intrinsics: 3x3 or Bx3x3 torch tensor
        - height: int
        - width: int
        - normalize_to_unit_sphere: bool

    Returns:
        - ray_origins: (HxWx3 or BxHxWx3) tensor
        - ray_directions: (HxWx3 or BxHxWx3) tensor
    """
    # Add batch dimension if not present
    if intrinsics.dim() == 2:
        intrinsics = intrinsics.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    batch_size = intrinsics.shape[0]
    device = intrinsics.device

    # Compute rays in camera frame associated with each pixel
    x_grid, y_grid = torch.meshgrid(
        torch.arange(width, device=device).float(),
        torch.arange(height, device=device).float(),
        indexing="xy",
    )
    x_grid = x_grid.unsqueeze(0).expand(batch_size, -1, -1)
    y_grid = y_grid.unsqueeze(0).expand(batch_size, -1, -1)

    fx = intrinsics[:, 0, 0].view(-1, 1, 1)
    fy = intrinsics[:, 1, 1].view(-1, 1, 1)
    cx = intrinsics[:, 0, 2].view(-1, 1, 1)
    cy = intrinsics[:, 1, 2].view(-1, 1, 1)

    ray_origins = torch.zeros((batch_size, height, width, 3), device=device)
    xx = (x_grid - cx) / fx
    yy = (y_grid - cy) / fy
    ray_directions = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)

    # Normalize ray directions to unit sphere if required (else rays will lie on unit plane)
    if normalize_to_unit_sphere:
        ray_directions = ray_directions / torch.norm(
            ray_directions, dim=-1, keepdim=True
        )

    # Remove batch dimension if it was added
    if squeeze_batch_dim:
        ray_origins = ray_origins.squeeze(0)
        ray_directions = ray_directions.squeeze(0)

    return ray_origins, ray_directions


def recover_pinhole_intrinsics_from_ray_directions(
    ray_directions, use_geometric_calculation=False
):
    """
    Recover pinhole camera intrinsics from ray directions, supporting both batched and non-batched inputs.
    All internal computation is done in fp32 for numerical stability (safe under bf16 autocast).

    Args:
        ray_directions: Tensor of shape [H, W, 3] or [B, H, W, 3] containing unit normalized ray directions

    Returns:
        Intrinsics matrix of shape [3, 3] or [B, 3, 3]
    """
    original_dtype = ray_directions.dtype
    ray_directions = ray_directions.float()

    if ray_directions.dim() == 3:
        ray_directions = ray_directions.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    batch_size, height, width, _ = ray_directions.shape
    device = ray_directions.device

    _Z_EPS = 0.05  # increased from 1e-4: prevents gradient explosion in dx/dz when dz is small
    _DENOM_EPS = 1e-3  # increased from 1e-6: prevents gradient explosion in fx = pixel_diff / dx
    _FX_MIN, _FX_MAX = 50.0, 5000.0

    x_grid, y_grid = torch.meshgrid(
        torch.arange(width, device=device).float(),
        torch.arange(height, device=device).float(),
        indexing="xy",
    )
    x_grid = x_grid.unsqueeze(0).expand(batch_size, -1, -1)
    y_grid = y_grid.unsqueeze(0).expand(batch_size, -1, -1)

    is_high_res = height * width > 1000000

    if is_high_res or use_geometric_calculation:
        center_h, center_w = height // 2, width // 2
        quarter_w, three_quarter_w = width // 4, 3 * width // 4
        quarter_h, three_quarter_h = height // 4, 3 * height // 4

        center_rays = ray_directions[:, center_h, center_w, :].clone()
        left_rays = ray_directions[:, center_h, quarter_w, :].clone()
        right_rays = ray_directions[:, center_h, three_quarter_w, :].clone()
        top_rays = ray_directions[:, quarter_h, center_w, :].clone()
        bottom_rays = ray_directions[:, three_quarter_h, center_w, :].clone()

        center_rays = center_rays / center_rays[:, 2:3].clamp(min=_Z_EPS)
        left_rays = left_rays / left_rays[:, 2:3].clamp(min=_Z_EPS)
        right_rays = right_rays / right_rays[:, 2:3].clamp(min=_Z_EPS)
        top_rays = top_rays / top_rays[:, 2:3].clamp(min=_Z_EPS)
        bottom_rays = bottom_rays / bottom_rays[:, 2:3].clamp(min=_Z_EPS)

        dx_left = left_rays[:, 0] - center_rays[:, 0]
        dx_right = right_rays[:, 0] - center_rays[:, 0]
        dx_left = torch.where(dx_left.abs() < _DENOM_EPS, torch.full_like(dx_left, _DENOM_EPS), dx_left)
        dx_right = torch.where(dx_right.abs() < _DENOM_EPS, torch.full_like(dx_right, _DENOM_EPS), dx_right)

        fx_left = (quarter_w - center_w) / dx_left
        fx_right = (three_quarter_w - center_w) / dx_right
        fx = (fx_left + fx_right) / 2
        cx = center_w - fx * center_rays[:, 0]

        dy_top = top_rays[:, 1] - center_rays[:, 1]
        dy_bottom = bottom_rays[:, 1] - center_rays[:, 1]
        dy_top = torch.where(dy_top.abs() < _DENOM_EPS, torch.full_like(dy_top, _DENOM_EPS), dy_top)
        dy_bottom = torch.where(dy_bottom.abs() < _DENOM_EPS, torch.full_like(dy_bottom, _DENOM_EPS), dy_bottom)

        fy_top = (quarter_h - center_h) / dy_top
        fy_bottom = (three_quarter_h - center_h) / dy_bottom
        fy = (fy_top + fy_bottom) / 2
        cy = center_h - fy * center_rays[:, 1]
    else:
        step_h = max(1, height // 50)
        step_w = max(1, width // 50)

        h_indices = torch.arange(0, height, step_h, device=device)
        w_indices = torch.arange(0, width, step_w, device=device)

        x_sampled = x_grid[:, h_indices[:, None], w_indices[None, :]]
        y_sampled = y_grid[:, h_indices[:, None], w_indices[None, :]]
        rays_sampled = ray_directions[:, h_indices[:, None], w_indices[None, :], :]

        x_flat = x_sampled.reshape(batch_size, -1)
        y_flat = y_sampled.reshape(batch_size, -1)

        dx = rays_sampled[..., 0].reshape(batch_size, -1)
        dy = rays_sampled[..., 1].reshape(batch_size, -1)
        dz = rays_sampled[..., 2].reshape(batch_size, -1).clamp(min=_Z_EPS)

        ratio_x = dx / dz
        ratio_y = dy / dz

        ones = torch.ones_like(x_flat)
        A_x = torch.stack([ones, ratio_x], dim=2)
        b_x = x_flat.unsqueeze(2)

        ATA_x = torch.bmm(A_x.transpose(1, 2), A_x)
        ATb_x = torch.bmm(A_x.transpose(1, 2), b_x)

        # Tikhonov regularization: bounds gradient norm of (ATA)^-1 to 1/reg
        _REG_EYE = 1e-1 * torch.eye(2, device=device, dtype=ATA_x.dtype).unsqueeze(0)
        solution_x = torch.linalg.solve(ATA_x + _REG_EYE, ATb_x).squeeze(2)
        cx, fx = solution_x[:, 0], solution_x[:, 1]

        A_y = torch.stack([ones, ratio_y], dim=2)
        b_y = y_flat.unsqueeze(2)

        ATA_y = torch.bmm(A_y.transpose(1, 2), A_y)
        ATb_y = torch.bmm(A_y.transpose(1, 2), b_y)

        solution_y = torch.linalg.solve(ATA_y + _REG_EYE, ATb_y).squeeze(2)
        cy, fy = solution_y[:, 0], solution_y[:, 1]

    fx = fx.clamp(min=_FX_MIN, max=_FX_MAX)
    fy = fy.clamp(min=_FX_MIN, max=_FX_MAX)
    cx = cx.clamp(min=0.0, max=float(width))
    cy = cy.clamp(min=0.0, max=float(height))

    intrinsics = torch.zeros(batch_size, 3, 3, device=device, dtype=torch.float32)
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy
    intrinsics[:, 2, 2] = 1.0

    if squeeze_batch_dim:
        intrinsics = intrinsics.squeeze(0)

    return intrinsics.to(original_dtype)


def quaternion_to_rotation_matrix(quat):
    """
    Convert a quaternion into a 3x3 rotation matrix.

    Args:
        - quat: 4 or Bx4 torch tensor (unit quaternions and notation is (x, y, z, w))

    Returns:
        - rot_matrix: 3x3 or Bx3x3 torch tensor
    """
    if quat.dim() == 1:
        quat = quat.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    # Ensure the quaternion is normalized
    quat = quat / quat.norm(dim=1, keepdim=True)
    x, y, z, w = quat.unbind(dim=1)

    # Compute the rotation matrix elements
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    # Construct the rotation matrix
    rot_matrix = torch.stack(
        [
            1 - 2 * (yy + zz),
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            1 - 2 * (xx + zz),
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            1 - 2 * (xx + yy),
        ],
        dim=1,
    ).view(-1, 3, 3)

    # Squeeze batch dimension if it was unsqueezed
    if squeeze_batch_dim:
        rot_matrix = rot_matrix.squeeze(0)

    return rot_matrix


def quaternion_inverse(quat):
    """
    Compute the inverse of a quaternion.

    Args:
        - quat: 4 or Bx4 torch tensor (unit quaternions and notation is (x, y, z, w))

    Returns:
        - inv_quat: 4 or Bx4 torch tensor (unit quaternions and notation is (x, y, z, w))
    """
    # Unsqueeze batch dimension if not present
    if quat.dim() == 1:
        quat = quat.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    # Compute the inverse
    quat_conj = quat.clone()
    quat_conj[:, :3] = -quat_conj[:, :3]
    quat_norm = torch.sum(quat * quat, dim=1, keepdim=True)
    inv_quat = quat_conj / quat_norm

    # Squeeze batch dimension if it was unsqueezed
    if squeeze_batch_dim:
        inv_quat = inv_quat.squeeze(0)

    return inv_quat


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.

    Args:
        - q1: 4 or Bx4 torch tensor (unit quaternions and notation is (x, y, z, w))
        - q2: 4 or Bx4 torch tensor (unit quaternions and notation is (x, y, z, w))

    Returns:
        - qm: 4 or Bx4 torch tensor (unit quaternions and notation is (x, y, z, w))
    """
    # Unsqueeze batch dimension if not present
    if q1.dim() == 1:
        q1 = q1.unsqueeze(0)
        q2 = q2.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    # Unbind the quaternions
    x1, y1, z1, w1 = q1.unbind(dim=1)
    x2, y2, z2, w2 = q2.unbind(dim=1)

    # Compute the product
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    # Stack the components
    qm = torch.stack([x, y, z, w], dim=1)

    # Squeeze batch dimension if it was unsqueezed
    if squeeze_batch_dim:
        qm = qm.squeeze(0)

    return qm


def transform_pose_using_quats_and_trans_2_to_1(quats1, trans1, quats2, trans2):
    """
    Transform quats and translation of pose2 from absolute frame (pose2 to world) to relative frame (pose2 to pose1).

    Args:
        - quats1: 4 or Bx4 torch tensor (unit quaternions and notation is (x, y, z, w))
        - trans1: 3 or Bx3 torch tensor
        - quats2: 4 or Bx4 torch tensor (unit quaternions and notation is (x, y, z, w))
        - trans2: 3 or Bx3 torch tensor

    Returns:
        - quats: 4 or Bx4 torch tensor (unit quaternions and notation is (x, y, z, w))
        - trans: 3 or Bx3 torch tensor
    """
    # Unsqueeze batch dimension if not present
    if quats1.dim() == 1:
        quats1 = quats1.unsqueeze(0)
        trans1 = trans1.unsqueeze(0)
        quats2 = quats2.unsqueeze(0)
        trans2 = trans2.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    # Compute the inverse of view1's pose
    inv_quats1 = quaternion_inverse(quats1)
    R1_inv = quaternion_to_rotation_matrix(inv_quats1)
    t1_inv = -1 * ein.einsum(R1_inv, trans1, "b i j, b j -> b i")

    # Transform view2's pose to view1's frame
    quats = quaternion_multiply(inv_quats1, quats2)
    trans = ein.einsum(R1_inv, trans2, "b i j, b j -> b i") + t1_inv

    # Squeeze batch dimension if it was unsqueezed
    if squeeze_batch_dim:
        quats = quats.squeeze(0)
        trans = trans.squeeze(0)

    return quats, trans


def convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
    ray_directions, depth_along_ray, pose_trans, pose_quats
):
    """
    Convert ray directions, depth along ray, pose translation, and
    unit quaternions (representing pose rotation) to a pointmap in world frame.

    Args:
        - ray_directions: (HxWx3 or BxHxWx3) torch tensor
        - depth_along_ray: (HxWx1 or BxHxWx1) torch tensor
        - pose_trans: (3 or Bx3) torch tensor
        - pose_quats: (4 or Bx4) torch tensor (unit quaternions and notation is (x, y, z, w))

    Returns:
        - pointmap: (HxWx3 or BxHxWx3) torch tensor
    """
    # Add batch dimension if not present
    if ray_directions.dim() == 3:
        ray_directions = ray_directions.unsqueeze(0)
        depth_along_ray = depth_along_ray.unsqueeze(0)
        pose_trans = pose_trans.unsqueeze(0)
        pose_quats = pose_quats.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    batch_size, height, width, _ = depth_along_ray.shape
    device = depth_along_ray.device

    # Normalize the quaternions to ensure they are unit quaternions
    pose_quats = pose_quats / torch.norm(pose_quats, dim=-1, keepdim=True)

    # Convert quaternions to rotation matrices (B x 3 x 3)
    rot_mat = quaternion_to_rotation_matrix(pose_quats)

    # Get pose matrix (B x 4 x 4)
    pose_mat = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    pose_mat[:, :3, :3] = rot_mat
    pose_mat[:, :3, 3] = pose_trans

    # Compute 3D points in local camera frame
    pts3d_local = depth_along_ray * ray_directions

    # Compute 3D points in world frame
    pts3d_homo = torch.cat([pts3d_local, torch.ones_like(pts3d_local[..., :1])], dim=-1)
    pts3d_world = ein.einsum(pose_mat, pts3d_homo, "b i k, b h w k -> b h w i")
    pts3d_world = pts3d_world[..., :3]

    # Remove batch dimension if it was added
    if squeeze_batch_dim:
        pts3d_world = pts3d_world.squeeze(0)

    return pts3d_world


def normalize_depth_using_non_zero_pixels(depth, return_norm_factor=False):
    """
    Normalize the depth by the average depth of non-zero depth pixels.

    Args:
        depth (torch.Tensor): Depth tensor of size [B, H, W, 1].
    Returns:
        normalized_depth (torch.Tensor): Normalized depth tensor.
        norm_factor (torch.Tensor): Norm factor tensor of size B.
    """
    assert depth.ndim == 4 and depth.shape[3] == 1
    # Calculate the sum and count of non-zero depth pixels for each batch
    valid_depth_mask = depth > 0
    valid_sum = torch.sum(depth * valid_depth_mask, dim=(1, 2, 3))
    valid_count = torch.sum(valid_depth_mask, dim=(1, 2, 3))

    # Calculate the norm factor
    norm_factor = valid_sum / (valid_count + 1e-8)
    while norm_factor.ndim < depth.ndim:
        norm_factor.unsqueeze_(-1)

    # Normalize the depth by the norm factor
    norm_factor = norm_factor.clip(min=1e-8)
    normalized_depth = depth / norm_factor

    # Create the output tuple
    output = (
        (normalized_depth, norm_factor.squeeze(-1).squeeze(-1).squeeze(-1))
        if return_norm_factor
        else normalized_depth
    )

    return output


def normalize_pose_translations(pose_translations, return_norm_factor=False):
    """
    Normalize the pose translations by the average norm of the non-zero pose translations.

    Args:
        pose_translations (torch.Tensor): Pose translations tensor of size [B, V, 3]. B is the batch size, V is the number of views.
    Returns:
        normalized_pose_translations (torch.Tensor): Normalized pose translations tensor of size [B, V, 3].
        norm_factor (torch.Tensor): Norm factor tensor of size B.
    """
    assert pose_translations.ndim == 3 and pose_translations.shape[2] == 3
    # Compute distance of all pose translations to origin
    pose_translations_dis = pose_translations.norm(dim=-1)  # [B, V]
    non_zero_pose_translations_dis = pose_translations_dis > 0  # [B, V]

    # Calculate the average norm of the translations across all views (considering only views with non-zero translations)
    sum_of_all_views_pose_translations = pose_translations_dis.sum(dim=1)  # [B]
    count_of_all_views_with_non_zero_pose_translations = (
        non_zero_pose_translations_dis.sum(dim=1)
    )  # [B]
    norm_factor = sum_of_all_views_pose_translations / (
        count_of_all_views_with_non_zero_pose_translations + 1e-8
    )  # [B]

    # Normalize the pose translations by the norm factor
    norm_factor = norm_factor.clip(min=1e-8)
    normalized_pose_translations = pose_translations / norm_factor.unsqueeze(
        -1
    ).unsqueeze(-1)

    # Create the output tuple
    output = (
        (normalized_pose_translations, norm_factor)
        if return_norm_factor
        else normalized_pose_translations
    )

    return output


def normalize_multiple_pointclouds(
    pts_list, valid_masks=None, norm_mode="avg_dis", ret_factor=False
):
    """
    Normalize multiple point clouds using a joint normalization strategy.

    Args:
        pts_list: List of point clouds, each with shape (..., H, W, 3) or (B, H, W, 3)
        valid_masks: Optional list of masks indicating valid points in each point cloud
        norm_mode: String in format "{norm}_{dis}" where:
            - norm: Normalization strategy (currently only "avg" is supported)
            - dis: Distance transformation ("dis" for raw distance, "log1p" for log(1+distance),
                  "warp-log1p" to warp points using log distance)
        ret_factor: If True, return the normalization factor as the last element in the result list

    Returns:
        List of normalized point clouds with the same shapes as inputs.
        If ret_factor is True, the last element is the normalization factor.
    """
    assert all(pts.ndim >= 3 and pts.shape[-1] == 3 for pts in pts_list)
    if valid_masks is not None:
        assert len(pts_list) == len(valid_masks)

    norm_mode, dis_mode = norm_mode.split("_")

    # Gather all points together (joint normalization)
    nan_pts_list = [
        invalid_to_zeros(pts, valid_masks[i], ndim=3)
        if valid_masks
        else invalid_to_zeros(pts, None, ndim=3)
        for i, pts in enumerate(pts_list)
    ]
    all_pts = torch.cat([nan_pts for nan_pts, _ in nan_pts_list], dim=1)
    nnz_list = [nnz for _, nnz in nan_pts_list]

    # Compute distance to origin
    eps = 1e-8
    all_dis = torch.sqrt((all_pts ** 2).sum(dim=-1) + eps)
    
    if dis_mode == "dis":
        pass  # do nothing
    elif dis_mode == "log1p":
        all_dis = torch.log1p(all_dis)
    elif dis_mode == "warp-log1p":
        # Warp input points before normalizing them
        log_dis = torch.log1p(all_dis)
        warp_factor = log_dis / all_dis.clip(min=1e-8)
        for i, pts in enumerate(pts_list):
            H, W = pts.shape[1:-1]
            pts_list[i] = pts * warp_factor[:, i * (H * W) : (i + 1) * (H * W)].view(
                -1, H, W, 1
            )
        all_dis = log_dis
    else:
        raise ValueError(f"bad {dis_mode=}")

    # Compute normalization factor
    norm_factor = all_dis.sum(dim=1) / (sum(nnz_list) + 1e-8)
    norm_factor = norm_factor.clip(min=1e-8)
    while norm_factor.ndim < pts_list[0].ndim:
        norm_factor.unsqueeze_(-1)

    # Normalize points
    res = [pts / norm_factor for pts in pts_list]
    if ret_factor:
        res.append(norm_factor)

    return res


def apply_log_to_norm(input_data):
    """
    Normalize the input data and apply a logarithmic transformation based on the normalization factor.

    Args:
        input_data (torch.Tensor): The input tensor to be normalized and transformed.

    Returns:
        torch.Tensor: The transformed tensor after normalization and logarithmic scaling.
    """
    org_d = input_data.norm(dim=-1, keepdim=True)
    input_data = input_data / org_d.clip(min=1e-8)
    input_data = input_data * torch.log1p(org_d)
    return input_data


def radius_to_sigma(radius, cutoff: float = 0.05):
    # Calculate sigma
    is_tensor = isinstance(radius, torch.Tensor)
    if not is_tensor:
        radius = torch.as_tensor(radius)
    sigma = radius / torch.sqrt(
        -2 * torch.log(torch.as_tensor(cutoff).to(radius, non_blocking=True))
    )
    if not is_tensor:
        sigma = sigma.item()
    return sigma


def dt_to_cov_t(dt: torch.Tensor, marginal_th: float = 0.05):
    is_tensor = isinstance(dt, torch.Tensor)
    if not is_tensor:
        dt = torch.as_tensor(dt)
    cov_t = (dt**2) / (
        torch.log(torch.as_tensor(marginal_th).to(dt, non_blocking=True)) / -0.5
    )
    # cov_t = (dt**2) / ( (torch.log(torch.as_tensor(marginal_th).to(dt, non_blocking=True)) / -0.5) * 4.0 )
    if not is_tensor:
        cov_t = cov_t.item()
    return cov_t
