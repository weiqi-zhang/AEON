"""Subset of EasyVolcap utilities (https://github.com/zju3dv/EasyVolcap) used by the 4DGS renderer
(chunk_utils, base_utils, math_utils, quat_utils, flow_utils), extracted verbatim."""
from __future__ import annotations
from copy import copy
from math import pi as PI
from typing import Callable, Dict, Mapping, TypeVar
from typing import Optional, Union
import torch


# ---- from easyvolcap/utils/chunk_utils.py ----
def multi_indexing(indices: torch.Tensor, shape: torch.Size, dim=-2):
    # index will first be augmented to match the values' dimentionality at the back
    # then we will try to broatcast index's shape to values shape
    shape = list(shape)
    back_pad = len(shape) - indices.ndim
    for _ in range(back_pad):
        indices = indices.unsqueeze(-1)
    expand_shape = shape
    expand_shape[dim] = -1
    return indices.expand(*expand_shape)


def multi_gather(values: torch.Tensor, indices: torch.Tensor, dim=-2):
    # Gather the value at the -2th dim of values, augment index shape on the back
    # Example: values: B, P, 3, index: B, N, -> B, N, 3

    # index will first be augmented to match the values' dimentionality at the back
    # take care of batch dimension of, and acts like a linear indexing in the target dimention
    # we assume that the values's second to last dimension is the dimension to be indexed on
    return values.gather(dim, multi_indexing(indices, values.shape, dim))


def multi_scatter(
    target: torch.Tensor, indices: torch.Tensor, values: torch.Tensor, dim=-2
):
    # backward of multi_gather
    return target.scatter(dim, multi_indexing(indices, values.shape, dim), values)


# ---- from easyvolcap/utils/base_utils.py ----
KT = TypeVar("KT")  # key type


VT = TypeVar("VT")  # value type


def type_to_torch_dtype(type):
    import numpy as np
    import torch

    if not hasattr(type_to_torch_dtype, "dtype_map"):
        type_to_torch_dtype.dtype_map = {
            int: torch.int64,
            float: torch.float32,
            bool: torch.bool,
            complex: torch.complex64,
            np.dtype("float32"): torch.float32,
            np.dtype("float64"): torch.float64,
            np.dtype("int32"): torch.int32,
            np.dtype("int64"): torch.int64,
            np.dtype("uint8"): torch.uint8,
            np.dtype("bool"): torch.bool,
            np.float32: torch.float32,
            np.float64: torch.float64,
            np.int32: torch.int32,
            np.int64: torch.int64,
            np.uint8: torch.uint8,
            # np.bool: torch.bool,
        }
    return type_to_torch_dtype.dtype_map[type]


def return_dotdict(func: Callable):
    def inner(*args, **kwargs):
        return dotdict(func(*args, **kwargs))

    return inner


class dotdict(dict, Dict[KT, VT]):
    """
    This is the default data passing object used throughout the codebase
    Main function: dot access for dict values & dict like merging and updates

    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = make_dotdict() or d = make_dotdict{'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """

    def update(self, dct: Dict = None, **kwargs):  # noqa: C901
        dct = copy(
            dct
        )  # avoid modifying the original dict, use super's copy to avoid recursion

        # Handle different arguments
        if dct is None:
            dct = kwargs
        elif isinstance(dct, Mapping):
            dct.update(kwargs)
        else:
            super().update(dct, **kwargs)
            return

        # Recursive updates
        for k, v in dct.items():
            if k in self:
                # Handle type conversions
                target_type = type(self[k])
                if not isinstance(v, target_type):
                    # NOTE: bool('False') will be True
                    if target_type == bool and isinstance(v, str):
                        dct[k] = v == "True"
                    else:
                        import numpy as np

                        # Lazy imports
                        import torch

                        if isinstance(v, torch.Tensor) and issubclass(
                            target_type, np.ndarray
                        ):
                            dct[k] = v
                        elif isinstance(v, torch.Tensor) and not issubclass(
                            target_type, torch.Tensor
                        ):
                            dct[k] = v.type(type_to_torch_dtype(target_type))
                        elif isinstance(v, np.ndarray) and not issubclass(
                            target_type, np.ndarray
                        ):
                            dct[k] = v.astype(target_type)
                        else:
                            dct[k] = target_type(v)

                if isinstance(v, dict):
                    self[k].update(v)  # recursion from here
                else:
                    self[k] = v
            else:
                if isinstance(v, dict):
                    self[k] = dotdict(v)  # recursion?
                elif isinstance(v, list):
                    self[k] = [dotdict(x) if isinstance(x, dict) else x for x in v]
                else:
                    self[k] = v
        return self

    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    copy = return_dotdict(dict.copy)
    fromkeys = return_dotdict(dict.fromkeys)

    # def __hash__(self):
    #     # return hash(''.join([str(self.values().__hash__())]))
    #     return super(dotdict, self).__hash__()

    # def __init__(self, *args, **kwargs):
    #     super(dotdict, self).__init__(*args, **kwargs)

    """
    Uncomment following lines and
    comment out __getattr__ = dict.__getitem__ to get feature:

    returns empty numpy array for undefined keys, so that you can easily copy things around
    TODO: potential caveat, harder to trace where this is set to np.array([], dtype=np.float32)
    """

    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError as e:
            raise AttributeError(e)

    # MARK: Might encounter exception in newer version of pytorch
    # Traceback (most recent call last):
    #   File "/home/xuzhen/miniconda3/envs/torch/lib/python3.9/multiprocessing/queues.py", line 245, in _feed
    #     obj = _ForkingPickler.dumps(obj)
    #   File "/home/xuzhen/miniconda3/envs/torch/lib/python3.9/multiprocessing/reduction.py", line 51, in dumps
    #     cls(buf, protocol).dump(obj)
    # KeyError: '__getstate__'
    # MARK: Because you allow your __getattr__() implementation to raise the wrong kind of exception.
    # FIXME: not working typing hinting code
    __getattr__: Callable[..., "torch.Tensor"] = __getitem__  # type: ignore # overidden dict.__getitem__
    __getattribute__: Callable[..., "torch.Tensor"]  # type: ignore
    # __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    # TODO: better ways to programmically define these special variables?

    @property
    def meta(self) -> dotdict:
        # Special variable used for storing cpu tensor in batch
        if "meta" not in self:
            self.meta = dotdict()
        return self.__getitem__("meta")

    @meta.setter
    def meta(self, meta):
        self.__setitem__("meta", meta)

    @property
    def output(self) -> dotdict:  # late annotation needed for this
        # Special entry for storing output tensor in batch
        if "output" not in self:
            self.output = dotdict()
        return self.__getitem__("output")

    @output.setter
    def output(self, output):
        self.__setitem__("output", output)

    @property
    def persistent(self) -> dotdict:  # late annotation needed for this
        # Special entry for storing persistent tensor in batch
        if "persistent" not in self:
            self.persistent = dotdict()
        return self.__getitem__("persistent")

    @persistent.setter
    def persistent(self, persistent):
        self.__setitem__("persistent", persistent)

    @property
    def type(self) -> str:  # late annotation needed for this
        # Special entry for type based construction system
        return self.__getitem__("type")

    @type.setter
    def type(self, type):
        self.__setitem__("type", type)

    def to_dict(self):
        out = {}
        for k, v in self.items():
            if isinstance(v, dotdict):
                v = v.to_dict()  # recursion point
            out[k] = v
        return out


# ---- from easyvolcap/utils/math_utils.py ----
@torch.jit.script
def affine_inverse(A: torch.Tensor):
    R = A[..., :3, :3]  # ..., 3, 3
    T = A[..., :3, 3:]  # ..., 3, 1
    P = A[..., 3:, :]  # ..., 1, 4
    return torch.cat([torch.cat([R.mT, -R.mT @ T], dim=-1), P], dim=-2)


# ---- from easyvolcap/utils/quat_utils.py ----
"""
Handles 4D rotation and two-sided quaternions along with the eigen decomposition of the 4D covariance matrix.
This essentially serves as the backward function of the compute_cov_4d routine
"""


@torch.jit.script
def angle_axis_to_quaternion(angle_axis: torch.Tensor) -> torch.Tensor:
    """
    Convert an angle axis to a quaternion.
    Args:
        angle_axis (Tensor): Tensor of Nx3 representing the rotations
    Returns:
        quaternions (Tensor): Tensor of Nx4 representing the quaternions
    """
    angle = torch.norm(angle_axis, p=2, dim=-1, keepdim=True)  # N, 1
    half_angle = 0.5 * angle
    eps: float = 1e-6
    small_angle = angle.abs() < eps
    sin_half_angle = torch.sin(half_angle)
    cos_half_angle = torch.cos(half_angle)
    sin_half_angle = torch.where(
        small_angle, half_angle - 0.5 * half_angle ** 3, sin_half_angle
    )
    cos_half_angle = torch.where(small_angle, 1.0 - 0.5 * half_angle ** 2, cos_half_angle)
    axis_normalized = angle_axis / (torch.norm(angle_axis, dim=-1, keepdim=True) + eps)
    quaternions = torch.cat(
        [cos_half_angle, sin_half_angle * axis_normalized], dim=-1
    )
    return quaternions


@torch.jit.script
def qmul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    Element-wise implementation: avoids bmm + N*4*4 intermediate tensor.
    """
    q0 = q[..., 0]
    q1 = q[..., 1]
    q2 = q[..., 2]
    q3 = q[..., 3]
    r0 = r[..., 0]
    r1 = r[..., 1]
    r2 = r[..., 2]
    r3 = r[..., 3]

    w = r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3
    x = r0 * q1 + r1 * q0 - r2 * q3 + r3 * q2
    y = r0 * q2 + r1 * q3 + r2 * q0 - r3 * q1
    z = r0 * q3 - r1 * q2 + r2 * q1 + r3 * q0
    return torch.stack((w, x, y, z), dim=-1)


# ---- from easyvolcap/utils/flow_utils.py ----
def get_color_wheel(device: torch.device) -> torch.Tensor:
    """
    Generates the color wheel.
    :param device: (torch.device) Device to be used
    :return: (torch.Tensor) Color wheel tensor of the shape [55, 3]
    """
    # Set constants
    RY: int = 15
    YG: int = 6
    GC: int = 4
    CB: int = 11
    BM: int = 13
    MR: int = 6
    # Init color wheel
    color_wheel: torch.Tensor = torch.zeros(
        (RY + YG + GC + CB + BM + MR, 3), dtype=torch.float32
    )
    # Init counter
    counter: int = 0
    # RY
    color_wheel[0:RY, 0] = 255
    color_wheel[0:RY, 1] = torch.floor(255 * torch.arange(0, RY) / RY)
    counter: int = counter + RY
    # YG
    color_wheel[counter : counter + YG, 0] = 255 - torch.floor(
        255 * torch.arange(0, YG) / YG
    )
    color_wheel[counter : counter + YG, 1] = 255
    counter: int = counter + YG
    # GC
    color_wheel[counter : counter + GC, 1] = 255
    color_wheel[counter : counter + GC, 2] = torch.floor(255 * torch.arange(0, GC) / GC)
    counter: int = counter + GC
    # CB
    color_wheel[counter : counter + CB, 1] = 255 - torch.floor(
        255 * torch.arange(CB) / CB
    )
    color_wheel[counter : counter + CB, 2] = 255
    counter: int = counter + CB
    # BM
    color_wheel[counter : counter + BM, 2] = 255
    color_wheel[counter : counter + BM, 0] = torch.floor(255 * torch.arange(0, BM) / BM)
    counter: int = counter + BM
    # MR
    color_wheel[counter : counter + MR, 2] = 255 - torch.floor(
        255 * torch.arange(MR) / MR
    )
    color_wheel[counter : counter + MR, 0] = 255
    # To device
    color_wheel: torch.Tensor = color_wheel.to(device)
    return color_wheel


def _flow_hw_to_color(
    flow_vertical: torch.Tensor,
    flow_horizontal: torch.Tensor,
    color_wheel: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Private function applies the flow color wheel to flow components (vertical and horizontal).
    :param flow_vertical: (torch.Tensor) Vertical flow of the shape [height, width]
    :param flow_horizontal: (torch.Tensor) Horizontal flow of the shape [height, width]
    :param color_wheel: (torch.Tensor) Color wheel tensor of the shape [55, 3]
    :param: device: (torch.device) Device to be used
    :return: (torch.Tensor) Visualized flow of the shape [3, height, width]
    """
    # Get shapes
    _, height, width = flow_vertical.shape
    # Init flow image
    flow_image: torch.Tensor = torch.zeros(
        3, height, width, dtype=torch.float32, device=device
    )
    # Get number of colors
    number_of_colors: int = color_wheel.shape[0]
    # Compute norm, angle and factors
    flow_norm: torch.Tensor = (flow_vertical**2 + flow_horizontal**2).sqrt()
    angle: torch.Tensor = torch.atan2(-flow_vertical, -flow_horizontal) / PI
    fk: torch.Tensor = (angle + 1.0) / 2.0 * (number_of_colors - 1.0)
    k0: torch.Tensor = torch.floor(fk).long()
    k1: torch.Tensor = k0 + 1
    k1[k1 == number_of_colors] = 0
    f: torch.Tensor = fk - k0
    # Iterate over color components
    for index in range(color_wheel.shape[1]):
        # Get component of all colors
        tmp: torch.Tensor = color_wheel[:, index]
        # Get colors
        color_0: torch.Tensor = tmp[k0] / 255.0
        color_1: torch.Tensor = tmp[k1] / 255.0
        # Compute color
        color: torch.Tensor = (1.0 - f) * color_0 + f * color_1
        # Get color index
        color_index: torch.Tensor = flow_norm <= 1
        # Set color saturation
        color[color_index] = 1 - flow_norm[color_index] * (1.0 - color[color_index])
        color[~color_index] = color[~color_index] * 0.75
        # Set color in image
        flow_image[index] = torch.floor(255 * color)
    return flow_image


def flow_to_color(
    flow: torch.Tensor,
    clip_flow: Optional[Union[float, torch.Tensor]] = None,
    normalize_over_video: bool = False,
    flow_min_max_norm: torch.Tensor | None = None,
    return_flow_max_norm: bool = False,
) -> torch.Tensor:
    """
    Function converts a given optical flow map into the classical color schema.
    :param flow: (torch.Tensor) Optical flow tensor of the shape [batch size (optional), 2, height, width].
    :param clip_flow: (Optional[Union[float, torch.Tensor]]) Max value of flow values for clipping (default None).
    :param normalize_over_video: (bool) If true scale is normalized over the whole video (batch).
    :return: (torch.Tensor) Flow visualization (float tensor) with the shape [batch size (if used), 3, height, width].
    """
    # Check parameter types
    assert torch.is_tensor(
        flow
    ), "Given flow map must be a torch.Tensor, {} given".format(type(flow))
    assert (
        torch.is_tensor(clip_flow) or isinstance(clip_flow, float) or clip_flow is None
    ), "Given clip_flow parameter must be a float, a torch.Tensor, or None, {} given".format(
        type(clip_flow)
    )
    # Check shapes
    assert flow.ndimension() in [
        3,
        4,
    ], "Given flow must be a 3D or 4D tensor, given tensor shape {}.".format(flow.shape)
    if torch.is_tensor(clip_flow):
        assert (
            clip_flow.ndimension() == 0
        ), "Given clip_flow tensor must be a scalar, given tensor shape {}.".format(
            clip_flow.shape
        )
    # Manage batch dimension
    batch_dimension: bool = True
    if flow.ndimension() == 3:
        flow = flow[None]
        batch_dimension: bool = False
    # Save shape
    batch_size, _, height, width = flow.shape
    # Check flow dimension
    assert (
        flow.shape[1] == 2
    ), "Flow dimension must have the shape 2 but tensor with {} given".format(
        flow.shape[1]
    )
    # Save device
    device: torch.device = flow.device
    # Clip flow if utilized
    if clip_flow is not None:
        flow = flow.clip(min=-clip_flow, max=clip_flow)
    # Get horizontal and vertical flow
    flow_vertical: torch.Tensor = flow[:, 0:1]
    flow_horizontal: torch.Tensor = flow[:, 1:2]
    # Get max norm of flow
    flow_max_norm: torch.Tensor = (
        (flow_vertical**2 + flow_horizontal**2)
        .sqrt()
        .view(batch_size, -1)
        .max(dim=-1)[0]
    )
    flow_max_norm: torch.Tensor = flow_max_norm.view(batch_size, 1, 1, 1)
    if normalize_over_video:
        flow_max_norm: torch.Tensor = flow_max_norm.max(dim=0, keepdim=True)[0]
    if flow_min_max_norm is not None:
        flow_max_norm = flow_max_norm.clip(min=flow_min_max_norm)
    # Normalize flow
    flow_vertical: torch.Tensor = flow_vertical / (flow_max_norm + 1e-05)
    flow_horizontal: torch.Tensor = flow_horizontal / (flow_max_norm + 1e-05)
    # Get color wheel
    color_wheel: torch.Tensor = get_color_wheel(device=device)
    # Init flow image
    flow_image = torch.zeros(batch_size, 3, height, width, device=device)
    # Iterate over batch dimension
    for index in range(batch_size):
        # if index == 96:
        #     breakpoint()
        flow_image[index] = _flow_hw_to_color(
            flow_vertical=flow_vertical[index],
            flow_horizontal=flow_horizontal[index],
            color_wheel=color_wheel,
            device=device,
        )
    if not return_flow_max_norm:
        return flow_image if batch_dimension else flow_image[0]
    else:
        return flow_image if batch_dimension else flow_image[0], flow_max_norm
