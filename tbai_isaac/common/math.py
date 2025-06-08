from typing import Tuple

import isaacgym  # noqa: F401  - isaacgym does not stand when pytorch is imported prior to it
import torch
from isaacgym.torch_utils import normalize, quat_apply


def quat_apply_yaw(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


def quat_apply_yaw_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat_yaw = quat.clone().view(-1, 4) * torch.tensor([-1.0, -1.0, -1.0, 1.0], device=quat.device).view(1, 4)
    quat_yaw[:, :2] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    angles %= 2 * torch.pi
    angles -= 2 * torch.pi * (angles > torch.pi)
    return angles


def torch_rand_sqrt_float(lower: float, upper: float, shape: Tuple[int, int], device: str) -> torch.Tensor:
    r = 2 * torch.rand(*shape, device=device) - 1
    r = torch.where(r < 0.0, -torch.sqrt(-r), torch.sqrt(r))
    r = (r + 1.0) / 2.0
    return (upper - lower) * r + lower
