#!/usr/bin/env python3

import isaacgym  # noqa: F401
import torch
from scipy.spatial.transform import Rotation as R
from tbai_isaac.common import math as tbai_math


def test_quat_apply_yaw_yaw():
    # scipy uses the [x, y, z, w] format, just like isaacgym
    quat = R.from_euler("z", [90], degrees=True).as_quat()
    quat = torch.Tensor(quat).reshape(1, -1)
    vec = torch.Tensor([1, 0, 0]).reshape(1, -1)
    out = tbai_math.quat_apply_yaw(quat, vec).squeeze()
    assert torch.allclose(out, torch.Tensor([0, 1, 0]), atol=1e-6)


def test_quat_apply_yaw_roll():
    # scipy uses the [x, y, z, w] format, just like isaacgym
    quat = R.from_euler("x", [54], degrees=True).as_quat()
    quat = torch.Tensor(quat).reshape(1, -1)
    vec = torch.Tensor([1, 0, 0]).reshape(1, -1)
    out = tbai_math.quat_apply_yaw(quat, vec).squeeze()
    assert torch.allclose(out, torch.Tensor([1, 0, 0]), atol=1e-6)


def test_quat_apply_yaw_inverse_yaw():
    # scipy uses the [x, y, z, w] format, just like isaacgym
    quat = R.from_euler("z", [90], degrees=True).as_quat()
    quat = torch.Tensor(quat).reshape(1, -1)
    vec = torch.Tensor([1, 0, 0]).reshape(1, -1)
    out = tbai_math.quat_apply_yaw_inverse(quat, vec).squeeze()
    assert torch.allclose(out, torch.Tensor([0, -1, 0]), atol=1e-6)


def test_quat_apply_yaw_inverse_roll():
    # scipy uses the [x, y, z, w] format, just like isaacgym
    quat = R.from_euler("x", [32], degrees=True).as_quat()
    quat = torch.Tensor(quat).reshape(1, -1)
    vec = torch.Tensor([1, 0, 0]).reshape(1, -1)
    out = tbai_math.quat_apply_yaw_inverse(quat, vec).squeeze()
    assert torch.allclose(out, torch.Tensor([1, 0, 0]), atol=1e-6)
