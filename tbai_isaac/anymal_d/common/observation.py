#!/usr/bin/env python3

import torch
from tbai_isaac.common.base_env import BaseEnv
from tbai_isaac.common.observation import Observation


class BodyLinearVelocityObservation(Observation):
    def __init__(self, env: BaseEnv, scale: float) -> None:
        super().__init__(env, scale)

    @property
    def size(self) -> int:
        return 3

    def compute(self):
        return self.env.base_lin_vel * self.scale


class BodyAngularVelocityObservation(Observation):
    def __init__(self, env: BaseEnv, scale: float) -> None:
        super().__init__(env, scale)

    @property
    def size(self) -> int:
        return 3

    def compute(self):
        return self.env.base_ang_vel * self.scale


class NormalizedGravityVectorObservation(Observation):
    def __init__(self, env: BaseEnv, scale: float) -> None:
        super().__init__(env, scale)

    @property
    def size(self) -> int:
        return 3

    def compute(self):
        return self.env.projected_gravity * self.scale


class JointPositionResidualsObservation(Observation):
    def __init__(self, env: BaseEnv, scale: float, default_positions: torch.Tensor) -> None:
        super().__init__(env, scale)

        assert default_positions.numel() == 12
        self.default_positions = default_positions

    @property
    def size(self) -> int:
        return 12

    def compute(self):
        return (self.env.dof_pos - self.default_positions.unsqueeze(0)) * self.scale


class JointVelicitiesObservation(Observation):
    def __init__(self, env: BaseEnv, scale: float) -> None:
        super().__init__(env, scale)

    @property
    def size(self) -> int:
        return 12

    def compute(self):
        return self.env.dof_vel * self.scale


class HeightmapObservation(Observation):
    def __init__(self, env: BaseEnv, scale: float) -> None:
        super().__init__(env, scale)

    @property
    def size(self) -> int:
        return 52 * 4

    def compute(self):
        heights = torch.clip(self.env.root_states[:, 2].unsqueeze(1) - 0.5 - self.env.measured_heights, -1, 1.0)
        return heights * self.scale
