#!/usr/bin/env python3

import torch


class AnymalInverseKinematics:
    def __init__(self, device="cpu"):
        height = 0.573
        self.device = device

        self.d2 = 0.20875
        self.a3 = 0.285
        self.a4 = 0.404999816790082

        self.default_positions = torch.Tensor(
            [
                0.00790786,
                0.05720384,
                -height,  # LF
                -0.00790786,
                0.05720384,
                -height,  # LH
                0.00790786,
                -0.05720384,
                -height,  # RF
                -0.00790786,
                -0.05720384,
                -height,  #  RH
            ]
        ).to(self.device)

    def compute_ik(self, heights):
        n_envs = heights.shape[0]
        positions = self.default_positions.repeat(n_envs, 1)
        positions[:, [2, 5, 8, 11]] += heights
        return self._ik_vectorized(positions)

    def _ik_vectorized(self, positions):
        n_envs = positions.shape[0]

        d2_t = torch.tensor([self.d2], device=self.device)
        d2_ts = torch.tensor([1.0, 1.0, -1.0, -1.0], device=self.device) * d2_t
        a3_t = torch.tensor([self.a3], device=self.device)
        a4_t = torch.tensor([self.a4], device=self.device)

        theta4_multipliets = torch.tensor([1.0, -1.0, 1.0, -1.0], device=self.device)

        x_indices = [0, 3, 6, 9]
        y_indices = [1, 4, 7, 10]
        z_indices = [2, 5, 8, 11]
        yz_indeces = [1, 2, 4, 5, 7, 8, 10, 11]

        Es = torch.pow(positions[:, yz_indeces].view(n_envs, 4, -1), 2).sum(dim=2) - d2_ts.pow(2).unsqueeze(0)
        Es_sqrt = Es.sqrt()

        theta1s = torch.atan2(Es_sqrt, d2_ts) + torch.atan2(positions[:, z_indices], positions[:, y_indices])

        Ds = (Es + torch.pow(positions[:, x_indices], 2) - a3_t.pow(2) - a4_t.pow(2)) / (2 * a3_t * a4_t)
        Ds[Ds > 1.0] = 1.0
        Ds[Ds < -1.0] = -1.0
        theta4_offset = torch.tensor([0.2484], device=self.device)
        theta4s = -torch.atan2(torch.sqrt(1 - Ds.pow(2)), Ds)
        theta4s_final = theta4s + theta4_offset

        theta4s *= theta4_multipliets
        theta4s_final *= theta4_multipliets

        theta3s = torch.atan2(-positions[:, x_indices], Es_sqrt) - torch.atan2(
            a4_t * torch.sin(theta4s), a3_t + a4_t * torch.cos(theta4s)
        )

        joint_angles = torch.cat([theta1s, theta3s, theta4s_final], dim=1)[
            :, [4 * i + j for j in range(4) for i in range(3)]
        ]
        return joint_angles
