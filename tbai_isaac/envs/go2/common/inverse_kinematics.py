#!/usr/bin/env python3

import torch


class Go2InverseKinematics:
    def __init__(self, device="cpu"):
        height = 0.263
        self.device = device

        self.d2 = 0.142 - 0.0465
        self.a3 = 0.213
        self.a4 = 0.426 - 0.213

        self.default_positions = torch.Tensor(
            [
                0.025,
                0.096,
                -height,  # LF
                -0.025,
                0.096,
                -height,  # LH
                0.025,
                -0.096,
                -height,  # RF
                -0.025,
                -0.096,
                -height,  # RH
            ]
        ).to(self.device)

    def compute_ik(self, heights):
        n_envs = heights.shape[0]
        positions = self.default_positions.repeat(n_envs, 1)
        positions[:, [2, 5, 8, 11]] += heights
        return self._ik_vectorized(positions)

    def _ik_vectorized(self, positions):
        n_envs = positions.shape[0]

        d2_t = torch.tensor([self.d2], device=self.device) # (1,)
        d2_ts = torch.tensor([1.0, 1.0, -1.0, -1.0], device=self.device) * d2_t # (4,), LF, LH, RF, RH
        a3_t = torch.tensor([self.a3], device=self.device)
        a4_t = torch.tensor([self.a4], device=self.device)

        x_indices = [0, 3, 6, 9]
        y_indices = [1, 4, 7, 10]
        z_indices = [2, 5, 8, 11]
        yz_indeces = [1, 2, 4, 5, 7, 8, 10, 11]

        Es = torch.pow(positions[:, yz_indeces].view(n_envs, 4, -1), 2).sum(dim=2) - d2_ts.pow(2).unsqueeze(0)
        Es_sqrt = Es.sqrt()

        theta1s = torch.atan2(Es_sqrt, d2_ts) + torch.atan2(positions[:, z_indices], positions[:, y_indices]) # (n_envs, 4)

        Ds = (Es + torch.pow(positions[:, x_indices], 2) - a3_t.pow(2) - a4_t.pow(2)) / (2 * a3_t * a4_t)
        Ds[Ds > 1.0] = 1.0
        Ds[Ds < -1.0] = -1.0
        theta4s = -torch.atan2(torch.sqrt(1 - Ds.pow(2)), Ds) # (n_envs, 4)

        theta3s = torch.atan2(-positions[:, x_indices], Es_sqrt) - torch.atan2(
            a4_t * torch.sin(theta4s), a3_t + a4_t * torch.cos(theta4s)
        ) # (n_envs, 4)

        # The joint ordering now is 
        # LF_HAA(0), LH_HAA(1), RF_HAA(2), RH_HAA(3),
        # LF_HFE(4), LH_HFE(5), RF_HFE(6), RH_HFE(7),
        # LF_KFE(8), LH_KFE(9), RF_KFE(10), RH_KFE(11)
        joint_angles = torch.cat([theta1s, theta3s, theta4s], dim=1)


        # Nevertheless, IsaaGym expects the following ordering:
        # LF_HAA(0), LF_HFE(1), LF_KFE(2),
        # LH_HAA(3), LH_HFE(4), LH_KFE(5),
        # RF_HAA(6), RF_HFE(7), RF_KFE(8),
        # RH_HAA(9), RH_HFE(10), RH_KFE(11)

        # So we need to reorder the joint angles
        joint_angles = joint_angles[:, [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]] # (n_envs, 12)

        return joint_angles
