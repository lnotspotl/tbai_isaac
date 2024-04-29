#!/usr/bin/env python3

import torch


class ExteroceptiveNoiseGenerator:
    def __init__(self, points_per_leg, n_envs, env_max_steps, n_legs=4, device="cuda"):
        self.n_legs = n_legs
        self.n_envs = n_envs
        self.points_per_leg = points_per_leg
        self.env_max_steps = env_max_steps
        self.device = device

        self.ck = 0.0
        self.ck_step = 1 / 25000

        self.env_steps = torch.zeros(self.n_envs, dtype=torch.long)
        self.zs = torch.zeros(self.n_envs, 8, device=self.device)

        self.update_zs()

        self.w_x_stored = torch.zeros(self.n_envs, self.n_legs, dtype=torch.float, device=self.device)
        self.w_y_stored = torch.zeros(self.n_envs, self.n_legs, dtype=torch.float, device=self.device)
        self.w_z_stored = torch.zeros(self.n_envs, self.n_legs, dtype=torch.float, device=self.device)

    def update_curriculum(self):
        self.ck = min(self.ck + self.ck_step, 1.0)

    def reset(self, idxs):
        if idxs.numel() != 0:
            self.env_steps[idxs] = 0
            self.update_zs(idxs)
            self.sample_ws(idxs)

    def step(self):
        self.env_steps += 1
        update_idxs = (self.env_steps == self.env_max_steps // 2).nonzero().squeeze()
        if update_idxs.numel() != 0:
            self.update_zs(update_idxs)
        self.update_curriculum()

    def update_zs(self, idxs=None):
        if idxs is None:
            idxs = torch.arange(self.n_envs)
        N = idxs.numel()
        p = torch.rand(N)
        new_zs = torch.zeros(N, self.zs.shape[1], device=self.device)
        new_zs[p < 0.6] = self.zs_nominal
        new_zs[torch.logical_and(p >= 0.6, p < 0.9)] = self.zs_offset
        new_zs[p >= 0.9] = self.zs_noisy
        self.zs[idxs] = new_zs

    def sample_ws(self, idxs):
        self.w_x_stored[idxs] = self.w_xy(idxs)
        self.w_y_stored[idxs] = self.w_xy(idxs)
        self.w_z_stored[idxs] = self.w_z(idxs)

    def sample_noise(self, idxs=None):
        if idxs is None:
            idxs = torch.arange(self.n_envs)

        N = idxs.numel()

        # x noise
        eps_px = self.eps_pxy(idxs).view(N, self.n_legs, self.points_per_leg)
        eps_fx = self.eps_fxy(idxs)
        w_x = self.w_x_stored[idxs]

        # y noise
        eps_py = self.eps_pxy(idxs).view(N, self.n_legs, self.points_per_leg)  # (idxs, n_legs, points_per_leg)
        eps_fy = self.eps_fxy(idxs)  # (idxs, n_legs)
        w_y = self.w_y_stored[idxs]  # (idxs, n_legs)

        # z_noise
        eps_pz = self.eps_pz(idxs).view(N, self.n_legs, self.points_per_leg)
        eps_fz = self.eps_fz(idxs)
        w_z = self.w_z_stored[idxs]

        # outlier noise
        eps_outlier = self.eps_outlier(idxs).view(N, self.n_legs, self.points_per_leg)

        x_noise = (eps_px + eps_fx.unsqueeze(-1) + w_x.unsqueeze(-1)).view(N, -1).to(self.device)
        y_noise = (eps_py + eps_fy.unsqueeze(-1) + w_y.unsqueeze(-1)).view(N, -1).to(self.device)
        z_noise = (eps_pz + eps_fz.unsqueeze(-1) + w_z.unsqueeze(-1) + eps_outlier).view(N, -1).to(self.device)

        return x_noise, y_noise, z_noise

    ## TODO: update values for zs

    @property
    def zs_nominal(self):
        return torch.tensor([0.004, 0.005, 0.01, 0.04, 0.03, 0.05, 0.1, 0.1]).to(self.device) * 1.0

    @property
    def zs_offset(self):
        return torch.tensor([0.004, 0.005, 0.01, 0.1 * self.ck, 0.1 * self.ck, 0.02, 0.1, 0.1]).to(self.device) * 1.0

    @property
    def zs_noisy(self):
        return (
            torch.tensor(
                [0.004, 0.1 * self.ck, 0.1 * self.ck, 0.3 * self.ck, 0.3 * self.ck, 0.3 * self.ck, 0.1, 0.1]
            ).to(self.device)
            * 1.0
        )

    def eps_pxy(self, idxs):
        """Per-step xy noise for each point"""
        return torch.randn(size=(idxs.numel(), self.n_legs, self.points_per_leg), device=self.device) * self.zs[
            idxs, 0
        ].view(-1, 1, 1).to(self.device)

    def eps_pz(self, idxs):
        """Per-step z noise for each point"""
        return torch.randn(size=(idxs.numel(), self.n_legs, self.points_per_leg), device=self.device) * self.zs[
            idxs, 1
        ].view(-1, 1, 1).to(self.device)

    def eps_fxy(self, idxs):
        """Per-step xy noise for each foot"""
        return torch.randn(size=(idxs.numel(), self.n_legs), device=self.device) * self.zs[idxs, 2].view(-1, 1).to(
            self.device
        )

    def eps_fz(self, idxs):
        """Per-step z noise for each foot"""
        return torch.randn(size=(idxs.numel(), self.n_legs), device=self.device) * self.zs[idxs, 3].view(-1, 1).to(
            self.device
        )

    def eps_outlier(self, idxs):
        """Per-step outlier noise for each point"""
        ret = torch.randn(size=(idxs.numel(), self.n_legs, self.points_per_leg), device=self.device) * self.zs[
            idxs, 4
        ].view(-1, 1, 1).to(self.device)
        is_outlier = torch.rand_like(ret) <= self.zs[idxs, 5].view(-1, 1, 1).to(self.device)
        return ret * is_outlier

    def w_xy(self, idxs):
        """Per-episode xy noise for each foot"""
        return torch.randn(size=(idxs.numel(), self.n_legs), device=self.device) * self.zs[idxs, 6].view(-1, 1).to(
            self.device
        )

    def w_z(self, idxs):
        """Per-episode xy noise for each foot"""
        return torch.randn(size=(idxs.numel(), self.n_legs), device=self.device) * self.zs[idxs, 7].view(-1, 1).to(
            self.device
        )
