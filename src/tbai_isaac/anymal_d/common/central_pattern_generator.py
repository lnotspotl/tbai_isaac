#!/usr/bin/env python3

import torch


class CentralPatternGenerator:
    def __init__(self, period, initial_offsets, n_envs, device="cpu"):
        self.period = period
        self.initial_offsets = initial_offsets.to(device)
        self.device = device
        self.n_envs = n_envs
        self.time = torch.zeros((self.n_envs,)).to(device)

    def step(self, dt):
        self.time += dt

    def get_observation(self, phases=None):
        if phases is None:
            phases = self.compute_phases()
        return torch.cat([phases.cos(), phases.sin()], dim=-1)

    def compute_phases(self, phase_offsets=None):
        leg_times = self.time.unsqueeze(1) + self.initial_offsets.view(-1, 4)
        # leg_phases = 2 * torch.pi * torch.remainder(leg_times,self.period) / self.period    # <-- alternative
        self.phases = 2 * torch.pi * torch.frac(leg_times / self.period)

        if phase_offsets is not None:
            self.phases = (self.phases + phase_offsets) % (2 * torch.pi)

        return self.phases

    def reset(self, env_idxs=None):
        if env_idxs is None:
            env_idxs = slice(0, self.n_envs)
        self.time[env_idxs] = 0.0

    def leg_heights(self, phase_offsets=None, phases=None):
        if phases is None:
            phases = self.compute_phases(phase_offsets)
        heights = torch.zeros_like(phases, device=self.device)

        # swing - going up
        swing_up_indeces = phases <= torch.pi / 2
        time_up = phases[swing_up_indeces] * (2 / torch.pi)
        heights[swing_up_indeces] = 0.2 * (-2 * torch.pow(time_up, 3) + 3 * torch.pow(time_up, 2))

        # swing - going down
        swing_down_indeces = torch.logical_and((phases <= torch.pi), torch.logical_not(swing_up_indeces))
        time_down = phases[swing_down_indeces] * (2 / torch.pi) - 1.0
        heights[swing_down_indeces] = 0.2 * (2 * torch.pow(time_down, 3) - 3 * torch.pow(time_down, 2) + 1.0)

        return heights
