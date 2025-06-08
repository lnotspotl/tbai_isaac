#!/usr/bin/env python3

import sys
from abc import abstractmethod
from typing import Any, Dict, Tuple, Union

import torch
from isaacgym import gymapi, gymutil


class BaseEnv:
    def __init__(
        self,
        num_obs: int,
        num_privileged_obs: int,
        num_actions: int,
        device: str = "cuda",
        num_envs: int = -1,
        max_episode_length: int = -1,
        headless: bool = False,
        sim_params: Dict[str, Any] = None,
        physics_engine: str = "physx",
        sim_device: str = "cuda",
    ):
        # Initialize parent class - VecEnv
        self.num_obs = num_obs
        self.num_privileged_obs = num_privileged_obs
        self.device = device
        self.num_envs = num_envs
        self.max_episode_length = max_episode_length

        self.num_actions = num_actions
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(sim_device)
        self.graphics_device_id = self.sim_device_id
        self.physics_engine = physics_engine

        if headless:
            self.graphics_device_id = -1
        self.sim_device = sim_device

        # Perform basic checks
        assert num_envs > 0, "Number of environments must be greater than 0"
        assert num_obs > 0, "Number of observations must be greater than 0"
        assert num_actions >= 0, "Number of actions must be greater than or equal to 0"
        assert "cuda" in device, "Only GPU is supported"
        assert "cuda" in sim_device, "Only GPU is supported"

        # Torch flags
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # Allocate buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), dtype=torch.float32, device=self.device)
        self.rew_buf = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.reset_buf = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
        self.episode_length_buf = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
        self.time_out_buf = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.privileged_obs_buf = torch.zeros(
            (self.num_envs, self.num_privileged_obs), dtype=torch.float32, device=self.device
        )

        # Remove buffers with zero size
        self.privileged_obs_buf = None if self.privileged_obs_buf.numel() == 0 else self.privileged_obs_buf
        # Get gym singleton
        self.gym = gymapi.acquire_gym()

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        self.enable_viewer_sync = True
        self.viewer = None
        self.headless = headless

        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        _, self.sim_device_id = gymutil.parse_device_str(self.sim_device)

    def get_observations(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return self.obs_buf

    def get_privileged_observations(self) -> Union[torch.Tensor, None]:
        return self.privileged_obs_buf

    @abstractmethod
    def reset_idx(self, idx):
        raise NotImplementedError

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device))
        return obs, privileged_obs

    @abstractmethod
    def step(self, actions):
        raise NotImplementedError

    @abstractmethod
    def create_sim(self):
        raise NotImplementedError

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)
