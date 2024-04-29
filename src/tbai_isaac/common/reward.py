#!/usr/bin/env python3

from abc import ABC, abstractmethod
from collections import OrderedDict

import torch

from tbai_isaac.common.base_env import BaseEnv


class Reward(ABC):
    def __init__(self, env: BaseEnv, scale: float) -> None:
        self.env = env
        self.scale = scale

    @abstractmethod
    def compute(self) -> torch.Tensor:
        """Compute reward tensor with shape (num_envs, )"""
        raise NotImplementedError


class RewardManager:
    def __init__(self, env: BaseEnv) -> None:
        self.env = env
        self.rewards = OrderedDict()

    def add_reward(self, name: str, reward: Reward) -> None:
        if name in self.rewards:
            raise ValueError(f"Reward {name} already exists")
        self.rewards[name] = reward

    def get_reward(self, name: str) -> Reward:
        if name not in self.rewards:
            raise ValueError(f"Reward {name} not found")
        return self.rewards[name]

    def compute(self) -> torch.Tensor:
        return sum(reward.compute() for reward in self.rewards.values())
