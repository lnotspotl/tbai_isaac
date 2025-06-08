#!/usr/bin/env python3

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Union

import isaacgym  # noqa: F401  - isaacgym does not stand when pytorch is imported prior to it
import numpy as np
import torch

from tbai_isaac.common.base_env import BaseEnv


class Observation(ABC):
    def __init__(self, env: BaseEnv, scale: float) -> None:
        self.env = env
        self.scale = scale

    @property
    @abstractmethod
    def size(self) -> int:
        """Return observation size"""
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> torch.Tensor:
        """Compute observation tensor with shape (num_envs, self.size)"""
        raise NotImplementedError


class ObservationManager:
    def __init__(self, env: BaseEnv) -> None:
        self.env = env
        self.observations = OrderedDict()

        self.obs = None

    def add_observation(self, name: str, observation: Observation) -> None:
        if name in self.observations:
            raise ValueError(f"Observation {name} already exists")
        self.observations[name] = observation

    def get_observation(self, name: str) -> Observation:
        if name not in self.observations:
            raise ValueError(f"Observation {name} not found")
        return self.observations[name]

    def get_slice(self, names: Union[List[str], str]) -> slice:
        if isinstance(names, str):
            names = [names]

        registered_names = list(self.observations.keys())

        # All requested names need to be registered
        for name in names:
            if name not in registered_names:
                raise ValueError(f"Observation {name} not found")

        # All requested names need to be ordered
        requested_indeces = [registered_names.index(name) for name in names]
        if requested_indeces != sorted(requested_indeces):
            raise ValueError("Requested observations are not ordered")

        # All requested names need to be part of a single slice
        for i in range(len(requested_indeces) - 1):
            if requested_indeces[i] + 1 != requested_indeces[i + 1]:
                raise ValueError("Requested observations contiguous! Only contiguous tensor slices are supported.")

        # Find start index and end index
        cummulative_indeces = np.cumsum([self.observations[name].size for name in registered_names])
        start_index = None
        for index, name in enumerate(registered_names):
            if name in names:
                start_index = cummulative_indeces[index] - self.observations[name].size
                break

        end_index = None
        for index, name in enumerate(registered_names):
            if name in names:
                end_index = cummulative_indeces[index]

        return slice(start_index, end_index)

    @property
    def size(self):
        return sum([observation.size for observation in self.observations.values()])

    def compute(self) -> None:
        self.obs = torch.cat([observation.compute() for observation in self.observations.values()], dim=1)

    def get(self) -> torch.Tensor:
        assert self.obs is not None, "You need to call compute first"
        return self.obs
