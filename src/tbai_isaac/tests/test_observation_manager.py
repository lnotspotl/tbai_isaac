#!/usr/bin/env python3

import pytest
from tbai_isaac.common.observation import Observation, ObservationManager


class DummyObservation1(Observation):
    def __init__(self, env, scale):
        super().__init__(env, scale)

    @property
    def size(self):
        return 1

    def compute(self):
        raise NotImplementedError


class DummyObservation2(Observation):
    def __init__(self, env, scale):
        super().__init__(env, scale)

    @property
    def size(self):
        return 2

    def compute(self):
        raise NotImplementedError


def test_get_slice():
    env = None
    manager = ObservationManager(env)
    manager.add_observation("obs1", DummyObservation1(env, 1))
    manager.add_observation("obs2", DummyObservation2(env, 1))
    manager.add_observation("obs3", DummyObservation2(env, 1))

    assert manager.get_slice("obs1") == slice(0, 1)
    assert manager.get_slice("obs2") == slice(1, 3)
    assert manager.get_slice("obs3") == slice(3, 5)
    assert manager.get_slice(["obs1", "obs2"]) == slice(0, 3)
    assert manager.get_slice(["obs2", "obs3"]) == slice(1, 5)

    # # Raises ValueError if observation not found
    with pytest.raises(ValueError):
        manager.get_slice("obs4")

    # Raises ValueError if observations are not ordered
    with pytest.raises(ValueError):
        manager.get_slice(["obs2", "obs1"])

    # Raises ValueError if slice is not contiguous
    with pytest.raises(ValueError):
        manager.get_slice(["obs1", "obs3"])


def test_size():
    env = None
    manager = ObservationManager(env)

    # Empty observation manager
    assert manager.size == 0

    # Single observation
    manager.add_observation("obs1", DummyObservation1(env, 1))
    assert manager.size == 1

    # Multiple observations
    manager.add_observation("obs2", DummyObservation2(env, 1))
    manager.add_observation("obs3", DummyObservation2(env, 1))
    assert manager.size == 5


def test_get_observation():
    env = None
    manager = ObservationManager(env)

    # Raises ValueError if observation not found
    with pytest.raises(ValueError):
        manager.get_observation("obs1")

    # Does not raise after adding observation
    manager.add_observation("obs1", DummyObservation1(env, 1))
    try:
        manager.get_observation("obs1")
    except ValueError:
        assert False, "get_observation() raised ValueError unexpectedly"
