#!/usr/bin/env python3

import os
import time

import pytest
from tbai_isaac.common.config import load_config

TEST_CONFIG = f"test_config_{time.time()}.yaml"


@pytest.fixture(scope="session", autouse=True)
def setup_config(request):
    config = """
    model:
        name: "test_model"
        version: 1
        items:
            a: "item1"

    algorithm: "PPO"
    random_values: [1, 2, 3]
    random_values2: [4, 5, 6]
    """
    with open(TEST_CONFIG, "w") as f:
        f.write(config)

    # Remove config once all tests are done
    request.addfinalizer(lambda: os.remove(TEST_CONFIG))


def test_config_new():
    cfg = load_config(TEST_CONFIG)
    assert cfg.model.name == "test_model"
    assert cfg.model.version == 1


def test_config_new2():
    cfg = load_config(TEST_CONFIG, "model")
    assert cfg.name == "test_model"


def test_config_new3():
    with pytest.raises(AssertionError):
        load_config(TEST_CONFIG + ".json")
