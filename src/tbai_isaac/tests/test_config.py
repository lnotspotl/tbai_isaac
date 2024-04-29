#!/usr/bin/env python3

import os
import time

import pytest
from tbai_isaac.common.config import YamlConfig
from tbai_isaac.common.config import load_config, store_config, select

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


def test_config():
    config = YamlConfig(TEST_CONFIG)
    assert config["model/name"] == "test_model"
    assert config("model/version", int) == 1
    assert config("model/version", type=str) == "1"
    assert config["model/version", str] == "1"
    assert config["model/version", int] == 1
    assert config("algorithm") == "PPO"
    assert config.read_key("algorithm") == "PPO"
    assert config("random_values") == [1, 2, 3]
    assert config("random_values2", str) == "[4, 5, 6]"
    assert config("model/items/a") == "item1"


def test_delim():
    config = YamlConfig(TEST_CONFIG, delim=".")
    assert config["model.name"] == "test_model"
    assert config("model.version", type=str) == "1"
    assert config("algorithm") == "PPO"

    config = YamlConfig(TEST_CONFIG, delim="#")
    assert config["model#name"] == "test_model"
    assert config("model#version", type=str) == "1"
    assert config("algorithm") == "PPO"


def test_as_dict():
    config = YamlConfig(TEST_CONFIG)
    assert isinstance(config.as_dict(), dict)
    assert config.as_dict()["model"]["name"] == "test_model"
    assert config.as_dict()["model"]["version"] == 1
    assert config.as_dict()["random_values"] == [1, 2, 3]


def test_as_dataclass():
    config = YamlConfig(TEST_CONFIG).as_dataclass()
    assert config.model.name == "test_model"
    assert config.model.version == 1
    assert config.random_values == [1, 2, 3]
    assert config.random_values2 == [4, 5, 6]
    assert config.algorithm == "PPO"
    assert config.model.items.a == "item1"


def test_set_key():
    config = YamlConfig(TEST_CONFIG)

    # Set item using `set_key`
    config.set_key("model/items/b", "item2")
    assert config("model/items/b") == "item2"

    # Set item using `[]` operator
    config["model/items/c"] = "item3"
    assert config("model/items/c") == "item3"

    # Set item using `[]` operator with nested keys
    config["model/items/d/e/f/g"] = "item4"
    assert config["model/items/d/e/f/g"] == "item4"

    # Overwrite existing item
    config["model/items/d/e/f/g"] = "item5"
    assert config["model/items/d/e/f/g"] == "item5"


def test_raises():
    config = YamlConfig(TEST_CONFIG)
    config["model/items/d/e/f"] = "3"

    # Raises - "model/items/d/e" not a leaf node
    with pytest.raises(ValueError):
        config.set_key("model/items/d/e", "4")

    # Raises - "model" not a leaf node
    with pytest.raises(ValueError):
        config["model"] = "item"

    # Does not raise, "model/items/d/e/f" is a leaf node
    config["model/items/d/e/f"] = 3
    assert config["model/items/d/e/f"] == 3


def test_root():
    config = YamlConfig(TEST_CONFIG)
    assert config.as_dict("model")["name"] == "test_model"
    assert config.as_dataclass("model").name == "test_model"

    # Invalid - root is a leaf node
    with pytest.raises(ValueError):
        config.as_dict("model/version")


def test_contains():
    config = YamlConfig(TEST_CONFIG)
    assert "model" in config
    assert "model/name" in config
    assert "model/random" not in config
    assert "" in config
    assert "a/b/c/d/e/f/g" not in config


def test_store():
    path = f"{TEST_CONFIG}_copy.yaml"
    config = YamlConfig(TEST_CONFIG)
    config.store(path)
    config2 = YamlConfig(path)
    assert config.as_dict() == config2.as_dict()
    os.remove(path)


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
