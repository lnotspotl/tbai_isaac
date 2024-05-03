#!/usr/bin/env python3

from omegaconf import OmegaConf


def load_config(config_path: str, prefix=None) -> OmegaConf:
    assert config_path.endswith(".yaml"), f"Invalid config file: {config_path} Only .yaml files are supported."
    conf = OmegaConf.load(config_path)
    if prefix is not None:
        conf = OmegaConf.select(conf, prefix)
    return conf


def store_config(config: OmegaConf, config_path: str) -> None:
    assert config_path.endswith(".yaml"), f"Invalid config file: {config_path} Only .yaml files are supported."
    OmegaConf.save(config, config_path)


def select(config: OmegaConf, prefix: str) -> OmegaConf:
    return OmegaConf.select(config, prefix)
