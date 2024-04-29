#!/usr/bin/env python3

from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import yaml
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


class YamlConfig:
    def __init__(self, path: str, delim: str = "/") -> None:
        self._cfg = self._read_config(path)
        self._path = path
        self._delim = delim

    def as_dict(self, root: str = "") -> Dict[str, Any]:
        if self.is_leaf(root):
            raise ValueError(f"Root '{root}' is a leaf node in config: {self._path}")
        return self._find_root(root)

    def as_dataclass(self, root: str = "") -> Any:
        return self._as_dataclass_impl(self.as_dict(root))

    def _read_config(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _split(self, s: str) -> List[str]:
        return s.split(self._delim)

    def read_key(self, key: str, type: Optional[Callable] = None) -> Any:
        if key not in self:
            raise ValueError(f"Key '{key}' not found in config: {self._path}")
        node = self._cfg
        for item in self._split(key):
            node = node[item]
        if type is not None:
            node = type(node)
        return node

    def set_key(self, key: str, value: Any) -> None:
        node = self._cfg
        keys = self._split(key)
        for item in keys[:-1]:
            if item not in node:
                node[item] = dict()
            node = node[item]
        if keys[-1] in node and isinstance(node[keys[-1]], dict):
            raise ValueError(f"Key '{self._delim.join(keys)}' already exists in config: {self._path}")
        node[keys[-1]] = value

    def is_leaf(self, key: str) -> bool:
        if key not in self:
            raise ValueError(f"Key '{key}' not found in config: {self._path}")
        if len(key) == 0:
            return False
        node = self._cfg
        for item in self._split(key):
            node = node[item]
        return not isinstance(node, dict)

    def store(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(self._cfg, f)

    def __setitem__(self, key: str, value: Any) -> None:
        self.set_key(key, value)

    def __getitem__(self, key: Union[str, Tuple[str, Callable]]) -> Any:
        if isinstance(key, tuple):
            assert len(key) == 2, "key must be (key, type)"
            return self.read_key(*key)

        if isinstance(key, str):
            return self.read_key(key)

    def __call__(self, key: str, type: Optional[Any] = None) -> Any:
        return self.read_key(key, type)

    def __contains__(self, key: str) -> bool:
        if len(key) == 0:
            return True  # "" means root

        node = self._cfg
        for item in self._split(key):
            if item not in node:
                return False
            node = node[item]
        return True

    def _find_root(self, root: str) -> Dict[str, Any]:
        if len(root) == 0:
            return self._cfg
        if root not in self:
            raise ValueError(f"Root '{root}' not found in config: {self._path}")
        node = self._cfg
        for item in self._split(root):
            node = node[item]
        return node

    def _as_dataclass_impl(self, cfg: Dict[str, Any], name: Union[str, None] = None) -> Any:
        if name is None:
            name = "Config"
        cl = namedtuple(name.capitalize(), cfg.keys())
        vals = list()
        for key, value in cfg.items():
            if isinstance(value, dict):
                vals.append(self._as_dataclass_impl(value, key))
            else:
                vals.append(value)
        return cl(*vals)
