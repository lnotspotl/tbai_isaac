#!/usr/bin/env python3

import os
import random

import isaacgym  # noqa: F401
import numpy as np
import torch
from isaacgym import gymutil

import tbai_isaac.common.config as ac


def parse_args():
    custom_parameters = [
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {
            "name": "--num_envs",
            "type": int,
            "help": "Number of environments to create. Overrides config file if provided.",
        },
        {
            "name": "--max_iterations",
            "type": int,
            "help": "Maximum number of training iterations. Overrides config file if provided.",
            "default": 1500
        },
        {
            "name": "--seed",
            "type": int,
            "default": -1,
            "help": "Seed for random number generation. Default is -1, which will generate a random seed.",
        },
        {
            "name": "--log_dir",
            "type": str,
            "help": "Directory to save logs and models to.",
        },
        {
            "name": "--config",
            "type": str,
            "help": "Path to the config file.",
        },
        {
            "name": "--model",
            "type": str,
            "help": "Path to the model file.",
        },
        {
            "name": "--writer_type",
            "type": str,
            "default": "none",
            "options": ["wandb", "tensorboard", "none"],
            "help": "Writer type for logging.",
        }
    ]

    # parse arguments
    args = gymutil.parse_arguments(description="tbai parameters", custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def set_seed(seed):
    # Taken from: https://github.com/leggedrobotics/legged_gym/blob/20f7f92e89865084b958895d988513108b307e6c/legged_gym/utils/helpers.py#L67
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def store_config(args, config):
    log_dir = args.log_dir
    config_file = os.path.join(log_dir, "config.yaml")
    ac.store_config(config, config_file)

def create_dir(dir):
    os.makedirs(dir, exist_ok=True)