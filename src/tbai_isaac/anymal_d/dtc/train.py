#!/usr/bin/env python3

from ocs2_interface import get_interface
from isaacgym import gymapi, gymtorch, gymutil
import torch
from tbai_isaac.anymal_d.dtc.agent import AgentNetwork
from tbai_isaac.anymal_d.dtc.env import LeggedRobot
from tbai_isaac.common.config import load_config
from tbai_isaac.common.utils import parse_args
from tbai_isaac.ppo.coach import Coach

from tbai_isaac.common.utils import parse_args, set_seed, store_config


def train(args):
    config = load_config(args.config)

    # Set seed
    if "seed" not in config:
        seed = set_seed(args.seed)
        config["seed"] = seed
    else:
        seed = config["seed"]
        set_seed(seed)

    store_config(args, config)

    env = LeggedRobot(config, args.headless, ig_threads=8)

    actor_critic = AgentNetwork(config)

    coach = Coach(env, config.ppo, actor_critic, args.log_dir, writer_type=args.writer_type)

    coach.train(args.max_iterations, True)


if __name__ == "__main__":
    args = parse_args()
    train(args)
