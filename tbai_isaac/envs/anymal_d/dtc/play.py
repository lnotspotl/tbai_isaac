#!/usr/bin/env python3

import os

from ocs2_interface import get_interface
from isaacgym import gymapi, gymtorch, gymutil
import torch
from tbai_isaac.envs.anymal_d.dtc.agent import AgentNetwork
from tbai_isaac.envs.anymal_d.dtc.env import LeggedRobot
from tbai_isaac.common.config import load_config
from tbai_isaac.common.utils import parse_args
from tbai_isaac.ppo.coach import Coach

from tbai_isaac.common.utils import parse_args, set_seed, store_config


def play(args):
    config = load_config(args.config)
    config.environment.env.num_envs = min(config.environment.env.num_envs, 2)
    config.environment.terrain.num_rows = 10
    config.environment.terrain.num_cols = 20
    config.environment.terrain.curriculum = False
    config.environment.noise.add_noise = False
    config.environment.domain_randomization.randomize_friction = True
    config.environment.domain_randomization.push_robots = True
    config.environment.terrain.curriculum = False

    # Set seed
    if "seed" not in config:
        seed = set_seed(args.seed)
        config["seed"] = seed
    else:
        seed = config["seed"]
        set_seed(seed)

    env = LeggedRobot(config, args.headless, 1)
    assert args.model is not None, "Model must be provided"
    model_path = os.path.join(args.log_dir, args.model)

    actor_critic = AgentNetwork(config)

    coach = Coach(env, config, actor_critic, args.log_dir, writer_type="none")
    coach.load(model_path)

    policy = coach.get_inference_policy()

    obs = env.get_observations()

    while True:
        with torch.no_grad():
            actions = policy(obs)
        obs, privileged, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    args = parse_args()
    play(args)
