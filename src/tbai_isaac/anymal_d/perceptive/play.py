#!/usr/bin/env python3

import os

import isaacgym  # noqa: F401
import torch
from tbai_isaac.anymal_d.perceptive.env import LeggedRobot
from tbai_isaac.anymal_d.perceptive.teacher import TeacherNetwork
from tbai_isaac.common.config import load_config
from tbai_isaac.common.utils import parse_args, set_seed, store_config
from tbai_isaac.ppo.coach import Coach


def play(args):
    config = load_config(args.config)
    config.environment.env.num_envs = min(config.environment.env.num_envs, 50)
    config.environment.terrain.num_rows = 5
    config.environment.terrain.num_cols = 5
    config.environment.terrain.curriculum = False
    config.environment.noise.add_noise = False
    config.environment.domain_randomization.randomize_friction = False
    config.environment.domain_randomization.push_robots = False

    # Set seed
    if "seed" not in config:
        seed = set_seed(args.seed)
        config["seed"] = seed
    else:
        seed = config["seed"]
        set_seed(seed)

    store_config(args, config)

    env = LeggedRobot(config, args.headless)
    assert args.model is not None, "Model must be provided"
    model_path = os.path.join(args.log_dir, args.model)

    actor_critic = TeacherNetwork(config)

    coach = Coach(env, config.ppo, actor_critic, "./logs", writer_type="none")
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
