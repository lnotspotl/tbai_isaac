#!/usr/bin/env python3

import torch
from tbai_isaac.anymal_d.dtc.agent import AgentNetwork
from tbai_isaac.anymal_d.dtc.env import LeggedRobot
from tbai_isaac.common.args import parse_args
from tbai_isaac.ppo.coach import Coach

from tbai_isaac.common.config import load_config, select


def train(args):

    config = load_config("./config.yaml")
    config.environment.env.num_envs = min(config.environment.env.num_envs, 5)
    config.environment.terrain.num_rows = 5
    config.environment.terrain.num_cols = 5
    config.environment.terrain.curriculum = False
    config.environment.noise.add_noise = False
    config.environment.domain_randomization.randomize_friction = False
    config.environment.domain_randomization.push_robots = False

    env = LeggedRobot(config, args.rl_device, args.headless)

    model_path = "./w_default_angles.pt"
    model_path = "./w_optimized_angles.pt"
    model_path = "./w_default_angles_w_noise.pt"
    # model_path="./w_optimized_angles_w_noise.pt"
    model_path = "./different_gaits.pt"
    model_path = "./model.pt"

    actor_critic = AgentNetwork(config)

    coach = Coach(env, config.ppo, actor_critic, "./logs", "cuda")
    coach.load(model_path)

    policy = coach.get_inference_policy()

    obs = env.get_observations()

    while True:
        with torch.no_grad():
            actions = policy(obs)
        obs, privileged, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    args = parse_args()
    train(args)
