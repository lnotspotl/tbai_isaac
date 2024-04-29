#!/usr/bin/env python3


import torch
from tbai_isaac.anymal_d.perceptive.config import AnymalConfig
from tbai_isaac.anymal_d.perceptive.env import LeggedRobot
from tbai_isaac.anymal_d.perceptive.teacher import TeacherNetwork
from tbai_isaac.common.args import parse_args
from tbai_isaac.ppo.coach import Coach


def train(args):
    config = AnymalConfig("./config.yaml")
    config["environment/env/num_envs"] = min(config["environment/env/num_envs"], 50)
    config["environment/terrain/num_rows"] = 5
    config["environment/terrain/num_cols"] = 5
    config["environment/terrain/curriculum"] = False
    config["environment/noise/add_noise"] = False
    config["environment/domain_randomization/randomize_friction"] = False
    config["environment/domain_randomization/push_robots"] = False

    env = LeggedRobot(config, args.rl_device, args.headless)

    model_path = "./logs2/model_1300.pt"

    actor_critic = TeacherNetwork(config)

    coach = Coach(env, config.as_dict()["ppo"], actor_critic, "./logs", "cuda")
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
