#!/usr/bin/env python3


from tbai_isaac.anymal_d.dtc.agent import AgentNetwork
from tbai_isaac.anymal_d.dtc.env import LeggedRobot
from tbai_isaac.common.args import parse_args
from tbai_isaac.ppo.coach import Coach

from tbai_isaac.common.config import load_config, select

def train(args):
    config = load_config("./config.yaml")
    if args.max_iterations is None:
        args.max_iterations = config.ppo.runner.max_iterations

    env = LeggedRobot(config, args.rl_device, args.headless)

    actor_critic = AgentNetwork(config)

    coach = Coach(env, config.ppo, actor_critic, "./logs", "cuda", "wandb")
    coach.load("logs/model_pretrained.pt")

    coach.train(args.max_iterations, True)


if __name__ == "__main__":
    args = parse_args()
    train(args)
