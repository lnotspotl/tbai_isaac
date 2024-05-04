#!/usr/bin/env python3


from tbai_isaac.anymal_d.dtc.agent import AgentNetwork
from tbai_isaac.anymal_d.dtc.env import LeggedRobot
from tbai_isaac.common.config import load_config
from tbai_isaac.common.utils import parse_args
from tbai_isaac.ppo.coach import Coach


def train(args):
    config = load_config(args.config)
    if args.max_iterations is None:
        args.max_iterations = config.ppo.runner.max_iterations

    env = LeggedRobot(config, args.headless, ig_threads=5)

    actor_critic = AgentNetwork(config)

    coach = Coach(env, config.ppo, actor_critic, "./logs", "cuda", "wandb")
    coach.load("logs/model_pretrained.pt")

    coach.train(args.max_iterations, True)


if __name__ == "__main__":
    args = parse_args()
    train(args)
