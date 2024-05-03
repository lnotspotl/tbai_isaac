#!/usr/bin/env python3


from tbai_isaac.anymal_d.perceptive.env import LeggedRobot
from tbai_isaac.anymal_d.perceptive.teacher import TeacherNetwork
from tbai_isaac.common.config import load_config
from tbai_isaac.common.utils import parse_args
from tbai_isaac.ppo.coach import Coach


def train(args):
    config = load_config(args.config)
    if args.max_iterations is None:
        args.max_iterations = config["ppo/runner/max_iterations"]

    env = LeggedRobot(config, args.headless)

    actor_critic = TeacherNetwork(config)

    coach = Coach(env, config.ppo, actor_critic, "./logs3", "cuda")

    coach.train(args.max_iterations, True)


if __name__ == "__main__":
    args = parse_args()
    train(args)
