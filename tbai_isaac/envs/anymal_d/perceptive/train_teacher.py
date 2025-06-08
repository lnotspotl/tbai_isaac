#!/usr/bin/env python3


from tbai_isaac.envs.anymal_d.perceptive.env import LeggedRobot
from tbai_isaac.envs.anymal_d.perceptive.teacher import TeacherNetwork
from tbai_isaac.common.config import load_config
from tbai_isaac.common.utils import parse_args, create_dir
from tbai_isaac.ppo.coach import Coach


def train(args):
    config = load_config(args.config)

    env = LeggedRobot(config, args.headless)

    actor_critic = TeacherNetwork(config)

    create_dir(args.log_dir)
    coach = Coach(env, config, actor_critic, args.log_dir, args.writer_type)

    coach.train(args.max_iterations, True)


if __name__ == "__main__":
    args = parse_args()
    train(args)
