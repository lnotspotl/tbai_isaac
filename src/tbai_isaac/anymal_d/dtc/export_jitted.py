#!/usr/bin/env python3

import torch
import torch.nn as nn
from tbai_isaac.anymal_d.dtc.agent import AgentNetwork
from tbai_isaac.anymal_d.dtc.env import LeggedRobot
from tbai_isaac.common.config import load_config
from tbai_isaac.common.utils import parse_args
from tbai_isaac.ppo.coach import Coach


def export(args):
    config = load_config(args.config)

    config.environment.env.num_envs = min(config.environment.env.num_envs, 5)
    config.environment.terrain.num_rows = 5
    config.environment.terrain.num_cols = 5
    config.environment.terrain.curriculum = False
    config.environment.noise.add_noise = False
    config.environment.domain_randomization.randomize_friction = False
    config.environment.domain_randomization.push_robots = False
    if args.max_iterations is None:
        args.max_iterations = config.runner.max_iterations

    env = LeggedRobot(config, args.headless)

    model_in = "./model.pt"
    model_out = "./model_deploy_jitted.pt"

    actor_critic = AgentNetwork(config)

    coach = Coach(env, config, actor_critic, "./logs", "cuda")
    coach.load(model_in)

    class ExportedActor(nn.Module):
        def __init__(self, actor_critic):
            super().__init__()
            self.actor = actor_critic.actor

        def forward(self, x):
            return self.actor(x)

        def export(self):
            self.to("cpu")
            self.eval()
            torch.jit.save(torch.jit.script(self), model_out)

    # Get actor
    ea = ExportedActor(actor_critic)
    ea.export()


if __name__ == "__main__":
    args = parse_args()
    export(args)
