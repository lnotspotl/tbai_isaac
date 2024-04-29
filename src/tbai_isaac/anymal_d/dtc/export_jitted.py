#!/usr/bin/env python3

import torch
import torch.nn as nn
from tbai_isaac.anymal_d.dtc.agent import AgentNetwork
from tbai_isaac.anymal_d.dtc.config import AnymalConfig
from tbai_isaac.anymal_d.dtc.env import LeggedRobot
from tbai_isaac.common.args import parse_args
from tbai_isaac.ppo.coach import Coach


def train(args):
    config = AnymalConfig("./config.yaml")

    config = AnymalConfig("./config.yaml")
    config["environment/env/num_envs"] = min(config["environment/env/num_envs"], 5)
    config["environment/terrain/num_rows"] = 5
    config["environment/terrain/num_cols"] = 5
    config["environment/terrain/curriculum"] = False
    config["environment/noise/add_noise"] = False
    config["environment/domain_randomization/randomize_friction"] = False
    config["environment/domain_randomization/push_robots"] = False
    if args.max_iterations is None:
        args.max_iterations = config["ppo/runner/max_iterations"]

    env = LeggedRobot(config, args.rl_device, args.headless)

    model_in = "./model.pt"
    model_out = "./model_deploy_jitted.pt"

    actor_critic = AgentNetwork(config)

    coach = Coach(env, config.as_dict()["ppo"], actor_critic, "./logs", "cuda")
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
    train(args)
