#!/usr/bin/env python3

import argparse

import torch
import torch.nn as nn
from tbai_isaac.envs.anymal_d.dtc.agent import AgentNetwork

from tbai_isaac.common.config import load_config


def export(args):
    config = load_config(args.config)
    model = torch.load(args.model_in)
    actor_critic = AgentNetwork(config)
    actor_critic.load_state_dict(model["model_state_dict"])

    class ExportedActor(nn.Module):
        def __init__(self, actor_critic):
            super().__init__()
            self.actor = actor_critic.actor

        def forward(self, x):
            return self.actor(x)

        def export(self):
            self.to("cpu")
            self.eval()
            torch.jit.save(torch.jit.script(self), args.model_out)

    ea = ExportedActor(actor_critic)
    ea.export()

    model = torch.jit.load(args.model_out).eval()

    n_obs = config.environment.env.num_observations
    test_input = torch.arange(n_obs).reshape(1, -1).float()

    assert torch.allclose(actor_critic.get_mean(test_input), model(test_input))
    print(model(test_input))
    print(f"Model exported successfully as {args.model_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model_in", type=str, required=True, help="Path to model file")
    parser.add_argument("--model_out", type=str, required=True, help="Path to output model file")
    args = parser.parse_args()
    export(args)
