#!/usr/bin/env python3

import torch
import torch.nn as nn
from tbai_isaac.ppo.algorithm import ActorCritic
from torch.distributions import Normal


class AgentNetwork(ActorCritic):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        self.cfg = cfg

        num_actor_obs = cfg["environment/env/num_observations"]
        num_critic_obs = cfg["environment/env/num_privileged_observations"]

        if num_critic_obs == 0:
            num_critic_obs = num_actor_obs

        activation = cfg["ppo/policy/activation"]
        actor_hidden_dims = cfg["ppo/policy/actor_hidden_dims"]
        critic_hidden_dims = cfg["ppo/policy/critic_hidden_dims"]
        init_noise_std = cfg["ppo/policy/init_noise_std"]

        activation = self.get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        num_actions = 12

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation())
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation())
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation())
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation())
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(0 * init_noise_std * torch.ones(num_actions))
        self.distribution = None

        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def preprocess_input(self, observation):
        return observation

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_mean(self, observation):
        return self.actor(self.preprocess_input(observation))

    def get_std(self, observation):
        return self.std.exp()

    def update_distribution(self, observations):
        mean = self.get_mean(observation=observations)
        std = self.get_std(observation=observations)
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    @torch.no_grad()
    def act_inference(self, observations):
        actions_mean = self.get_mean(observation=observations)
        return actions_mean

    def evaluate(self, observation, **kwargs):
        return self.critic(self.preprocess_input(observation))
