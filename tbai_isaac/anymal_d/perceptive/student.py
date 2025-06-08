#!/usr/bin/env python3

import copy

import torch
import torch.nn as nn

# hardcoded constants - TODO: remove these
proprioceptive_size = 3 + 3 + 3 + 3 + 12 + 12 + 3 * 12 + 2 * 12 + 2 * 12 + 8
exteroceptive_latent_size = 4 * 24
priviliged_latent_size = 24
hidden_size = 50
exteroceptive_size = 4 * 52
priviliged_size = 41
gru_num_layers = 2


class BeliefEncoder(nn.Module):
    def __init__(self, n_envs, hidden_size, device):
        super().__init__()
        self.n_envs = n_envs
        self.hidden_size = hidden_size
        self.device = device

        self.gru_input_size = proprioceptive_size + exteroceptive_latent_size

        # RNN
        self.gru_num_layers = gru_num_layers
        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.gru_num_layers,
        )
        # self.gru = nn.GRUCell(input_size=self.gru_input_size, hidden_size=self.hidden_size)

        self.register_buffer("hidden", self.init_hidden())

        # encoders
        self.ga = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=64),
            nn.ELU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ELU(),
            nn.Linear(in_features=64, out_features=96),
        )

        self.gb_out_size = exteroceptive_latent_size + priviliged_latent_size
        self.exteroceptive_latent_size = exteroceptive_latent_size
        self.gb = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=64),
            nn.ELU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ELU(),
            nn.Linear(in_features=64, out_features=self.gb_out_size),
        )

        self.reset_every = 100
        self.i = 0

    def forward(self, proprioceptive, exteroceptive):
        # Forward pass through the RNN
        hidden = self.hidden
        rnn_input = self.concat(proprioceptive, exteroceptive)

        rnn_input = rnn_input.unsqueeze(1)
        output, hidden_new = self.gru(rnn_input, hidden)
        output = output.squeeze(1)

        # Gate
        alpha = torch.sigmoid(self.ga(output))
        exteroceptive_attenuated = alpha * exteroceptive

        belief_state = self.gb(output)
        belief_state[:, : self.exteroceptive_latent_size] += exteroceptive_attenuated

        self.hidden = hidden_new
        return belief_state, self.hidden, output

    def reset_graph(self):
        self.i += 1
        self.hidden = self.hidden.detach()

    def concat(self, proprio, extero_latent):
        return torch.cat((proprio, extero_latent), dim=-1)

    def init_hidden(self):
        return torch.zeros(self.gru_num_layers, self.n_envs, self.hidden_size, device=self.device)


class BeliefDecoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        # Gate
        self.ga = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=64),
            nn.ELU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ELU(),
            nn.Linear(in_features=128, out_features=4 * 52),
        )

        # Exteroceptive decoder
        self.exteroceptive_decoder = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=64),
            nn.ELU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ELU(),
            nn.Linear(in_features=128, out_features=4 * 52),
        )

        # Priviliged decoder
        self.priviliged_decoder = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=64),
            nn.ELU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ELU(),
            nn.Linear(in_features=64, out_features=priviliged_size),
        )

    def forward(self, hidden, exteroceptive):
        alpha = torch.sigmoid(self.ga(hidden))  # (n_envs, 1)
        exteroceptive_attenuated = alpha * exteroceptive

        exteroceptive_decoded = self.exteroceptive_decoder(hidden) + exteroceptive_attenuated
        priviliged_decoded = self.priviliged_decoder(hidden)

        return exteroceptive_decoded, priviliged_decoded


class StudentPolicy(nn.Module):
    def __init__(self, n_envs, teacher_policy, device="cuda"):
        super().__init__()
        self.n_envs = n_envs
        self.device = device
        self.mlp = copy.deepcopy(teacher_policy.actor).to(self.device)

        self.belief_encoder = BeliefEncoder(n_envs, hidden_size, self.device).to(self.device)
        self.belief_decoder = BeliefDecoder(hidden_size).to(self.device)

        self.ge = copy.deepcopy(teacher_policy.heights_encoder).to(self.device)
        for param in self.ge.parameters():
            param.requires_grad = True

    def reset(self, dones):
        if not dones.any():
            return
        self.belief_encoder.hidden[:, dones] = self.belief_encoder.init_hidden()[:, dones]

    def forward(self, proprioceptive, exteroceptive):
        n_envs = proprioceptive.shape[0]
        exteroceptive_encoded = self.ge(exteroceptive).view(n_envs, -1)

        belief_state, hidden, output = self.belief_encoder(proprioceptive, exteroceptive_encoded)

        mlp_in = torch.cat((proprioceptive, belief_state), dim=-1)
        action = self.mlp(mlp_in)

        reconstructed = self.belief_decoder(output, exteroceptive)

        return action, torch.cat(reconstructed, dim=-1)

    def reset_graph(self):
        self.belief_encoder.reset_graph()

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    @torch.no_grad()
    def inference(self, proprioceptive, exteroceptive):
        return self(proprioceptive, exteroceptive)


class StudentPolicyJitted(nn.Module):
    def __init__(self, student_policy):
        super().__init__()
        self.student_policy = student_policy

        self.proprioceptive_size = proprioceptive_size
        self.exteroceptive_size = exteroceptive_size
        self.priviliged_size = priviliged_size
        self.hidden_size = hidden_size
        self.exteroceptive_size = exteroceptive_size
        self.priviliged_size = priviliged_size
        self.gru_num_layers = gru_num_layers

    def forward(self, observation):
        proprioceptive = observation[:, : self.proprioceptive_size]
        exteroceptive = observation[:, self.proprioceptive_size :]
        action, reconstructed = self.student_policy(proprioceptive, exteroceptive)
        return torch.cat((action, reconstructed), dim=-1).squeeze()

    def load_from_file(self, path):
        self.student_policy.load_weights(path)

    @torch.jit.export
    def reset_hidden(self):
        self.student_policy.belief_encoder.hidden[:] = 0.0

    @torch.jit.export
    def set_hidden_size(self):
        # One because we are controlling a single robot
        self.student_policy.belief_encoder.hidden = torch.zeros(self.gru_num_layers, 1, self.hidden_size)

    def export(self, path):
        self.to("cpu")
        torch.jit.save(torch.jit.script(self), path)


## helper functions
def proprioceptive_from_observation(obs):
    return obs[:, :proprioceptive_size]


def exteroceptive_from_observation(obs):
    return obs[:, proprioceptive_size : proprioceptive_size + exteroceptive_size]


def priviliged_from_observation(obs):
    return obs[:, proprioceptive_size + exteroceptive_size :]


def priviliged_from_decoded(decoded):
    return decoded[:, -priviliged_size:]


def exteroceptive_from_decoded(decoded):
    return decoded[:, :exteroceptive_size]
