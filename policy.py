import torch
import torch.nn as nn
import numpy as np
import gymnasium
import pufferlib.pytorch
import pufferlib.models


class MLPPolicy(nn.Module):
    def __init__(self, env, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size

        # Observation space info
        obs_shape = env.single_observation_space.shape
        self.obs_size = np.prod(obs_shape)

        # Action space info
        self.is_multidiscrete = isinstance(
            env.single_action_space, gymnasium.spaces.MultiDiscrete
        )
        if self.is_multidiscrete:
            self.action_nvec = tuple(env.single_action_space.nvec)
            num_actions = sum(self.action_nvec)
        else:
            num_actions = env.single_action_space.n

        # Encoder: raw observations -> hidden representation
        self.encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.obs_size, hidden_size)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )

        # Actor head: hidden -> action logits
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, num_actions), std=0.01
        )

        # Critic head: hidden -> value estimate
        self.critic = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1.0)
        self.is_continuous = False

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations)
        logits, values = self.decode_actions(hidden)
        return logits, values

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

    def encode_observations(self, observations, state=None):
        batch_size = observations.shape[0]
        x = observations.reshape(batch_size, -1)
        if x.shape[1] != self.obs_size:
            x = observations.view(-1, self.obs_size)

        return self.encoder(x.float())

    def decode_actions(self, hidden):
        if self.is_multidiscrete:
            logits = self.actor(hidden).split(self.action_nvec, dim=1)
        else:
            logits = self.actor(hidden)

        value = self.critic(hidden)
        return logits, value.squeeze(-1)


class RecurrentPolicy(pufferlib.models.LSTMWrapper):
    def __init__(self, env, hidden_size=256):
        policy = MLPPolicy(env, hidden_size)
        super().__init__(env, policy, input_size=hidden_size, hidden_size=hidden_size)
        self.rnn_size = hidden_size
