import torch
import torch.nn as nn
import numpy as np
import gymnasium
import pufferlib.pytorch

class MLPPolicy(nn.Module):
    def __init__(self, env, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Observation space info
        obs_shape = env.single_observation_space.shape
        self.obs_size = np.prod(obs_shape)
        
        # Action space info
        self.is_multidiscrete = isinstance(env.single_action_space, gymnasium.spaces.MultiDiscrete)
        if self.is_multidiscrete:
            self.action_nvec = tuple(env.single_action_space.nvec)
            num_actions = sum(self.action_nvec)
        else:
            num_actions = env.single_action_space.n

        # Encoder: raw observations -> hidden representation
        self.encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.obs_size, hidden_size)),
            nn.Tanh(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )
        
        # Actor head: hidden -> action logits
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, num_actions), std=0.01
        )
        
        # Critic head: hidden -> value estimate
        self.critic = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1.0
        )

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations)
        logits, values = self.decode_actions(hidden)
        return logits, values

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

    def encode_observations(self, observations, state=None):
        # observations might have shape (batch, obs_size) or (batch, 1, obs_size) or (batch, horizon, obs_size)
        # We need to flatten everything except the batch dimension
        batch_size = observations.shape[0]
        # Reshape to (batch_size, obs_size)
        # Using view or reshape to ensure it matches the linear layer's expectation
        x = observations.reshape(batch_size, -1)
        # Check if the shape matches
        if x.shape[1] != self.obs_size:
            # If it's still not matching, maybe it's (batch, horizon, ...) and we need (batch*horizon, ...)
            # But PuffeRL should have flattened it. Let's be robust.
            x = observations.view(-1, self.obs_size)
        
        return self.encoder(x.float())

    def decode_actions(self, hidden):
        # hidden shape: (batch, hidden_size)
        if self.is_multidiscrete:
            logits = self.actor(hidden).split(self.action_nvec, dim=1)
        else:
            logits = self.actor(hidden)
            
        value = self.critic(hidden)
        return logits, value.squeeze(-1)
