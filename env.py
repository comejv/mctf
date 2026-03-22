import pufferlib
import pufferlib.emulation
import gymnasium
import numpy as np


class RewardWrapper(pufferlib.PufferEnv):
    def __init__(self, env, reward_shaping=None):
        self.env = env
        self.num_agents = env.num_agents
        self.single_observation_space = env.single_observation_space
        self.single_action_space = env.single_action_space
        self.prev_dist_to_enemy_home = np.zeros(self.num_agents)
        self.prev_dist_to_own_home = np.zeros(self.num_agents)
        self.prev_is_tagged = np.zeros(self.num_agents)
        self.step_count = 0
        self._closed = False

        # Default reward shaping
        self.reward_shaping = {
            "reward_move_to_enemy": 0.01,
            "reward_move_to_own": 0.05,
            "reward_flag_hold": 0.01,
            "penalty_tagged": -0.5,
            "penalty_step": -0.001,
            "penalty_wall": 0.0,
        }
        if reward_shaping:
            self.reward_shaping.update(reward_shaping)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, actions):
        obs, rewards, terminals, truncations, infos = self.env.step(actions)
        self.step_count += 1

        new_rewards = rewards.copy().astype(np.float32)

        for i in range(self.num_agents):
            dist_to_enemy_home = obs[i, 1]
            dist_to_own_home = obs[i, 3]
            has_flag = obs[i, 15]  # Index 15: 1.0 if true, -1.0 if false
            is_tagged = obs[i, 18] # Index 18: 1.0 if true, -1.0 if false

            if self.step_count > 1:  # only if not first step
                if has_flag < 0:  # Doesn't have flag (-1.0)
                    # Reward for moving towards enemy home
                    diff = self.prev_dist_to_enemy_home[i] - dist_to_enemy_home
                    new_rewards[i] += diff * self.reward_shaping["reward_move_to_enemy"]
                else:  # Has flag (1.0)
                    # Reward for moving towards own home
                    diff = self.prev_dist_to_own_home[i] - dist_to_own_home
                    new_rewards[i] += diff * self.reward_shaping["reward_move_to_own"]
                    new_rewards[i] += self.reward_shaping["reward_flag_hold"]

            self.prev_dist_to_enemy_home[i] = dist_to_enemy_home
            self.prev_dist_to_own_home[i] = dist_to_own_home

            # Penalty for being tagged (edge-triggered)
            if is_tagged > 0 and self.prev_is_tagged[i] < 0:
                new_rewards[i] += self.reward_shaping["penalty_tagged"]

            self.prev_is_tagged[i] = is_tagged

            # Penalty for wall hugging (continuous)
            # Wall distances are at indices 5, 7, 9, 11 (normalized by env diagonal ~111)
            # If distance is < 0.02, they are within ~2 units of a wall
            min_wall_dist = min(obs[i, 5], obs[i, 7], obs[i, 9], obs[i, 11])
            if min_wall_dist < 0.02:
                new_rewards[i] += self.reward_shaping.get("penalty_wall", 0.0)

            # Small time penalty
            new_rewards[i] += self.reward_shaping["penalty_step"]

        return obs, new_rewards, terminals, truncations, infos

    def reset(self, seed=None):
        if seed is None:
            seed = 0
        obs, infos = self.env.reset(seed=seed)
        self.prev_dist_to_enemy_home = obs[:, 1].copy()
        self.prev_dist_to_own_home = obs[:, 3].copy()
        self.prev_is_tagged = obs[:, 18].copy()
        self.step_count = 0
        return obs, infos

    def close(self):
        if not self._closed:
            self._closed = True
            return self.env.close()


def make_ctf_c_env(num_envs=1, use_reward_wrapper=False, reward_shaping=None, **kwargs):
    """
    Creates the fast C implementation of the CTF environment.
    """
    from pufferlib.ocean.ctf import ctf

    env = ctf.CTF(num_envs=num_envs, **kwargs)
    if use_reward_wrapper:
        env = RewardWrapper(env, reward_shaping=reward_shaping)
    return env


def make_pyquaticus_env(team_size=2, **kwargs):
    """
    Creates the Python implementation of PyQuaticus.
    """
    import pyquaticus
    from pyquaticus.config import config_dict_std

    config = config_dict_std.copy()
    # Apply any config overrides from kwargs
    for k, v in kwargs.items():
        if k in config:
            config[k] = v

    env = pyquaticus.pyquaticus_v0.env(team_size=team_size, config_dict=config)
    return pufferlib.emulation.PettingZooPufferEnv(env)


def make_env(env_name="puffer_ctf", num_envs=1, reward_shaping=None, **kwargs):
    """
    Factory function for environments.
    """
    if env_name == "puffer_ctf":
        return make_ctf_c_env(
            num_envs=num_envs, reward_shaping=reward_shaping, **kwargs
        )
    elif env_name == "pyquaticus":
        return make_pyquaticus_env(**kwargs)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
