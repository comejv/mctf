import pufferlib
import pufferlib.emulation
import gymnasium
import numpy as np

class RewardWrapper:
    def __init__(self, env):
        self.env = env
        self.num_agents = env.num_agents
        self.single_observation_space = env.single_observation_space
        self.single_action_space = env.single_action_space
        self.prev_dist_to_enemy_home = np.zeros(self.num_agents)
        self.prev_dist_to_own_home = np.zeros(self.num_agents)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, actions):
        obs, rewards, terminals, truncations, infos = self.env.step(actions)
        
        # obs shape is (num_agents, 45)
        # indices for 2v2 non-lidar:
        # 0: opponent_home_bearing, 1: opponent_home_dist
        # 2: own_home_bearing, 3: own_home_dist
        # 17: has_flag, 18: on_side, 19: tagging_cooldown, 20: is_tagged
        
        new_rewards = rewards.copy().astype(np.float32)
        
        for i in range(self.num_agents):
            dist_to_enemy_home = obs[i, 1]
            dist_to_own_home = obs[i, 3]
            has_flag = obs[i, 17]
            is_tagged = obs[i, 20]
            
            step_count = infos[0].get('step', 0)
            if step_count > 1: # only if not first step
                if has_flag == 0:
                    # Reward for moving towards enemy home
                    diff = self.prev_dist_to_enemy_home[i] - dist_to_enemy_home
                    new_rewards[i] += diff * 0.1
                else:
                    # Reward for moving towards own home
                    diff = self.prev_dist_to_own_home[i] - dist_to_own_home
                    new_rewards[i] += diff * 0.2 # Higher reward for carrying flag
                    new_rewards[i] += 0.05 # Bonus for just holding the flag
            
            self.prev_dist_to_enemy_home[i] = dist_to_enemy_home
            self.prev_dist_to_own_home[i] = dist_to_own_home
            
            # Penalty for being tagged
            if is_tagged > 0:
                new_rewards[i] -= 0.1
                
            # Small time penalty
            new_rewards[i] -= 0.001

        return obs, new_rewards, terminals, truncations, infos

    def reset(self, seed=None):
        obs, infos = self.env.reset(seed=seed)
        self.prev_dist_to_enemy_home = obs[:, 1].copy()
        self.prev_dist_to_own_home = obs[:, 3].copy()
        return obs, infos

    def close(self):
        return self.env.close()

def make_ctf_c_env(num_envs=1, use_reward_wrapper=False, **kwargs):
    """
    Creates the fast C implementation of the CTF environment.
    """
    from pufferlib.ocean.ctf import ctf
    env = ctf.CTF(num_envs=num_envs, **kwargs)
    if use_reward_wrapper:
        env = RewardWrapper(env)
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

def make_env(env_name="puffer_ctf", num_envs=1, **kwargs):
    """
    Factory function for environments.
    """
    if env_name == "puffer_ctf":
        return make_ctf_c_env(num_envs=num_envs, **kwargs)
    elif env_name == "pyquaticus":
        return make_pyquaticus_env(**kwargs)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
