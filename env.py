import pufferlib
import pufferlib.emulation
import gymnasium
import numpy as np


def make_ctf_c_env(num_envs=1, reward_shaping=None, **kwargs):
    """
    Creates the fast C implementation of the CTF environment.
    Reward shaping is handled natively in the C backend.
    """
    from pufferlib.ocean.ctf import ctf
    
    if reward_shaping is None:
        reward_shaping = {}
        
    # These values are passed directly to the C engine
    env = ctf.CTF(
        num_envs=num_envs, 
        reward_move_to_enemy=reward_shaping.get("reward_move_to_enemy", 0.01),
        reward_move_to_own=reward_shaping.get("reward_move_to_own", 0.05),
        reward_flag_hold=reward_shaping.get("reward_flag_hold", 0.01),
        penalty_tagged=reward_shaping.get("penalty_tagged", -0.5),
        penalty_step=reward_shaping.get("penalty_step", -0.001),
        penalty_wall=reward_shaping.get("penalty_wall", 0.0),
        **kwargs
    )
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
