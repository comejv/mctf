import os
import time
import argparse
import configparser
import torch
import numpy as np
import pufferlib
from pufferlib.pufferl import PuffeRL, load_config_file
from env import make_env
from policy import MLPPolicy

def train():
    # Load config using PufferLib helper
    config = load_config_file("config.ini")
    
    train_config = config["train"]
    vec_config = config["vec"]
    env_name = config["env_name"]

    # Override/Ensure data_dir
    if "data_dir" not in train_config or not train_config["data_dir"]:
        train_config["data_dir"] = "experiments"
    
    if not os.path.exists(train_config["data_dir"]):
        os.makedirs(train_config["data_dir"])

    # Create environment
    env = make_env(env_name, num_envs=vec_config["num_envs"], use_reward_wrapper=True)
    
    # Create policy
    policy = MLPPolicy(env)
    policy.to(train_config["device"])
    
    # Setup Logger
    if config["wandb"]:
        from pufferlib.pufferl import WandbLogger
        logger = WandbLogger(train_config)
    else:
        from pufferlib.pufferl import NoLogger
        logger = NoLogger(train_config)

    # Initialize trainer
    trainer = PuffeRL(
        config=train_config,
        vecenv=env,
        policy=policy,
        logger=logger
    )

    # Training loop
    print(f"Starting training on {env_name} for {train_config['total_timesteps']} steps...")
    while trainer.global_step < train_config["total_timesteps"]:
        trainer.evaluate()
        trainer.train()
        
    # Finalize
    trainer.close()
    print(f"Model saved via trainer.close()")

if __name__ == "__main__":
    train()
