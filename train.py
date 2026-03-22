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


class TensorboardLogger:
    def __init__(self, log_dir, run_id):
        from torch.utils.tensorboard import SummaryWriter

        self.run_id = run_id
        self.writer = SummaryWriter(log_dir)

    def log(self, stats, step):
        for k, v in stats.items():
            if isinstance(v, (int, float, np.float32, np.int64)):
                self.writer.add_scalar(k, v, step)


def train():
    # Parse CLI arguments
    import sys

    resume = "--resume" in sys.argv
    if resume:
        sys.argv.remove("--resume")

    config_file = "config.ini"
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            config_file = sys.argv[idx + 1]
            del sys.argv[idx : idx + 2]
        else:
            print("Error: --config requires a file path")
            sys.exit(1)

    # Load config using PufferLib helper
    config = load_config_file(config_file)
    print(f"Loaded config from {config_file}")

    train_config = config["train"]
    vec_config = config["vec"]
    env_name = config["env_name"]

    use_reward_wrapper = train_config.get("use_reward_wrapper", True)
    hidden_size = train_config.get("hidden_size", 256)

    # Extract reward shaping parameters
    reward_shaping = {
        "reward_move_to_enemy": train_config.get("reward_move_to_enemy", 0.01),
        "reward_move_to_own": train_config.get("reward_move_to_own", 0.05),
        "reward_flag_hold": train_config.get("reward_flag_hold", 0.01),
        "penalty_tagged": train_config.get("penalty_tagged", -0.5),
        "penalty_step": train_config.get("penalty_step", -0.001),
        "penalty_wall": train_config.get("penalty_wall", 0.0),
    }

    # Override/Ensure data_dir
    if "data_dir" not in train_config or not train_config["data_dir"]:
        train_config["data_dir"] = "experiments"

    if not os.path.exists(train_config["data_dir"]):
        os.makedirs(train_config["data_dir"])

    # Create environment
    if vec_config.get("num_workers", 1) > 1:
        import pufferlib.vector

        backend = vec_config.get("backend", "Multiprocessing")
        if hasattr(pufferlib.vector, backend):
            backend = getattr(pufferlib.vector, backend)
        else:
            backend = pufferlib.vector.Multiprocessing

        envs_per_worker = vec_config["num_envs"] // vec_config["num_workers"]
        print(
            f"Creating vectorized environment with {vec_config['num_workers']} workers, {envs_per_worker} envs each"
        )
        env = pufferlib.vector.make(
            make_env,
            env_kwargs={
                "env_name": env_name,
                "num_envs": 1,
                "use_reward_wrapper": use_reward_wrapper,
                "reward_shaping": reward_shaping,
            },
            num_envs=vec_config["num_envs"],
            num_workers=vec_config["num_workers"],
            batch_size=vec_config.get("batch_size", vec_config["num_envs"]),
            backend=backend,
        )
    else:
        env = make_env(
            env_name,
            num_envs=vec_config["num_envs"],
            use_reward_wrapper=use_reward_wrapper,
            reward_shaping=reward_shaping,
        )

    # Set torch threads to use remaining CPU capacity for the trainer
    import multiprocessing
    total_cores = multiprocessing.cpu_count()
    num_workers = vec_config.get("num_workers", 1)
    
    # If we have many workers, the trainer should use fewer threads to avoid context switching
    if num_workers >= total_cores:
        num_threads = 1
    else:
        num_threads = max(1, total_cores - num_workers)
        
    torch.set_num_threads(num_threads)
    print(f"Total cores: {total_cores}, Workers: {num_workers}, Torch threads: {num_threads}")

    # Create policy
    if train_config.get("use_rnn"):
        from policy import RecurrentPolicy

        policy = RecurrentPolicy(env, hidden_size=hidden_size)
    else:
        policy = MLPPolicy(env, hidden_size=hidden_size)

    policy.to(train_config["device"])

    # Setup Logger and Paths
    # PufferRL expects 'env' and 'data_dir' in config
    train_config["env"] = env_name
    if "data_dir" not in train_config or not train_config["data_dir"]:
        train_config["data_dir"] = "experiments"

    run_id = f"{int(time.time())}"
    # This will result in experiments/env_name_run_id/
    run_dir = os.path.join(train_config["data_dir"], f"{env_name}_{run_id}")

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    # Check if we should resume
    latest_exp = None
    if resume:
        import glob

        exps = sorted(glob.glob(os.path.join(train_config["data_dir"], f"{env_name}*")))
        if exps:
            # Look for the latest experiment that has a trainer_state.pt
            for exp in reversed(exps):
                if os.path.exists(os.path.join(exp, "trainer_state.pt")):
                    latest_exp = exp
                    # Try to reuse the run_id to keep logs consistent if possible
                    # but PufferLib might create a new folder anyway.
                    # For now let's just use the checkpoint.
                    print(f"Resuming from experiment: {latest_exp}")
                    break

    if config.get("wandb"):
        from pufferlib.pufferl import WandbLogger

        logger = WandbLogger(train_config)
    else:
        # Default to our Tensorboard logger.
        # We pass run_id so PuffeRL can find it, and run_dir for the logs.
        logger = TensorboardLogger(run_dir, run_id)

    # Initialize trainer
    trainer = PuffeRL(
        config=train_config,
        vecenv=env,
        policy=policy,
        logger=logger,
    )

    # Load checkpoint if resuming
    if latest_exp:
        # Find latest model checkpoint in that dir
        model_checkpoints = sorted(glob.glob(os.path.join(latest_exp, "model_*.pt")))
        if model_checkpoints:
            latest_model = model_checkpoints[-1]
            print(f"Loading model checkpoint: {latest_model}")
            # Map location cpu since we are on CPU
            policy.load_state_dict(
                torch.load(
                    latest_model, map_location=train_config["device"], weights_only=True
                )
            )

            # Load trainer state
            trainer_state_path = os.path.join(latest_exp, "trainer_state.pt")
            if os.path.exists(trainer_state_path):
                print(f"Loading trainer state: {trainer_state_path}")
                # We need weights_only=False for the full trainer state which has dicts
                state = torch.load(
                    trainer_state_path,
                    map_location=train_config["device"],
                    weights_only=False,
                )
                trainer.optimizer.load_state_dict(state["optimizer_state_dict"])
                trainer.global_step = state["global_step"]
                # PufferRL uses 'update' key for epoch/update number
                if "update" in state:
                    trainer.epoch = state["update"]

    # Training loop
    print(
        f"Starting training on {env_name} for {train_config['total_timesteps']} steps..."
    )
    try:
        while trainer.global_step < train_config["total_timesteps"]:
            trainer.evaluate()
            trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training crashed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Finalize
        trainer.close()
        print(f"Environment closed and model saved.")


if __name__ == "__main__":
    train()
