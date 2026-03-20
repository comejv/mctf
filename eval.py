import os
import torch
import numpy as np
import argparse
from env import make_env
from policy import MLPPolicy


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, default="puffer_ctf", choices=["puffer_ctf", "pyquaticus"]
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    # Create one environment for evaluation
    env = make_env(args.env, num_envs=1, use_reward_wrapper=False)

    # Create policy and load weights
    policy = MLPPolicy(env)
    # Handle the fact that PuffeRL might have saved it differently
    state_dict = torch.load(args.model_path, map_location="cpu")
    # If saved via trainer.close(), it might be in a different format
    if "optimizer_state_dict" in state_dict:
        # This is a full checkpoint, not just the model
        # But wait, save_checkpoint saves the model separately
        pass

    policy.load_state_dict(state_dict)
    policy.eval()

    print(
        f"Evaluating {args.model_path} on {args.env} for {args.num_episodes} episodes..."
    )

    all_episode_returns = []

    for ep in range(args.num_episodes):
        obs, info = env.reset()
        done = False
        ep_return = 0

        while not done:
            with torch.no_grad():
                # obs shape is (num_agents, obs_shape)
                # policy expects (batch, obs_shape)
                logits, value = policy(torch.as_tensor(obs))

                # Sample actions
                if isinstance(logits, tuple):  # MultiDiscrete
                    actions = [torch.argmax(l, dim=-1).numpy() for l in logits]
                    actions = np.stack(actions, axis=1)
                else:
                    actions = torch.argmax(logits, dim=-1).numpy()

            obs, rewards, terminals, truncations, infos = env.step(actions)
            ep_return += np.sum(rewards)
            done = np.any(terminals) or np.any(truncations)

            if args.render:
                env.render()

        all_episode_returns.append(ep_return)
        print(f"Episode {ep+1}: Return = {ep_return:.2f}")

    print(
        f"\nAverage Return over {args.num_episodes} episodes: {np.mean(all_episode_returns):.2f}"
    )
    env.close()


if __name__ == "__main__":
    evaluate()
