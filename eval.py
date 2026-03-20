import os
import torch
import numpy as np
import argparse
import pufferlib
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
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of taking argmax")
    args = parser.parse_args()

    # Create one environment for evaluation
    # Use the same wrapper if we want to see the shaped returns, 
    # but the scores from infos are always the ground truth (flag captures)
    env = make_env(args.env, num_envs=1, use_reward_wrapper=True)

    # Create policy and load weights
    policy = MLPPolicy(env)
    state_dict = torch.load(args.model_path, map_location="cpu")
    policy.load_state_dict(state_dict)
    policy.eval()

    print(f"Evaluating {args.model_path} on {args.env} for {args.num_episodes} episodes...")

    all_episode_returns = []
    all_blue_scores = []
    all_red_scores = []

    for ep in range(args.num_episodes):
        obs, info = env.reset()
        done = False
        ep_return = 0
        ep_blue_score = 0
        ep_red_score = 0

        while not done:
            with torch.no_grad():
                logits, value = policy(torch.as_tensor(obs))

                if args.stochastic:
                    action, logprob, entropy = pufferlib.pytorch.sample_logits(logits)
                    actions = action.numpy()
                else:
                    if isinstance(logits, tuple): # MultiDiscrete
                        actions = [torch.argmax(l, dim=-1).numpy() for l in logits]
                        actions = np.stack(actions, axis=1)
                    else:
                        actions = torch.argmax(logits, dim=-1).numpy()

            obs, rewards, terminals, truncations, infos = env.step(actions)

            ep_return += np.sum(rewards)

            # Accumulate scores from infos if present
            if len(infos) > 0:
                # binding.vec_log returns a dict with blue_score, red_score, etc.
                # In puffer_ctf, these are often cumulative or last-interval.
                # Let's see if we can get the max seen.
                for info_dict in infos:
                    if info_dict:
                        ep_blue_score = max(ep_blue_score, info_dict.get('blue_score', 0))
                        ep_red_score = max(ep_red_score, info_dict.get('red_score', 0))

            done = np.any(terminals) or np.any(truncations)

            if args.render:
                env.render()

        all_episode_returns.append(ep_return)
        all_blue_scores.append(ep_blue_score)
        all_red_scores.append(ep_red_score)
        print(f"Episode {ep+1}: Return = {ep_return:.2f} | Blue: {ep_blue_score} | Red: {ep_red_score}")


    print(f"\nResults over {args.num_episodes} episodes:")
    print(f"  Average Return:     {np.mean(all_episode_returns):.2f}")
    print(f"  Average Blue Score: {np.mean(all_blue_scores):.2f}")
    print(f"  Average Red Score:  {np.mean(all_red_scores):.2f}")
    env.close()



if __name__ == "__main__":
    evaluate()
