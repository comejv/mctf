import os
import torch
import numpy as np
import argparse
import pufferlib
from env import make_env
from policy import MLPPolicy, RecurrentPolicy


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, default="puffer_ctf", choices=["puffer_ctf", "pyquaticus"]
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of taking argmax",
    )

    # We need to know if the model was trained with RNN
    # For now let's add an explicit flag or try to guess
    parser.add_argument("--rnn", action="store_true", help="Use recurrent policy")

    args, unknown = parser.parse_known_args()

    # Create one environment for evaluation
    env = make_env(args.env, num_envs=1, use_reward_wrapper=True)

    if args.rnn:
        policy = RecurrentPolicy(env, hidden_size=args.hidden_size)
        is_recurrent = True
    else:
        policy = MLPPolicy(env, hidden_size=args.hidden_size)
        is_recurrent = False

    state_dict = None
    model_path = args.model_path
    if os.path.isdir(model_path):
        import glob
        checkpoints = sorted(glob.glob(os.path.join(model_path, "model_*.pt")))
        if not checkpoints:
            print(f"No model checkpoints found in {model_path}")
            return
        model_path = checkpoints[-1]
        print(f"Using latest checkpoint: {model_path}")

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    policy.load_state_dict(state_dict)
    policy.eval()

    print(
        f"Evaluating {args.model_path} on {args.env} for {args.num_episodes} episodes..."
    )

    all_episode_returns = []
    all_blue_scores = []
    all_red_scores = []

    for ep in range(args.num_episodes):
        # Use random seed for each episode to see variety
        seed = int(np.random.randint(0, 1000000))
        obs, info = env.reset(seed=seed)
        done = False
        ep_return = 0
        ep_blue_score = 0
        ep_red_score = 0

        # Initialize LSTM state if recurrent
        state = None
        if is_recurrent:
            h = torch.zeros(env.num_agents, policy.rnn_size)
            c = torch.zeros(env.num_agents, policy.rnn_size)
            state = {"lstm_h": h, "lstm_c": c}

        while not done:
            with torch.no_grad():
                if is_recurrent:
                    # pufferlib.models.LSTMWrapper.forward_eval updates state in-place
                    # but we can also capture it just in case
                    logits, value = policy.forward_eval(torch.as_tensor(obs), state)
                else:
                    logits, value = policy(torch.as_tensor(obs))

                if args.stochastic:
                    action, logprob, entropy = pufferlib.pytorch.sample_logits(logits)
                    actions = action.numpy()
                else:
                    if isinstance(logits, tuple):  # MultiDiscrete
                        actions = [torch.argmax(l, dim=-1).numpy() for l in logits]
                        actions = np.stack(actions, axis=1)
                    else:
                        actions = torch.argmax(logits, dim=-1).numpy()

            # The observation at index 19 is 'team_score' and 20 is 'opponent_score'
            # (normalized by max_score).
            # Blue team: agents 0, 1. Red team: agents 2, 3.
            # We can use agent 0's obs to get both.
            # agent_0 team_score is blue, opponent_score is red.
            blue_score_norm = obs[0, 19]
            red_score_norm = obs[0, 20]
            
            # max_score is usually 3 in config.ini but let's be safe and just track 
            # if the normalized value changed.
            # Actually, let's just use the raw values if we can guess max_score=3.
            # Or better, just print when it increases.
            new_blue = int(round(blue_score_norm * 3))
            new_red = int(round(red_score_norm * 3))
            
            if new_blue > ep_blue_score:
                ep_blue_score = new_blue
            if new_red > ep_red_score:
                ep_red_score = new_red
            
            obs, rewards, terminals, truncations, infos = env.step(actions)
            ep_return += np.sum(rewards)

            if infos:
                for info_dict in infos:
                    if isinstance(info_dict, dict):
                        # The C code might report scores in its log
                        if "blue_score" in info_dict:
                             ep_blue_score = max(ep_blue_score, info_dict["blue_score"])
                        if "red_score" in info_dict:
                             ep_red_score = max(ep_red_score, info_dict["red_score"])

            done = np.any(terminals) or np.any(truncations)

            if args.render:
                env.render()
                import time
                time.sleep(1 / 30.0)  # ~30 FPS

        all_episode_returns.append(ep_return)
        all_blue_scores.append(ep_blue_score)
        all_red_scores.append(ep_red_score)
        print(
            f"Episode {ep+1}: Return = {ep_return:.2f} | Blue: {ep_blue_score} | Red: {ep_red_score}"
        )

    print(f"\nResults over {args.num_episodes} episodes:")
    print(f"  Average Return:     {np.mean(all_episode_returns):.2f}")
    print(f"  Average Blue Score: {np.mean(all_blue_scores):.2f}")
    print(f"  Average Red Score:  {np.mean(all_red_scores):.2f}")
    env.close()


if __name__ == "__main__":
    evaluate()
