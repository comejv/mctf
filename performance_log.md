# Performance Log - Tabula Rasa

## Iteration 0
- **Date**: 2026-03-21
- **Experiment ID**: `puffer_ctf_puffer_ctf_1774123842`
- **Environment**: `puffer_ctf`
- **Total Timesteps**: 2,000,000
- **Configuration**: `config_baseline_v1.ini`
- **Changes**: Balanced rewards (`reward_move_to_enemy=1.0`).
- **Results**: Average Return -8.35, Scores 0-0.

## Iteration 1
- **Date**: 2026-03-21
- **Experiment ID**: `puffer_ctf_1774124528`
- **Environment**: `puffer_ctf`
- **Total Timesteps**: 2,000,000
- **Configuration**: `config_baseline_v2.ini`
- **Changes**: Aggressive rewards (`reward_move_to_enemy=5.0`).
- **Results**: Average Return 11.61, Scores 0-0.
- **Observation**: Positive returns confirmed; agents are moving aggressively toward goals but haven't captured flags yet.

## Iteration 2: Recurrent Policy (LSTM) - Aborted
- **Date**: 2026-03-21
- **Experiment ID**: `puffer_ctf_1774185436`
- **Configuration**: LSTM with aggressive rewards, `learning_rate = 0.001`, `bptt_horizon = 16`, `ent_coef = 0.2`.
- **Intermediate Steps (Aborted)**:
  - **Take 1**: Failed due to `APIUsageError` (batch_size incompatibility with num_workers).
  - **Take 2**: Ran for ~3M steps but suffered from **Policy Collapse**. Entropy dropped to <0.15 and Explained Variance was very low (<0.05). Agents learned to "stay safe" (return ~0) but stopped exploring.
  - **Take 3**: Used `ent_coef = 0.2` to force exploration. Successfully maintained entropy at ~2.79, but return plateaued near 0.
- **Final Metrics at Abortion (Take 3)**:
  - **Steps**: 3.5M
  - **Final Return**: 0.0019 (Log-shape plateau from negatives)
  - **Final LR**: 9.8e-05 (Approaching zero)
  - **Results**: Average Return 13.46 (Constant/Degenerate), Scores 0-0.
- **Observation**: The learning rate decayed significantly and rewards plateaued at 0. Despite high exploration (entropy), the agents were unable to discover flag captures, and the aggressive reward shaping may have been counter-productive for initial discovery.

## Iteration 3: Golden Config (Successful Replication) - Aborted
- **Date**: 2026-03-21
- **Experiment ID**: `puffer_ctf_1774186038`
- **Configuration**: Reverted to "Golden" parameters: `learning_rate = 0.005`, `gamma = 0.995`, `bptt_horizon = 64`, `ent_coef = 0.01`, with original soft reward shaping.
- **Results at Abortion (5.9M steps)**:
  - SPS: ~9.7k
  - Average Episode Return: 0.0039
  - Average Blue/Red Score: 0.00
  - Entropy: 2.1193
  - Explained Variance: 0.1949
- **Observation**: Training was stable and entropy remained healthy (indicating continued exploration), but agents failed to score any points within the first 5.9M steps. No model checkpoint was saved as it was aborted before the first 200-epoch checkpoint interval (which would have been at 6.55M steps).

## Iteration 4: Dense Golden MLP
- **Date**: 2026-03-21
- **Experiment ID**: `puffer_ctf_1774187707`
- **Configuration**: `config_mlp_golden.ini` (MLP, `learning_rate = 0.005`, `gamma = 0.995`, original soft rewards).
- **Results at Abortion (11.1M steps)**:
  - Average Episode Return: ~0.02 to 0.04
  - Average Blue/Red Score: 0.00
  - Entropy: 1.9085
  - Explained Variance: 0.6883
- **Observation**: The MLP learned to collect tiny micro-rewards from the distance-based shaping without ever committing to capturing the flag. The high explained variance means the value function perfectly predicted this safe, low-yield strategy.

## Iteration 5: Golden Config + No Fear + Wall Penalty
- **Date**: 2026-03-21
- **Status**: **Completed (15M Steps) - Bug Discovered**
- **Configuration**: `config_golden_no_fear.ini` (LSTM, `penalty_tagged = 0.0`, `penalty_wall = -0.1`).
- **Results**:
  - Average Episode Return: 12.26
  - Average Blue/Red Score: 0.00
- **Observation**: Agents stayed in their own half. The high return without scores prompted a deep dive into the code. 
- **Bug Discovery**: The `RewardWrapper` in `env.py` was reading the wrong observation indices from the C environment! 
  - `has_flag` was reading `tagging_cooldown` (idx 17 instead of 15), which is `1.0`. The wrapper thought agents ALWAYS had the flag!
  - As a result, agents were constantly given `reward_flag_hold` and `reward_move_to_own`, explaining why they stayed in their own half and accumulated +12 return per episode without doing anything.
  - This bug has affected all "Dense/Refined" runs. 

## Next Step
- Fixed `env.py` to use correct indices (`has_flag`=15, `is_tagged`=18).
- Ready for a true "Tabula Rasa" run with the fixed reward wrapper.

## Iteration 6: Post-Fix High Entropy Run
- **Date**: 2026-03-22
- **Experiment ID**: `puffer_ctf_1774194424`
- **Environment**: `puffer_ctf`
- **Total Timesteps**: 20,000,000
- **Configuration**: LSTM, `ent_coef = 0.1`, `learning_rate = 0.001`, `reward_move_to_enemy = 1.0`, `reward_move_to_own = 2.0`.
- **Results**: Average Episode Return: -18.50, Scores 0-0.
- **Observation**: The entropy coefficient (`0.1`) was far too high, leading to policy collapse where agents completely failed to learn any coherent behavior. The return was deeply negative, implying they continuously received penalties (likely from step penalties or getting tagged) without successfully exploiting the aggressive reward shaping.
- **Next Step**: Start a new run with `ent_coef = 0.01` (standard PPO), a lower learning rate (`0.0005`), and more balanced rewards (`0.1` for moving) to encourage stable learning.

## Iteration 7: High Reactivity (Smaller Network, High LR)
- **Date**: 2026-03-22
- **Experiment ID**: `puffer_ctf_1774198595`
- **Total Timesteps**: 20,000,000
- **Configuration**: LSTM, `hidden_size = 128`, `bptt_horizon = 32`, `batch_size = auto (16384)`, `learning_rate = 0.005`, `ent_coef = 0.05`.
- **Results**: 
  - Peak Blue Score: `0.2063` (at ~11.7M steps).
  - Final Blue Score: Crashed back to `0.0` to `0.07`.
  - Average Episode Return: Highly volatile (between -0.5 and +0.25).
- **Observation**: The massive learning rate and smaller network successfully allowed the agents to break the zero-score plateau and discover flag captures! However, due to the high learning rate and policy churn (catastrophic forgetting), the policy quickly collapsed and lost the behavior.
- **Render Eval Observation**: Visual evaluation of the final checkpoints revealed that agents learned a degenerate policy of simply spinning in place and doing nothing, likely to avoid movement penalties or collisions after forgetting the capture sequence.
- **Next Step**: Retain the smaller network (`128`) and BPTT (`32`) which proved capable of learning, but lower the learning rate (`0.001`) and ensure a large batch size to stabilize the gradients and prevent forgetting. We should also remove the step penalty so they don't learn to spin in place to avoid accumulating negative rewards over time.

## Iteration 8: Stabilization Run (Low LR + Large Batch)
- **Date**: 2026-03-22
- **Experiment ID**: `puffer_ctf_1774199635`
- **Total Timesteps**: 20,000,000 (Interrupted at 10M)
- **Configuration**: LSTM, `hidden_size = 128`, `learning_rate = 0.001`, `batch_size = 16384`, `penalty_step = 0.0`.
- **Results**: 
  - Average Episode Return: `-0.0176` (Stagnant near zero).
  - Blue Score: `0.0` (No captures discovered).
- **Observation**: Training was extremely stable but too slow. At 10M steps, it had failed to discover any captures. The low learning rate prevents catastrophic forgetting but also makes initial discovery much slower.

## Iteration 9: "Compass" Discovery Strategy
- **Date**: 2026-03-22
- **Experiment ID**: `puffer_ctf_1774207091`
- **Total Timesteps**: 100,000,000 (Ongoing)
- **Configuration**: LSTM, `hidden_size = 128`, `learning_rate = 0.002`, `gamma = 0.999`, `ent_coef = 0.05`, `reward_move_to_enemy = 0.2`, `reward_move_to_own = 0.4`, `penalty_step = 0.0`.
- **Results at 12.3M steps**: 
  - Average Episode Return: `-0.0020` (Recovered from -1.35).
  - Blue Score: Sporadic hits of `0.0078`, indicating discovery.
- **Observation**: The stronger, unclamped "Compass" rewards are working. The agents have learned to navigate towards the enemy home and have successfully discovered the capture sequence several times. The high gamma (0.999) is helping value the long-term capture reward.
- **Next Step**: Continue training to 100M steps to stabilize the learned capture behavior.



