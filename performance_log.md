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

