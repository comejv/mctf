# Performance Log - Puffer MAS

## Iteration 0: Baseline
- **Date**: 2026-03-20
- **Experiment ID**: `puffer_ctf_177402327945`
- **Environment**: `puffer_ctf`
- **Total Timesteps**: 1,000,000
- **Configuration**: Default PPO from `config.ini`, `hidden_size=128`
- **Changes**: None (Baseline)
- **Results**:
  - SPS: ~80k
  - Average Episode Return: 7.18
  - Average Blue Score: 0.00
  - Average Red Score: 0.00

## Iteration 1: Dense Rewards
- **Date**: 2026-03-20
- **Experiment ID**: `puffer_ctf_177402339617`
- **Environment**: `puffer_ctf`
- **Total Timesteps**: 2,000,000
- **Configuration**: `hidden_size=128`
- **Changes**:
  - Added `RewardWrapper` with potential-based rewards for moving towards homes and tagging penalties.
- **Results**:
  - SPS: ~80k
  - Average Episode Return: 11.51
  - Average Blue Score: 0.00
  - Average Red Score: 0.00

## Iteration 2: Larger Network & Refined Rewards
- **Date**: 2026-03-20
- **Experiment ID**: `puffer_ctf_177402395387`
- **Environment**: `puffer_ctf`
- **Total Timesteps**: 5,000,000
- **Configuration**: `hidden_size=256`
- **Changes**:
  - Modified `RewardWrapper` to use edge-triggered penalty for tagging (`-0.5`).
  - Increased `MLPPolicy` network size to `[256, 256]`.
- **Results**:
  - SPS: ~60k
  - Average Episode Return: 13.16
  - Average Blue Score: 0.00
  - Average Red Score: 0.00

## Iteration 3: Recurrent Policy (LSTM)
- **Date**: 2026-03-21
- **Experiment ID**: `puffer_ctf_puffer_ctf_1774115407`
- **Environment**: `puffer_ctf`
- **Total Timesteps**: ~15,000,000 (finished at checkpoint 004252)
- **Changes**:
  - Switched to `RecurrentPolicy` (LSTM wrapper around 256-unit MLP).
  - Increased `total_timesteps` to 15M.
  - Adjusted `gamma` to 0.995 and `learning_rate` to 0.005 for longer horizon and stability.
- **Results**:
  - SPS: ~17k (much slower due to LSTM overhead)
  - Average Episode Return: 10.21
  - Average Blue Score: 0.85
  - Average Red Score: 0.80
  - **Note**: This model successfully learned to capture flags and score points.

> **Important Note on Previous Iterations**: Iterations 0, 1, and 2 used a broken evaluation script that failed to correctly parse team scores from the observation buffer, resulting in reported scores of "0-0" regardless of actual performance. This has been fixed as of Iteration 3.

