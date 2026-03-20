# Performance Log - Puffer MAS

## Iteration 0: Baseline
- **Date**: 2026-03-20
- **Environment**: `puffer_ctf`
- **Total Timesteps**: 1,000,000
- **Configuration**: Default PPO from `config.ini`
- **Changes**: None (Baseline)
- **Results**:
  - SPS: ~80k
  - Episode Return (Original): 0.00
  - Flags Captured: 0

## Iteration 1: Dense Rewards
- **Date**: 2026-03-20
- **Environment**: `puffer_ctf`
- **Total Timesteps**: 2,000,000
- **Changes**:
  - Added `RewardWrapper` with:
    - Potential-based reward for moving towards enemy home (if no flag).
    - Potential-based reward for moving towards own home (if has flag).
    - Penalty for being tagged.
    - Small step penalty.
- **Results**:
  - SPS: ~80k
  - Episode Return (Dense): ~-0.047
  - Flags Captured: 0

## Iteration 2: Larger Network & Refined Rewards
- **Date**: 2026-03-20
- **Environment**: `puffer_ctf`
- **Total Timesteps**: 5,000,000
- **Changes**:
  - Modified `RewardWrapper` to use edge-triggered penalty for tagging (`-0.5`).
  - Tuned distance reward scales.
  - Increased `MLPPolicy` network size to `[256, 256]` with `ReLU` activations.
- **Results**:
  - SPS: ~60k (slower due to larger network)
  - Episode Return (Dense): ~-0.188 (mostly step penalties, some positive distance reward)
  - Flags Captured: 0

## Iteration 3: Recurrent Policy (LSTM)
- **Date**: 2026-03-20
- **Environment**: `puffer_ctf`
- **Total Timesteps**: 15,000,000
- **Changes**:
  - Switched to `RecurrentPolicy` (LSTM wrapper around 256-unit MLP).
  - Increased `total_timesteps` to 15M.
  - Adjusted `gamma` to 0.995 and `learning_rate` to 0.005 for longer horizon and stability.
- **Results**:
  - SPS: ~17k (much slower due to LSTM overhead)
  - Episode Return (Dense): TBD
  - Flags Captured: TBD

