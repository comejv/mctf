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
  - SPS: TBD
  - Episode Return (Dense): TBD
  - Flags Captured: TBD
