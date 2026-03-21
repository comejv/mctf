# Performance Log - Tabula Rasa
## Iteration 0 (V1): Simple MLP Baseline with Balanced Rewards
- **Date**: 2026-03-21
- **Experiment ID**: `puffer_ctf_puffer_ctf_1774123842`
- **Environment**: `puffer_ctf`
- **Total Timesteps**: 2,000,000
- **Configuration**: `config_baseline_v1.ini`
- **Changes**:
  - Balanced rewards: `reward_move_to_enemy=1.0`, `penalty_tagged=-0.1`, `penalty_step=-0.01`.
- **Results**:
  - SPS: ~14k
  - Average Episode Return: -8.35
  - Average Blue Score: 0.00
  - Average Red Score: 0.00
  - **Observation**: Negative return suggests agents are getting tagged frequently while exploring, but not yet capturing flags.

