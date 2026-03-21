# Train-Eval-Change Loop

This document describes the iterative development workflow for training and evaluating agents in the Maritime Capture The Flag (MCTF) environment.

## 1. The Development Cycle

The core loop consists of four steps:
1.  **Change**: Modify reward shaping, policy architecture, or hyperparameters.
2.  **Train**: Run the training script to optimize the policy.
3.  **Eval**: Run the evaluation script to visualize behavior and benchmark performance.
4.  **Analyze**: Review Tensorboard logs and performance metrics.

---

## 2. Phase 1: Change

### Reward Shaping (`env.py`)
Most improvements come from refining the `RewardWrapper` in `env.py`. You can add penalties for being tagged, rewards for moving toward the flag, or bonuses for capturing it.
-   **File**: `env.py`
-   **Class**: `RewardWrapper`
-   **Method**: `step()`

### Policy Architecture (`policy.py`)
If the task requires more complex reasoning (e.g., long-term memory), you might modify the MLP or Recurrent policies.
-   **Files**: `policy.py`
-   **Classes**: `MLPPolicy`, `RecurrentPolicy`

### Hyperparameters (`config.ini`)
Tune PPO and environment settings here.
-   `[vec]`: Number of environments and workers for parallelization.
-   `[train]`: Learning rate, batch size, entropy coefficient, etc.

---

## 3. Phase 2: Train

Start a training run using the `train.py` script.

### Basic Training
```bash
.venv/bin/python train.py
```

### Resuming Training
To pick up from the latest checkpoint in `experiments/`:
```bash
.venv/bin/python train.py --resume
```

### Overriding Config via CLI
You can override any setting in `config.ini` directly from the command line using PufferLib's argument syntax:
```bash
# Override total timesteps and learning rate
.venv/bin/python train.py --train.total-timesteps 5000000 --train.learning_rate 0.001
```

---

## 4. Phase 3: Eval

Evaluate your trained model to see how it behaves and get score statistics.

### Basic Evaluation
```bash
.venv/bin/python eval.py --model-path experiments/YOUR_RUN_ID
```

### Useful Flags
-   `--render`: Show the environment window (requires `raylib`).
-   `--num-episodes 20`: Run for more episodes to get a better average.
-   `--rnn`: **Mandatory** if you trained with `use_rnn = true` in `config.ini`.
-   `--stochastic`: Sample actions instead of taking the best one (useful for seeing varied behavior).

---

## 5. Phase 4: Analyze

### Tensorboard
Monitor training progress (rewards, loss, entropy) in real-time.
```bash
tensorboard --logdir experiments/
```
Open [http://localhost:6006](http://localhost:6006) in your browser.

## 6. Reward Rebalancing & Shaping

### Findings on Reward-Averse Behavior
During initial training, you may notice that **average return climbs toward 0** while **scores drop toward 0**. This is a sign of reward-averse behavior, where agents learn to "do nothing" to avoid penalties.
-   **Cause**: The tagging penalty (default `-0.5`) and step penalty (`-0.001`) can outweigh the sparse rewards for movement (`0.01`). If moving toward the flag results in a net negative reward per step, agents will stay still.
-   **Solution**: Increase movement and flag-carrying rewards, or reduce penalties to encourage risk-taking.

### Configuring Rewards
Reward weights are now configurable directly in your `.ini` files under the `[train]` section.

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `reward_move_to_enemy` | `0.01` | Scale for moving towards enemy home (flag). |
| `reward_move_to_own` | `0.05` | Scale for moving towards own home (with flag). |
| `reward_flag_hold` | `0.01` | Bonus per step for holding the flag. |
| `penalty_tagged` | `-0.5` | One-time penalty when tagged. |
| `penalty_step` | `-0.001` | Penalty per step (to encourage speed). |

Example `config.ini` override:
```ini
[train]
reward_move_to_enemy = 1.0
reward_move_to_own = 2.0
penalty_tagged = -0.2
```
