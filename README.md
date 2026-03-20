# Maritime Capture The Flag (MCTF) - PufferLib + PyQuaticus

This project provides a working Reinforcement Learning pipeline for the MCTF project, supporting both the fast C implementation (`puffer_ctf`) and the full Python implementation (`pyquaticus`).

## Environment Setup

This project uses `uv` for fast, reproducible dependency management. It relies on two external local repositories:
1. `APM_5DA01_TP/05-pufferquaticus/`: A fork of PufferLib containing the `puffer_ctf` C environment.
2. `pyquaticus/`: The official PyQuaticus environment.

### 1. Installation

From the project root (`puffer_mas/`):

```bash
# Initialize and sync dependencies via uv
uv sync

# Build the C extensions (puffer_ctf)
# This requires raylib and box2d headers/libs which are included in the APM folder
cd ../APM_5DA01_TP/05-pufferquaticus/
../../puffer_mas/.venv/bin/python setup.py build_ext --inplace --force
```

**Note on Raylib/Box2D**: These libraries (`raylib-5.5_linux_amd64`, etc.) are pre-compiled and provided in the `APM` folder. They are required by `setup.py` to compile the fast C bindings for the environment. If you want to use rendering (`env.render()`), these must be present and correctly linked during build.

### 2. Project Structure

- `env.py`: Environment factory. Contains the `RewardWrapper` for dense rewards.
- `policy.py`: MLP-based Actor-Critic policy.
- `train.py`: Training script using PufferLib's `PuffeRL`.
- `eval.py`: Evaluation script to benchmark trained models.
- `config.ini`: Hyperparameters and configuration.
- `experiments/`: Directory where checkpoints and logs are saved.

## Usage

### Monitoring with Tensorboard

This project logs training progress to Tensorboard. To view the logs:

```bash
# In a separate terminal
tensorboard --logdir experiments/
```

Then open http://localhost:6006 in your browser.

### Training

To start training with the default `puffer_ctf` (fast C implementation):

```bash
.venv/bin/python train.py --train.total-timesteps 5000000 --vec.num-envs 8
```

To train with `pyquaticus` (Python implementation):

```bash
.venv/bin/python train.py --env pyquaticus --train.total-timesteps 1000000
```

### Evaluation

To evaluate a trained model:

```bash
.venv/bin/python eval.py --model-path experiments/YOUR_MODEL_PATH.pt --num-episodes 20 --render
```

## Performance Tracking

See `performance_log.md` for a history of iterations, reward shaping changes, and benchmarks.
