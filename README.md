# Deep Q-Learning Agent — LunarLander

> A production-quality Deep Q-Network (DQN) implementation that learns to land
> a spacecraft using PyTorch and Gymnasium — built as a graded assignment and
> portfolio project demonstrating applied reinforcement learning engineering.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange)](https://pytorch.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.1.1-green)](https://gymnasium.farama.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **Note on repository name:** This repository is named
> `deep-q-learning-atari-agent` for historical reasons (the original assignment
> brief referenced Atari environments). The actual implementation targets
> **LunarLander-v3**, a Box2D-based Gymnasium environment — not an Atari
> ROM-based environment. No Atari ROMs, ALE, or pixel preprocessing are used.

---

## 📋 Overview

This project trains a DQN agent to solve the **LunarLander-v3** environment
from Gymnasium. The agent receives an 8-dimensional state vector (position,
velocity, angle, leg contacts) and must learn to fire four engine commands
to land safely on the pad without crashing.

**Problem:** Sequential decision-making under uncertainty — the agent receives
a reward signal and must discover the landing strategy from scratch, with no
prior knowledge of the physics.

**Approach:** Deep Q-Network with experience replay, target network, and
epsilon-greedy exploration — the foundational algorithm from Mnih et al. (2015)
that first demonstrated human-level performance on classic control tasks.

**Why LunarLander-v3?**
- CPU-friendly — trains in 30–60 minutes (no GPU required)
- No ROM licensing issues (unlike Atari environments)
- Clean 8-dim vector observation — focuses on DQN architecture, not image preprocessing
- Dense reward signal — ideal for demonstrating Q-value convergence
- Visually compelling results for demos and video recordings

---

## ✨ Features

- **DQN with experience replay** — 50,000-transition ring buffer breaks
  temporal correlation between training samples
- **Target network** — hard-copied every 10 episodes for stable Bellman targets
- **Epsilon-greedy + Softmax exploration** — both strategies implemented and
  compared via controlled experiments
- **Controlled hyperparameter experiments** — learning rate, gamma, and epsilon
  decay sweeps with CSV outputs and comparison plots
- **Structured logging** — Python `logging` module + CSV writer via `TrainingLogger`
- **Optional reward shaping** — `ShapedLunarLander` wrapper (leg-contact bonus)
- **Full theory documentation** — 6-section writeup covering Bellman equations,
  RLHF, and a proposed DQN + LLM integration architecture
- **Fully reproducible** — global seed set across Python, NumPy, and PyTorch

---

## 🛠 Tech Stack

| Tool | Version | Role |
|---|---|---|
| Python | 3.9+ | Core language |
| PyTorch | 2.2.2 | Neural network, autograd, optimiser |
| Gymnasium | 1.1.1 | LunarLander-v3 environment (Box2D) |
| NumPy | 1.26.4 | Array operations, replay buffer |
| Matplotlib | 3.8.4 | Training plots and dashboards |
| PyYAML | 6.0.1 | Hyperparameter config files |

> **Tested with:** Python 3.11, macOS Sequoia (arm64) and Ubuntu 22.04 (x86_64).
> Gymnasium 1.1.1 is required — earlier versions (≤ 0.29.x) do not include
> `LunarLander-v3` and will raise `VersionNotFound`.


---

## 📁 Project Structure

```
deep-q-learning-atari-agent/
├── src/
│   ├── model.py              # DQN MLP: Linear(8 → 128 → 128 → 4)
│   ├── replay_buffer.py      # Experience replay (deque, 50k capacity)
│   ├── train.py              # Baseline training loop + CLI entry point
│   ├── test_env.py           # Environment smoke test
│   ├── agent/
│   │   └── dqn_agent.py      # DQNAgent: epsilon-greedy, learn(), save/load
│   ├── env/
│   │   └── reward_shaping.py # ShapedLunarLander wrapper (leg-contact bonus)
│   ├── experiments/
│   │   ├── engine.py         # Shared training engine (pluggable action selector)
│   │   ├── exploration.py    # softmax_action() + epsilon_greedy_action()
│   │   ├── plot_utils.py     # Multi-run comparison plots
│   │   └── run_all.py        # Master experiment runner
│   ├── training/
│   │   ├── train.py          # Config-driven training loop
│   │   └── evaluate.py       # Load checkpoint + greedy evaluation
│   └── utils/
│       ├── logger.py         # Python logging + CSV TrainingLogger
│       ├── metrics.py        # compute_metrics(), save_metrics_csv()
│       ├── plot.py           # plot_training_rewards(), plot_full_dashboard()
│       └── seed.py           # set_global_seed() — full reproducibility
├── experiments/
│   ├── metrics/              # CSV training logs
│   ├── plots/                # Experiment comparison PNGs
│   ├── baseline_run/         # Baseline run outputs
│   └── exp_*.yaml            # Hyperparameter experiment configs
├── models/                   # Saved .pth checkpoints
├── docs/                     # Theory writeup, environment/reward analysis
├── config.yaml               # All hyperparameters in one place
├── requirements.txt
├── LICENSE
├── ATTRIBUTION.md
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.9+
- `swig` (required by Box2D physics engine)

```bash
# macOS
brew install swig

# Ubuntu / Debian
sudo apt-get install swig
```

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/PragatiAN1109/deep-q-learning-atari-agent.git
cd deep-q-learning-atari-agent

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify the environment works
python src/test_env.py
```

Expected output from Step 4:
```
✅  Environment is working correctly.
```

> **Important:** This project requires `gymnasium>=1.0` for `LunarLander-v3`.
> Running with `gymnasium==0.29.x` will raise `VersionNotFound`.

---

## ▶️ How to Run

### Baseline Training

```bash
# Train with all baseline defaults (500 episodes):
python src/train.py

# Override specific parameters:
python src/train.py --episodes 300 --lr 0.001 --gamma 0.95
python src/train.py --episodes 500 --out-dir experiments/my_run

# See all options:
python src/train.py --help
```

### Config-Driven Training

```bash
# Train using config.yaml (600 episodes):
python src/training/train.py

# Use a custom experiment config:
python src/training/train.py --config experiments/exp_low_lr.yaml
```

### Evaluate a Saved Model

```bash
python src/training/evaluate.py --model models/dqn_best.pth
python src/training/evaluate.py --model models/dqn_best.pth --episodes 20 --render
```

---

## 🧪 Running Experiments

```bash
# Run all 3 experiment tasks (α/γ sweep, exploration, epsilon decay):
python src/experiments/run_all.py

# Quick smoke test (50 episodes each):
python src/experiments/run_all.py --episodes 50

# Run a specific task only:
python src/experiments/run_all.py --task 1   # learning rate + gamma sweep
python src/experiments/run_all.py --task 2   # epsilon-greedy vs softmax
python src/experiments/run_all.py --task 3   # epsilon decay rate comparison
```

Outputs appear in:
```
experiments/metrics/          ← CSV training logs
experiments/plots/            ← Reward curves and comparison charts
```

---

## 📊 Results

### Environment

| Property | Value |
|---|---|
| Environment | `LunarLander-v3` (Gymnasium 1.1.1) |
| Observation | 8-dim continuous vector |
| Action space | Discrete(4): do nothing / fire left / fire main / fire right |
| Reward signal | +100–140 landing, −100 crash, fuel penalties per step |
| Solved threshold | Avg reward ≥ 200 over 100 consecutive episodes |

### Reward Curve

> *Run `python src/train.py` to generate — plot saved to
> `experiments/plots/reward_curve.png`*

```
experiments/plots/
├── reward_curve.png           ← Episode rewards + 100-ep rolling average
├── dashboard.png              ← 3-panel: rewards, rolling avg, MSE loss
├── reward_hp_lr_sweep.png     ← Learning rate α comparison
├── reward_hp_gamma_sweep.png  ← Discount factor γ comparison
├── reward_exploration.png     ← Epsilon-greedy vs Softmax
└── epsilon_decay_grid.png     ← Epsilon decay rate comparison
```

### Hyperparameter Findings

| Parameter | Tested Values | Best | Rationale |
|---|---|---|---|
| Learning rate α | 0.0001, 0.0005, 0.001 | **0.0005** | Standard Adam LR for DQN |
| Discount γ | 0.90, 0.95, 0.99 | **0.99** | γ=0.80 makes landing bonus invisible (0.8^200 ≈ 0) |
| Epsilon decay | 0.015, 0.005, 0.002 | **0.005** | Balanced exploration / exploitation |
| Exploration | Epsilon-greedy, Softmax τ=1.0/0.5 | **Epsilon-greedy** | Proven convergence via scheduled decay |

---

## 🏗️ DQN Architecture

```
Input: state ∈ ℝ^8  (x, y, vx, vy, angle, ang_vel, leg_L, leg_R)
         │
    Linear(8 → 128) + ReLU
         │
    Linear(128 → 128) + ReLU
         │
    Linear(128 → 4)
         │
Output: Q-values ∈ ℝ^4  (one per discrete action)
```

**Key DQN innovations:**
- **Experience Replay** — random mini-batches from a 50k-transition buffer
- **Target Network** — hard-copied every 10 episodes for stable Bellman targets
- **Gradient Clipping** — `max_norm=1.0` prevents exploding updates in early training

**Bellman update (per training step):**
```
target = r  +  γ · max_a' Q_target(s', a')  ·  (1 - done)
loss   = MSE( Q_online(s, a_taken),  target )
```

---

## 🔧 Configuration

All hyperparameters live in `config.yaml` — no source code changes needed.

```yaml
environment:
  name: "LunarLander-v3"
  seed: 42

agent:
  learning_rate: 0.0005
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay: 0.995
  batch_size: 64
  replay_buffer_size: 50000
  target_update_freq: 10
  hidden_size: 128
```

---

## 🚀 Future Improvements

- [ ] **Double DQN** — use online net to select action, target net to evaluate
- [ ] **Dueling DQN** — separate value and advantage streams
- [ ] **Prioritised Experience Replay** — weight high-TD-error transitions
- [ ] **Noisy Networks** — learnable noise layers to replace epsilon-greedy
- [ ] **Video recording** — save MP4 of trained agent via `imageio-ffmpeg`
- [ ] **TensorBoard integration** — `SummaryWriter` alongside CSV logging
- [ ] **Continuous action space** — extend to `LunarLander-v3` continuous mode

---

## 📄 Documentation

| Document | Description |
|---|---|
| [`docs/environment_analysis.md`](docs/environment_analysis.md) | State space, action space, Q-table feasibility |
| [`docs/reward_analysis.md`](docs/reward_analysis.md) | Reward structure, shaping strategies |
| [`docs/experiments.md`](docs/experiments.md) | Hyperparameter sweep design and results |
| [`docs/experiment_analysis.md`](docs/experiment_analysis.md) | Baseline experiment hypotheses |
| [`docs/theory_writeup.md`](docs/theory_writeup.md) | DQN theory, RLHF, DQN+LLM architecture |
| [`ATTRIBUTION.md`](ATTRIBUTION.md) | Code attribution and research references |

---

## 📝 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Attribution

See [ATTRIBUTION.md](ATTRIBUTION.md) for a full breakdown of original code,
adapted algorithms, external libraries, and research references.

Key references:
- Mnih et al. (2015). *Human-level control through deep reinforcement learning.* Nature.
- Sutton & Barto (2018). *Reinforcement Learning: An Introduction* (2nd ed.).
- Gymnasium: https://gymnasium.farama.org
