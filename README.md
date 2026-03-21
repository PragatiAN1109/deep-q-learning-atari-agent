# 🚀 Deep Q-Learning Atari Agent

A production-quality **Deep Q-Network (DQN)** agent that learns to play
[LunarLander-v2](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
using PyTorch and Gymnasium — built as a graded assignment and portfolio project
demonstrating applied reinforcement learning engineering.

---

## 🎯 Environment: LunarLander-v2

| Property | Value |
|---|---|
| Environment ID | `LunarLander-v2` |
| Observation | 8-dim continuous vector (position, velocity, angle, leg contacts) |
| Action Space | Discrete(4): do nothing / fire left / fire main / fire right engine |
| Reward Signal | +100–140 landing, −100 crash, small fuel penalties |
| Solved Threshold | Average reward ≥ 200 over 100 consecutive episodes |

**Why LunarLander-v2?**
Faster to train than pixel-based Atari ROMs (CPU-friendly), no ROM licensing
issues, clean vector observations that keep the focus on DQN architecture rather
than image preprocessing, and visually compelling results for demos.

---

## 📁 Project Structure

```
deep-q-learning-atari-agent/
├── src/
│   ├── agent/          # Q-network, replay buffer, DQN agent
│   ├── env/            # Environment factory and wrappers
│   ├── training/       # Training loop, evaluation, checkpointing
│   ├── utils/          # Plotting, seeding, logging helpers
│   └── test_env.py     # Environment smoke test (run first!)
├── models/             # Saved model checkpoints (.pth)
├── experiments/        # Training logs (CSV) and reward curves (PNG)
├── docs/               # Analysis write-ups and observations
├── video/              # Demo recordings
├── config.yaml         # ALL hyperparameters in one place
├── requirements.txt    # Pinned Python dependencies
└── README.md
```

---

## ⚙️ Setup

### Prerequisites
- Python 3.9+
- `swig` (required for Box2D physics engine)

```bash
# macOS
brew install swig

# Ubuntu / Debian
sudo apt-get install swig
```

### Installation

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
```

---

## ▶️ How to Run

### Step 1 — Verify environment setup
```bash
python src/test_env.py
```
Expected output: 1 random episode printed step-by-step, ending with
`✅ Environment is working correctly.`

### Step 2 — Train the DQN agent *(coming soon)*
```bash
python src/training/train.py
# Optional: point to a custom config
python src/training/train.py --config experiments/exp_low_lr.yaml
```

### Step 3 — Evaluate a saved model *(coming soon)*
```bash
python src/training/evaluate.py --model models/dqn_best.pth
```

---

## 🔧 Configuration

All hyperparameters live in `config.yaml` — no code changes needed for experiments.

| Parameter | Default | Description |
|---|---|---|
| `environment.name` | `LunarLander-v2` | Gymnasium environment ID |
| `environment.seed` | `42` | Global random seed |
| `agent.learning_rate` | `0.0005` | Adam optimizer LR |
| `agent.gamma` | `0.99` | Discount factor |
| `agent.epsilon_start` | `1.0` | Initial exploration rate |
| `agent.epsilon_decay` | `0.995` | Epsilon decay per episode |
| `agent.batch_size` | `64` | Replay buffer sample size |
| `agent.target_update_freq` | `10` | Episodes between target net syncs |
| `training.max_episodes` | `600` | Total training episodes |
| `training.solve_score` | `200.0` | Avg reward threshold for "solved" |

---

## 📊 Results

*Training reward curves and analysis will be populated after training runs complete.*

---

## 🧪 Experiment Branches

Each experiment is its own branch with documented changes and measured impact.

| Branch | Change | Status |
|---|---|---|
| `feature/env-setup-and-project-structure` | Repo scaffold + environment smoke test | ✅ Done |
| `feature/dqn-model` | Q-network, replay buffer, DQN agent class | 🔜 Next |
| `feature/training-loop` | Full training + evaluation pipeline | 🔜 |
| `feature/reward-analysis` | Training graphs and metrics | 🔜 |
| `feature/epsilon-decay` | Decay schedule comparison experiment | 🔜 |
| `feature/hyperparameter-tuning` | LR / batch size / gamma sweeps | 🔜 |

---

## 🏗️ Architecture (DQN)

```
State (8-dim)
     │
     ▼
┌─────────────────────┐
│  Q-Network (Online) │  ← trained every step
│  FC(128) → FC(128)  │
│  → Q-values (4)     │
└─────────────────────┘
     │                        ┌──────────────────────┐
     │  Experience Replay     │ Target Network       │
     │  Buffer (50k)    ───►  │ (soft-synced every   │
     │                        │  10 episodes)        │
     ▼                        └──────────────────────┘
  Bellman update:
  Q(s,a) ← r + γ · max Q_target(s', a')
```

---

## 📄 License

MIT — free to use, modify, and distribute with attribution.
