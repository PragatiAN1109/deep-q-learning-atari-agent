# Code Attribution

This document transparently identifies which parts of this project were
written from scratch, adapted from published resources, and which external
references influenced the design and implementation.

---

## 1. Original Work

The following components were **designed and implemented from scratch**
by **Pragati Narote** for this project:

| Component | File(s) | Description |
|---|---|---|
| Full DQN training loop | `src/train.py` | Episode/step loop, epsilon decay, CSV logging |
| Experiment runner | `src/experiments/run_all.py` | Hyperparameter sweep framework |
| Pluggable exploration engine | `src/experiments/engine.py` | Supports swappable action selectors |
| Softmax / Boltzmann exploration | `src/experiments/exploration.py` | Alternative to epsilon-greedy |
| Reward shaping wrapper | `src/env/reward_shaping.py` | Leg-contact bonus for LunarLander-v3 |
| Metrics utility | `src/utils/metrics.py` | compute_metrics(), save_metrics_csv(), print_metrics_table() |
| Structured logging system | `src/utils/logger.py` | Python logging + CSV via TrainingLogger |
| Training dashboard plot | `src/utils/plot.py` | 3-panel reward/avg/loss figure |
| Environment screenshot script | `scripts/capture_env_screenshot.py` | Headless render using rgb_array |
| Real training + plot script | `scripts/train_and_plot.py` | Self-contained training + CSV + figure |
| All documentation write-ups | `docs/` | Environment, reward, experiment, theory analysis |
| Experiment YAML configs | `experiments/exp_*.yaml` | Hyperparameter experiment configs |

---

## 2. Adapted / Inspired Code

The following components were **adapted from well-known published sources**
with significant modifications:

### 2.1 DQN Model Architecture (`src/model.py`)

- **Source:** PyTorch official DQN tutorial
  https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
- **Original:** CNN-based network for CartPole pixel observations
- **Adaptation:** Replaced CNN with a 3-layer MLP (Linear 8→128→128→4)
  to match LunarLander-v3's 8-dimensional vector observation space.
  All layer dimensions, activation choices, and forward-pass logic were
  written independently to match this project's state/action space.

### 2.2 Replay Buffer (`src/replay_buffer.py`)

- **Source 1:** Mnih et al. (2015). *Human-level control through deep
  reinforcement learning.* Nature, 518, 529–533.
  — Original experience replay concept.
- **Source 2:** OpenAI Spinning Up implementation pattern
  https://spinningup.openai.com/en/latest/
  — Circular buffer design using `collections.deque`.
- **Adaptation:** Implemented independently using Python's `deque(maxlen)`.
  The `is_ready()` guard method, typed NumPy output dtypes (`float32`/`int64`),
  and the `__repr__` method are original additions not present in either source.

### 2.3 DQN Agent — Two-Network Design (`src/agent/dqn_agent.py`)

- **Source 1:** Mnih et al. (2015) — online + target network concept.
- **Source 2:** van Hasselt et al. (2016). *Deep Reinforcement Learning
  with Double Q-learning.* AAAI.
- **Adaptation:** Hard-copy target sync, gradient clipping (`max_norm=1.0`),
  and checkpoint I/O (save/load) were implemented independently.
  The Bellman update logic follows the standard equation but all PyTorch
  tensor operations were written from scratch.

### 2.4 Epsilon-Greedy Exploration

- **Source:** Sutton & Barto, *Reinforcement Learning: An Introduction*
  (2nd ed., 2018), Chapter 2.
- **Note:** This is a standard algorithm — no code was copied.
  The exponential decay formula `ε = max(ε_min, exp(−rate × episode))`
  matches the assignment specification.

### 2.5 Softmax / Boltzmann Exploration (`src/experiments/exploration.py`)

- **Source:** Sutton & Barto (2018), Chapter 2.
- **Note:** Standard algorithm. The numerically stable softmax trick
  (subtract max before exp) is a common numerical methods practice.
  No code was copied from any source.

---

## 3. External Libraries

All libraries used are open-source and listed with their licenses:

| Library | Version | License | Purpose |
|---|---|---|---|
| [Gymnasium](https://gymnasium.farama.org/) | 1.1.1 | MIT | LunarLander-v3 environment |
| [PyTorch](https://pytorch.org/) | 2.2.2 | BSD-3-Clause | Neural network, autograd |
| [NumPy](https://numpy.org/) | 1.26.4 | BSD-3-Clause | Array operations |
| [Matplotlib](https://matplotlib.org/) | 3.8.4 | PSF | Training plots and diagrams |
| [PyYAML](https://pyyaml.org/) | 6.0.1 | MIT | Config file parsing |
| [imageio](https://imageio.readthedocs.io/) | 2.34.1 | BSD-2-Clause | Video recording |

---

## 4. Key Research References

1. **Mnih, V. et al.** (2015). Human-level control through deep reinforcement
   learning. *Nature*, 518, 529–533.
   https://www.nature.com/articles/nature14236
   — Foundational DQN paper: experience replay + target network.

2. **van Hasselt, H., Guez, A., & Silver, D.** (2016). Deep Reinforcement
   Learning with Double Q-learning. *AAAI*.
   — Double DQN concept referenced in architecture design.

3. **Sutton, R. S., & Barto, A. G.** (2018). *Reinforcement Learning:
   An Introduction* (2nd ed.). MIT Press.
   http://incompleteideas.net/book/the-book-2nd.html
   — Foundational RL theory: Bellman equations, epsilon-greedy, Q-learning.

4. **Ouyang, L. et al.** (2022). Training language models to follow
   instructions with human feedback. *NeurIPS*.
   — RLHF section of the theory writeup.

5. **Farama Foundation** (2023). Gymnasium Documentation.
   https://gymnasium.farama.org/environments/box2d/lunar_lander/
   — LunarLander-v3 environment specification and reward structure.

---

## 5. Academic Integrity Statement

All code in this repository was written or substantially adapted by the
author for this graded assignment. Any adapted logic is explicitly cited
above with the original source. No code was copied verbatim from online
solutions, peer submissions, or generative AI outputs without disclosure.
The implementations draw on published algorithms and standard library
patterns, all of which are credited in this document.
