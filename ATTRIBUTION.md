# Code Attribution

This document transparently identifies which parts of this project were written
from scratch, which were adapted from published resources, and which external
references influenced the design.

---

## Original Work

The following components were designed and implemented from scratch by
**Pragati Narote** for this project:

| Component | File(s) |
|---|---|
| Full DQN training loop with episode/step structure | `src/train.py` |
| Experiment runner and hyperparameter sweep framework | `src/experiments/run_all.py` |
| Pluggable exploration engine (softmax + epsilon-greedy) | `src/experiments/engine.py`, `src/experiments/exploration.py` |
| Reward shaping wrapper for LunarLander-v2 | `src/env/reward_shaping.py` |
| Metrics utility (compute, save, print) | `src/utils/metrics.py` |
| Structured logging combining Python logging + CSV | `src/utils/logger.py` |
| Three-panel training dashboard plot | `src/utils/plot.py` |
| All documentation and analysis write-ups | `docs/` |
| Experiment configuration YAML files | `experiments/exp_*.yaml` |
| Environment analysis and reward analysis | `docs/environment_analysis.md`, `docs/reward_analysis.md` |
| Theory and design writeup (DQN + LLM integration) | `docs/theory_writeup.md` |

---

## Adapted / Inspired Code

The following components were adapted from well-known sources with modifications:

### DQN Model Architecture (`src/model.py`)
- **Inspired by:** PyTorch official DQN tutorial
  https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
- **Adaptation:** Changed from CNN (used for CartPole pixel observations) to
  MLP for LunarLander-v2 vector observations. All layer sizes, activation
  choices, and forward-pass logic were written independently to match this
  project's state/action dimensions.

### Replay Buffer (`src/replay_buffer.py`)
- **Inspired by:** Standard experience replay as described in:
  Mnih et al. (2015). *Human-level control through deep reinforcement learning.* Nature.
  and the OpenAI Spinning Up implementation:
  https://spinningup.openai.com/en/latest/
- **Adaptation:** Implemented from scratch using Python's `collections.deque`
  rather than copying any existing code. The `is_ready()` guard method and
  typed NumPy output dtypes (`float32`/`int64`) are original additions.

### DQN Agent (`src/agent/dqn_agent.py`)
- **Inspired by:** The two-network (online + target) architecture from:
  Mnih et al. (2015) and van Hasselt et al. (2016) *Deep Reinforcement Learning
  with Double Q-learning.*
- **Adaptation:** The hard-copy target sync, gradient clipping (`max_norm=1.0`),
  and checkpoint I/O were implemented independently. The Bellman update logic
  follows the standard equation but all PyTorch tensor manipulation was written
  from scratch.

### Epsilon-Greedy Exploration
- **Standard algorithm** described in Sutton & Barto,
  *Reinforcement Learning: An Introduction* (2nd ed., 2018), Chapter 2.
- No code was copied; the implementation directly follows the standard formula.

### Softmax / Boltzmann Exploration (`src/experiments/exploration.py`)
- **Standard algorithm** described in Sutton & Barto, Chapter 2.
- The numerically stable softmax trick (subtract max before exp) is a
  well-known numerical methods practice, not original to any single source.

---

## External Libraries

This project uses the following open-source libraries, each under their
respective licenses:

| Library | Version | License | Purpose |
|---|---|---|---|
| [Gymnasium](https://gymnasium.farama.org/) | 0.29.1 | MIT | LunarLander-v2 environment |
| [PyTorch](https://pytorch.org/) | 2.2.2 | BSD-3-Clause | Neural network, autograd |
| [NumPy](https://numpy.org/) | 1.26.4 | BSD-3-Clause | Array operations |
| [Matplotlib](https://matplotlib.org/) | 3.8.4 | PSF | Training plots |
| [PyYAML](https://pyyaml.org/) | 6.0.1 | MIT | Config file parsing |
| [imageio](https://imageio.readthedocs.io/) | 2.34.1 | BSD-2-Clause | Video recording |

---

## Key Research References

- Mnih, V. et al. (2015). Human-level control through deep reinforcement
  learning. *Nature*, 518, 529–533.
  https://www.nature.com/articles/nature14236

- van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning
  with Double Q-learning. *AAAI*.

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An
  Introduction* (2nd ed.). MIT Press.
  http://incompleteideas.net/book/the-book-2nd.html

- Ouyang, L. et al. (2022). Training language models to follow instructions
  with human feedback. *NeurIPS*. (RLHF section of theory writeup)

---

## Academic Integrity Statement

All code in this repository was written or substantially adapted by the author
for this graded assignment. Any adapted logic is clearly cited above.
No code was copied verbatim from online solutions or peer submissions.
The implementations draw on published algorithms and standard library patterns,
which are credited in this document.
