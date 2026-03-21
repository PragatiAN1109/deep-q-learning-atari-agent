# Environment Analysis — LunarLander-v2

## 1. Overview

This document provides a complete technical analysis of the environment
used in this DQN project: **LunarLander-v2** from the
[Gymnasium](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
`box2d` suite.

Every detail below is tied directly to how this repo's code
(`src/model.py`, `src/replay_buffer.py`, `src/agent/dqn_agent.py`,
`src/train.py`) interacts with the environment.

---

## 2. Why LunarLander-v2?

| Criterion | LunarLander-v2 | Raw Atari (e.g. Breakout) |
|---|---|---|
| Observation type | 8-dim float vector | 210×160×3 RGB image |
| Preprocessing needed | None | Grayscale, resize, frame stack |
| CPU training time | ~30–60 min | 4–12 hours |
| ROM licensing | None required | Separate ALE ROM install |
| Reward signal | Dense (every step) | Sparse (score changes only) |
| Suited for MLP | ✅ Yes | ❌ No (needs CNN) |

LunarLander-v2 keeps the focus on DQN architecture and learning dynamics
rather than on image preprocessing pipelines.

---

## 3. State Space

### Type
**Continuous** — the observation is a vector of real-valued floats,
not a discrete or image-based representation.

### Shape
`(8,)` — a 1-dimensional NumPy array with 8 float32 elements.

### Observation Vector — All 8 Dimensions

| Index | Variable | Range | Description |
|---|---|---|---|
| 0 | `x` position | ~[−1.5, 1.5] | Horizontal position of the lander |
| 1 | `y` position | ~[−1.5, 1.5] | Vertical position above the pad |
| 2 | `x` velocity | ~[−5, 5] | Horizontal speed |
| 3 | `y` velocity | ~[−5, 5] | Vertical speed (negative = falling) |
| 4 | angle | ~[−π, π] | Lander tilt in radians |
| 5 | angular velocity | ~[−5, 5] | Rate of rotation |
| 6 | left leg contact | {0.0, 1.0} | 1.0 if left leg touching ground |
| 7 | right leg contact | {0.0, 1.0} | 1.0 if right leg touching ground |

### Example Observation (at episode reset)
```
[ 0.0094,  1.4120, -0.0031, -0.0872, -0.0032,  0.0016,  0.0,  0.0 ]
  x_pos   y_pos   x_vel    y_vel    angle     ang_vel  L_leg R_leg
```
The lander starts near the top of the screen with near-zero velocity
and both legs off the ground.

### How the Repo Uses the State
In `src/model.py`, the DQN's input layer is sized to `state_dim=8`:
```python
nn.Linear(state_dim, hidden_size)   # Linear(8, 128)
```
In `src/train.py` and `src/agent/dqn_agent.py`, observations are
converted to `float32` tensors before being fed to the network:
```python
state = np.array(state, dtype=np.float32)   # shape: (8,)
state_t = torch.FloatTensor(state).unsqueeze(0)  # shape: (1, 8)
```

---

## 4. Action Space

### Type
**Discrete(4)** — the agent picks exactly one integer action per step
from the set `{0, 1, 2, 3}`.

### Action Meanings

| Action | Label | Effect |
|---|---|---|
| `0` | Do nothing | No engine fired; lander falls under gravity |
| `1` | Fire left engine | Pushes lander right, rotates clockwise |
| `2` | Fire main engine | Pushes lander upward, slows descent |
| `3` | Fire right engine | Pushes lander left, rotates counter-clockwise |

### How the Repo Uses Actions
The agent's output layer in `src/model.py` produces one Q-value per action:
```python
nn.Linear(hidden_size, action_dim)   # Linear(128, 4)
# Output shape: (batch_size, 4)
# q_values[i][a] = estimated return for action a in sample i
```
During **epsilon-greedy selection** in `src/train.py`:
```python
# Exploit: take the action with the highest Q-value
action = q_values.argmax(dim=1).item()   # integer in {0, 1, 2, 3}
```
During **learning** in `src/agent/dqn_agent.py`, `gather` selects only
the Q-value corresponding to the action actually taken:
```python
q_current = all_q.gather(1, actions_t.unsqueeze(1)).squeeze(1)
```

---

## 5. Q-Table Feasibility Analysis

### What a Q-table is
A tabular Q-function stores one Q-value for every `(state, action)` pair:
```
Q[state][action] = expected cumulative reward
```
This works perfectly for small discrete environments (e.g. FrozenLake-4×4
has 16 states × 4 actions = 64 entries).

### Why a Q-table is NOT feasible for LunarLander-v2

The state space is **continuous** — each of the 8 observation dimensions
can take infinitely many real values. To build a Q-table we would need to
discretise every dimension. Even with a coarse grid of just 10 bins per
dimension:

```
Q-table size = 10^8 × 4 = 400,000,000 entries
```

That is **400 million entries** for a very coarse approximation — and
the vast majority would never be visited during training, making
learning impossibly slow (the curse of dimensionality).

With finer resolution (100 bins per dim):
```
Q-table size = 100^8 × 4 = 40,000,000,000,000,000 entries
```
This is completely infeasible in memory or compute.

### How DQN Solves This

Instead of storing a table, DQN uses a **neural network as a function
approximator**:

```
Q(s, a; θ) ≈ Q*(s, a)
```

The network takes a raw state vector `s` as input and outputs Q-values
for all actions simultaneously. It generalises across similar states — if
the network has learned that hovering slowly with legs near the ground
has high value, it will assign similar Q-values to all nearby states,
even ones never seen before.

**This repo's Q-network** (`src/model.py`):
```
Input:  s ∈ ℝ^8   (continuous state vector)
         ↓
    Linear(8 → 128) + ReLU      ← learns state features
         ↓
    Linear(128 → 128) + ReLU    ← combines features
         ↓
    Linear(128 → 4)              ← one Q-value per action
Output: Q(s, ·) ∈ ℝ^4
```
Total trainable parameters: `8×128 + 128 + 128×128 + 128 + 128×4 + 4 = 18,308`
— trivially small, yet powerful enough to solve LunarLander-v2.

---

## 6. Episode Lifecycle

```
env.reset()
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Step loop (max 1000 steps in training/train.py)        │
│                                                         │
│  agent selects action (epsilon-greedy)                  │
│       ↓                                                 │
│  env.step(action)                                       │
│    → next_state (8-dim vector)                         │
│    → reward     (float)                                 │
│    → terminated (bool: crash or successful landing)     │
│    → truncated  (bool: step limit exceeded)             │
│       ↓                                                 │
│  done = terminated OR truncated                         │
│  if done: break                                         │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Episode ends. Metrics logged. Epsilon decayed.
```

**Termination conditions:**
- `terminated=True`: Lander crashed (body hits ground) OR landed successfully
- `truncated=True`: Episode hit the step limit (1000 steps in this repo)

This distinction matters for the Bellman update — see `src/train.py`:
```python
done = terminated or truncated
# In learning_step():
q_target = r + gamma * max_Q(s') * (1.0 - done)
# When done=1: no future reward is bootstrapped
```

---

## 7. Optional Utility — Print Environment Info

A lightweight utility is available in `src/test_env.py` to print
all space metadata without touching training code:

```bash
python src/test_env.py
```

Sample output:
```
──────────────────────────────────────────────────────────────
  Observation Space : Box([-1.5 -1.5 ...], [1.5 1.5 ...], (8,), float32)
  Observation Shape : (8,)
  Action Space      : Discrete(4)
  Number of Actions : 4
──────────────────────────────────────────────────────────────
```

---

## 8. Summary

| Property | Value |
|---|---|
| Environment ID | `LunarLander-v2` |
| Observation type | Continuous, float32 |
| Observation shape | `(8,)` |
| Action space | `Discrete(4)` |
| Q-table feasible? | ❌ No — infinite continuous state space |
| Q-network used | MLP: 8 → 128 → 128 → 4 |
| Trainable parameters | 18,308 |
| Episode max steps | 1,000 (this repo's config) |
| Solved threshold | Avg reward ≥ 200 over 100 consecutive episodes |
| Seed for reproducibility | 42 (set in `config.yaml`) |
