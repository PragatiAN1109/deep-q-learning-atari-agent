# Reward Structure Analysis — LunarLander-v3

## 1. Overview

This document analyses the reward function of **LunarLander-v3** as it
operates in this repo — how rewards are given, when penalties are applied,
why the structure works for DQN training, its limitations, and two concrete
reward shaping strategies (one implemented).

---

## 2. Native Reward Structure

LunarLander-v3 provides a **dense, shaped reward** — the agent receives a
non-zero scalar reward at almost every step, not just at the end of the
episode. This is in contrast to sparse-reward environments (like MountainCar)
where reward only arrives at the goal.

### 2.1 Reward Components — Full Breakdown

| Event | Reward | When it occurs |
|---|---|---|
| Moving toward landing pad | `+` small | Every step (based on distance delta) |
| Moving away from landing pad | `−` small | Every step |
| Reducing speed | `+` small | Every step (proportional to velocity reduction) |
| Increasing speed | `−` small | Every step |
| Correct orientation (upright) | `+` small | Every step |
| Tilted / rotating | `−` small | Every step |
| Left leg touches ground | `+10` | Once per contact event |
| Right leg touches ground | `+10` | Once per contact event |
| **Successful landing** | `+100` to `+140` | Terminal — episode ends |
| **Crash** (body hits ground) | `−100` | Terminal — episode ends |
| Firing main engine | `−0.3` per frame | Every step engine is active |
| Firing left/right engine | `−0.03` per frame | Every step engine is active |

### 2.2 Total Episode Reward Ranges

| Outcome | Typical range |
|---|---|
| Perfect landing (efficient, upright) | +200 to +300 |
| Clumsy landing (lots of fuel used) | +100 to +200 |
| Soft crash (legs touch, body hits) | −50 to +50 |
| Hard crash | −100 to −200 |
| Random policy (untrained) | −200 to −100 |

The **solve threshold** in this repo is **avg reward ≥ 200** over 100
consecutive episodes — set in `config.yaml` as `solve_score: 200.0`.

---

## 3. How Rewards Flow Through This Repo

In `src/train.py` (and `src/training/train.py`), rewards are collected
per step and accumulated:

```python
next_state, reward, terminated, truncated, _ = env.step(action)
episode_reward += reward                       # accumulate each step

# Stored raw in the replay buffer:
replay_buf.push(state, action, reward, next_state, done)
```

In `src/agent/dqn_agent.py`, the Bellman equation uses the raw reward
directly:
```python
# r is the raw per-step reward from the environment
q_target = rewards_t + gamma * q_next_max * (1.0 - dones_t)
```

No reward clipping or normalisation is applied — the raw signal from
the environment is used as-is.

---

## 4. Why This Reward Structure Works

### 4.1 Dense Signal — No Cold Start Problem
Because the agent receives small positive/negative rewards at every step
(not just on landing), it gets a learning signal even in early episodes
when it never reaches the pad. A random agent scoring −150 is still
receiving gradient information at every step about which directions are
better or worse.

Compare this to sparse reward environments where a random agent may
receive zero reward for thousands of episodes — making early learning
nearly impossible without exploration bonuses.

### 4.2 Fuel Penalty Encourages Efficiency
The per-step engine costs (`−0.3` main, `−0.03` side) prevent the agent
from hovering indefinitely. Without this penalty, an agent that learned
to hover above the pad would score high without ever landing.
The penalty creates pressure to land quickly and efficiently.

### 4.3 Terminal Rewards Create a Clear Goal
The `+100`–`+140` landing bonus and `−100` crash penalty are large enough
to dominate the episode total. This means the agent cannot reach a high
average score without actually landing — the terminal signals anchor
the learning toward the true objective.

### 4.4 Compatibility with Gamma = 0.99
This repo uses `gamma = 0.99` in `config.yaml`. With dense rewards and a
max episode length of 1000 steps, high gamma is essential — a landing
reward received 200 steps in the future must still be discounted to a
meaningful value:
```
0.99^200 ≈ 0.134
```
At `gamma = 0.8` (the assignment's original value):
```
0.8^200 ≈ 0.000000000000000 (effectively zero)
```
This is why `gamma = 0.8` was changed to `0.99` in this repo — at 0.8,
the agent would be completely myopic and unable to plan beyond ~10 steps.

---

## 5. Limitations of the Current Reward Structure

### 5.1 Delayed Landing Signal
The large terminal reward (+100 to +140) is only given at the very end
of an episode. In a 400-step episode, that reward must propagate backward
through 400 Bellman updates before it influences early-episode Q-values.
This is the core challenge of temporal credit assignment in RL.

With `gamma=0.99`, the terminal reward is discounted by `0.99^400 ≈ 0.018`
when viewed from step 0 — meaning early-episode decisions are barely
influenced by whether the lander eventually lands or crashes.

### 5.2 Reward Scale Imbalance
The terminal rewards (±100) are ~300× larger than the per-step rewards
(~±0.3 per step). This creates noisy gradients when the replay buffer
contains a mix of terminal and non-terminal transitions. A batch of 64
samples may include 1–2 terminal transitions with very large Bellman errors
that dominate the gradient update.

### 5.3 No Guidance on Intermediate Progress
The reward does not explicitly guide the agent through a sequence of
sub-goals (e.g., "get above the pad → reduce speed → align angle →
lower legs → touch down"). The agent must discover this entire sequence
purely from the dense-but-noisy step rewards.

---

## 6. Reward Shaping Strategies

### Strategy 1 — Leg Contact Bonus (Recommended, Safe)
Add a small supplementary bonus when both legs make simultaneous contact
with the ground. This rewards partial progress toward landing without
requiring a full successful episode.

```
shaped_reward = raw_reward + 5.0 * (obs[6] + obs[7])
```
where `obs[6]` and `obs[7]` are the left/right leg contact flags (0 or 1).

**Why it helps:** Encourages the agent to keep legs on the ground longer,
increasing the chance of a stable landing. The native `+10` per leg is
already present, but the shaped bonus makes it more immediate.

**Risk:** Low — the bonus is proportional to a clear binary signal.
Does not interfere with the crash/landing terminal signals.

### Strategy 2 — Velocity Penalty Near Ground (Advanced, Optional)
When the lander is close to the ground (`obs[1] < 0.3`) and moving fast
(`|obs[3]| > 1.0`), add a penalty:

```
if obs[1] < 0.3 and abs(obs[3]) > 1.0:
    shaped_reward -= 5.0 * abs(obs[3])
```

**Why it helps:** Explicitly penalises fast approaches to the ground,
making the agent learn to slow down before landing. Addresses the delay
in the crash penalty signal.

**Risk:** Medium — incorrect threshold values can make training unstable.
Requires tuning.

---

## 7. Implementation — Strategy 1 (Leg Contact Bonus)

A safe, optional reward shaping wrapper is provided in
`src/env/reward_shaping.py` (see next section). It does NOT modify
any existing training files — it is an optional drop-in wrapper.

To use it, wrap the environment before training:
```python
from src.env.reward_shaping import ShapedLunarLander
env = ShapedLunarLander(gym.make("LunarLander-v3"))
```
All other training code remains identical.

---

## 8. How Reward Trends Reflect Learning

This repo logs per-episode metrics to `experiments/baseline_run/training_metrics.csv`
with columns: `episode, reward, steps, epsilon, moving_avg_50, moving_avg_100`.

### Expected learning curve phases:

**Phase 1 — Exploration (episodes 1–100):**
- Avg reward: `−200` to `−100`
- Agent fires engines randomly, usually crashes immediately
- Replay buffer fills with crash transitions
- Loss starts high and noisy as Q-values adjust from random initialisation

**Phase 2 — Early Learning (episodes 100–250):**
- Avg reward rises from `−100` toward `0`
- Agent learns not to crash immediately (crash penalty = −100 dominates)
- Leg contact bonuses (+10 each) occasionally appear in logs
- Loss stabilises as Q-values become more consistent

**Phase 3 — Policy Improvement (episodes 250–450):**
- Avg reward: `0` to `+150`
- Agent learns to hover near the pad and reduce speed
- Steps per episode increase (agent stays alive longer)
- Moving average (50-ep) in CSV shows clear upward trend

**Phase 4 — Convergence (episodes 450–500+):**
- Avg100 approaches and may exceed +200 (solve threshold)
- Agent reliably lands with both legs
- Epsilon near `epsilon_min` — mostly exploitation
- Loss is low and stable

### Reading the CSV log:
```python
import pandas as pd
df = pd.read_csv("experiments/baseline_run/training_metrics.csv")

# Check if solved:
print(df[df["moving_avg_100"] >= 200].head(1))

# Plot reward trend:
df.plot(x="episode", y=["reward", "moving_avg_50"], figsize=(10,4))
```

---

## 9. Summary

| Aspect | Assessment |
|---|---|
| Reward type | Dense — signal at every step |
| Terminal bonus | +100 to +140 (landing), −100 (crash) |
| Fuel penalty | −0.3/step (main), −0.03/step (side) |
| Reward sparsity | Low — good for learning |
| Credit assignment difficulty | Medium — terminal rewards delayed 200–400 steps |
| Gamma recommendation | 0.99 (used in this repo) ✅ |
| Shaping implemented | Strategy 1: leg contact bonus (optional wrapper) |
| Training signal quality | Strong — suitable for DQN without modification |
