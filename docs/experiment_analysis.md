# DQN Experiment Analysis — LunarLander-v2

## 1. Overview

This document describes the experimental design, hypotheses, and expected
observations for the DQN hyperparameter study conducted on the
**LunarLander-v2** environment.

Each experiment changes **one parameter** relative to the baseline to isolate
the effect of that parameter on learning performance.

---

## 2. Environment Summary

| Property         | Value                                             |
|------------------|---------------------------------------------------|
| Environment      | `LunarLander-v2` (Gymnasium)                      |
| Observation      | 8-dim vector: x/y pos, x/y vel, angle, ang vel, leg contacts |
| Action Space     | Discrete(4): do nothing, fire left, fire main, fire right |
| Reward Signal    | +100–140 for landing, −100 crash, fuel penalties  |
| Solved Threshold | Avg reward ≥ 200 over 100 consecutive episodes    |

---

## 3. DQN Architecture

```
Input:  (batch, 8)
         │
    Linear(8 → 128) + ReLU
         │
    Linear(128 → 128) + ReLU
         │
    Linear(128 → 4)
         │
Output: (batch, 4)  ← Q-value for each action
```

Two networks are maintained:
- **Online network** — trained every step via Bellman MSE loss
- **Target network** — synced from online every 10 episodes; provides stable bootstrap targets

---

## 4. Baseline Configuration

| Parameter              | Value   |
|------------------------|---------|
| Learning rate          | 0.0005  |
| Gamma (discount)       | 0.99    |
| Epsilon start          | 1.0     |
| Epsilon end            | 0.05    |
| Epsilon decay          | 0.995   |
| Batch size             | 64      |
| Replay buffer capacity | 50,000  |
| Target update freq     | 10 eps  |
| Hidden layer size      | 128     |
| Max episodes           | 600     |

Epsilon reaches its minimum (~0.05) after approximately **300 episodes**
with decay = 0.995: `1.0 × 0.995^300 ≈ 0.22` → reaches 0.05 around ep 580.

---

## 5. Experiments

### Experiment 1 — Baseline
**Config:** `experiments/exp_baseline.yaml`

All default hyperparameters. This is the control experiment that every other
run is compared against.

**Expected behaviour:**
- Agent starts randomly, accumulates experience in the replay buffer
- After ~100 episodes, the buffer has enough diverse samples for useful updates
- Reward begins rising steadily after epsilon falls below ~0.5 (~ep 140)
- Agent should approach or exceed the solve threshold by episode 500–600

---

### Experiment 2 — Low Learning Rate
**Config:** `experiments/exp_low_lr.yaml`
**Changed:** `learning_rate: 0.0001` (baseline: 0.0005)

**What this tests:**
The learning rate controls the step size of each gradient update.
A smaller LR means the Q-network weights change more slowly per update.

**Hypothesis:**
- Slower convergence — will take more episodes to see improvement
- Potentially more stable final policy if baseline LR caused overshooting
- If the network oscillates with LR=0.0005, LR=0.0001 should be smoother

**What to look for in the plot:**
- Does the reward curve rise more slowly but reach a similar or higher ceiling?
- Is the reward curve less noisy (lower variance per episode)?
- Does the agent need all 600 episodes to converge, or does it stall entirely?

---

### Experiment 3 — Fast Epsilon Decay
**Config:** `experiments/exp_fast_epsilon.yaml`
**Changed:** `epsilon_decay: 0.990` (baseline: 0.995)

**What this tests:**
Epsilon controls the exploration-exploitation trade-off.
With decay = 0.990, epsilon reaches its minimum after ~150 episodes
(vs ~580 for baseline). The agent commits to exploitation much sooner.

**Hypothesis:**
- Initial reward rise will be steeper — the agent exploits its early policy faster
- If the early policy is poor, the agent gets stuck in a local optimum
- Expected: faster early gains but a lower performance ceiling than baseline

**Epsilon schedule comparison:**

| Episode | Baseline (0.995) | Fast (0.990) |
|---------|-----------------|--------------|
| 50      | 0.78            | 0.61         |
| 100     | 0.61            | 0.37         |
| 150     | 0.47            | 0.22         |
| 200     | 0.37            | 0.13         |
| 300     | 0.22            | 0.05 (min)   |

**What to look for in the plot:**
- Does the reward peak earlier but plateau lower?
- Is there a "cliff" in improvement once epsilon hits its minimum?

---

### Experiment 4 — Slow Epsilon Decay
**Config:** `experiments/exp_slow_epsilon.yaml`
**Changed:** `epsilon_decay: 0.998` (baseline: 0.995)

**What this tests:**
With decay = 0.998, epsilon stays above 0.05 for nearly all 600 episodes,
keeping the agent in near-random exploration mode throughout training.

**Hypothesis:**
- Slower convergence — agent barely exploits its policy
- Replay buffer stays diverse throughout training
- May outperform fast-decay if the environment has sparse or deceptive rewards
  that require thorough exploration to discover good strategies
- For LunarLander-v2 (dense rewards), we expect this to underperform baseline

**What to look for in the plot:**
- Does reward stay low for much longer before climbing?
- Does it eventually catch up to baseline if given enough episodes?
- Compare variance — more exploration may produce more stable final policy

---

## 6. Metrics Tracked Per Experiment

Each training run produces `experiments/<name>/training_log.csv` with:

| Column           | Description                                         |
|------------------|-----------------------------------------------------|
| `episode`        | Episode number (1-indexed)                          |
| `total_reward`   | Raw cumulative reward for this episode              |
| `avg_reward_100` | Rolling mean over last 100 episodes                 |
| `epsilon`        | Exploration rate at end of episode                  |
| `steps`          | Number of env steps taken in this episode           |
| `loss`           | Mean Bellman MSE loss across all learning steps     |

---

## 7. How to Reproduce Results

```bash
# Run all 4 experiments sequentially (takes ~30–60 min on CPU):
python src/training/run_experiments.py

# Run just one experiment:
python src/training/train.py --config experiments/exp_baseline.yaml

# Re-generate comparison plot without re-training:
python src/training/run_experiments.py --plot-only

# Evaluate the best saved model:
python src/training/evaluate.py --model models/baseline/dqn_best.pth
```

---

## 8. Expected Results Summary

| Experiment      | Convergence Speed | Final Avg Reward | Stability |
|-----------------|:-----------------:|:----------------:|:---------:|
| Baseline        | Medium (~300 ep)  | ~200 (target)    | Medium    |
| Low LR          | Slow  (~400 ep)   | Similar/Higher   | High      |
| Fast ε decay    | Fast  (~150 ep)   | Lower            | Low       |
| Slow ε decay    | Very slow         | Lower            | Medium    |

> **Note:** These are pre-run hypotheses. Actual results will be filled in
> after training completes. Update this table with measured values.

---

## 9. Key DQN Concepts Referenced

**Experience Replay:** Stores past transitions and samples randomly to break
temporal correlation, stabilising gradient updates.

**Target Network:** A frozen copy of the Q-network used to compute Bellman
targets. Prevents the "chasing a moving target" problem that causes divergence.

**Epsilon-Greedy:** Balances exploration (random actions) with exploitation
(greedy Q-value actions). Epsilon decays over training as the policy improves.

**Bellman Update:**
```
target = r + γ · max_a Q_target(s', a) · (1 - done)
loss   = MSE(Q_online(s, a_taken),  target)
```

---

## 10. References

- Mnih et al. (2015). *Human-level control through deep reinforcement learning.* Nature.
- Gymnasium Documentation: https://gymnasium.farama.org/environments/box2d/lunar_lander/
- PyTorch DQN Tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
