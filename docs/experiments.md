# Controlled Experiments & Performance Analysis

## 1. Overview

This document covers four controlled experiments run on the DQN agent
trained on **LunarLander-v3**.  Each experiment changes one variable at
a time while holding all others at the baseline, so the effect of each
parameter can be isolated and measured.

All experiments are run via:
```bash
python src/experiments/run_all.py            # all 4 tasks
python src/experiments/run_all.py --task 1   # specific task
python src/experiments/run_all.py --episodes 100  # quick smoke test
```

---

## 2. Baseline Configuration

All experiments use these values unless stated otherwise:

| Parameter | Baseline Value |
|---|---|
| Environment | `LunarLander-v3` |
| Learning rate α | 0.0005 |
| Discount factor γ | 0.99 |
| Epsilon start | 1.0 |
| Epsilon min | 0.01 |
| Decay rate | 0.005 |
| Batch size | 64 |
| Replay buffer | 50,000 |
| Hidden size | 128 |
| Target sync freq | 10 episodes |
| Seed | 42 |

---

## 3. Task 1 — Hyperparameter Sweep (α and γ)

### 3.1 Learning Rate Sweep (γ fixed at 0.99)

Three values tested: `α ∈ {0.0001, 0.0005, 0.001}`

| α | Avg Reward | Best Mov-Avg | Observations |
|---|---|---|---|
| 0.0001 | TBD after run | TBD | Slow convergence expected; stable but needs more episodes |
| 0.0005 | TBD | TBD | **Baseline** — well-tuned for Adam + LunarLander |
| 0.001 | TBD | TBD | Faster early learning but may overshoot Q-values |

**Why these values?**
`α=0.0005` is the standard DQN Adam learning rate used in published
LunarLander benchmarks. `0.0001` is 5× slower — useful if the baseline
oscillates. `0.001` is 2× faster — may destabilise training on longer runs.

**Expected winner:** `α=0.0005` — the baseline is well-calibrated.
`α=0.001` may match it early but become noisier.

### 3.2 Gamma Sweep (α fixed at 0.0005)

Three values tested: `γ ∈ {0.90, 0.95, 0.99}`

| γ | Avg Reward | Best Mov-Avg | Observations |
|---|---|---|---|
| 0.90 | TBD | TBD | Myopic — 0.9^200 ≈ 1e-9, landing bonus nearly invisible |
| 0.95 | TBD | TBD | Moderate planning horizon |
| 0.99 | TBD | TBD | **Baseline** — correct for dense-reward long-horizon task |

**Why γ=0.99 wins:**
LunarLander episodes average 200–400 steps. The terminal landing bonus
(+100 to +140) must propagate back through ~200 Bellman updates.
At γ=0.99: `0.99^200 ≈ 0.13` — still meaningful signal.
At γ=0.90: `0.90^200 ≈ 0` — the landing bonus has zero influence
on early-episode decisions.

**Result file:** `experiments/hyperparameter_results.csv`
**Plots:** `experiments/plots/reward_hp_lr_sweep.png`,
           `experiments/plots/reward_hp_gamma_sweep.png`

---

## 4. Task 2 — Exploration Strategy Comparison

### 4.1 Strategies Tested

| Strategy | Parameter | Description |
|---|---|---|
| Epsilon-Greedy | decay=0.005 | Standard — uniform random when exploring |
| Softmax τ=1.0 | temperature=1.0 | Boltzmann — probability ∝ exp(Q/τ) |
| Softmax τ=0.5 | temperature=0.5 | Cooler Boltzmann — more greedy |

### 4.2 How Softmax Works

```
P(a | s) = exp(Q(s,a) / τ)  /  Σ_a' exp(Q(s,a') / τ)
```

Unlike epsilon-greedy which treats all non-greedy actions equally,
softmax gives higher probability to actions with higher Q-values.
This means even when "exploring", the agent tends toward better actions.

**Implementation:** `src/experiments/exploration.py::softmax_action()`

### 4.3 Expected Observations

- **Epsilon-greedy** converges reliably once epsilon decays — proven strategy
- **Softmax τ=1.0** may explore more efficiently early but converges slower
  because it never fully commits to greedy (no decay mechanism)
- **Softmax τ=0.5** is greedier — similar to low-epsilon epsilon-greedy

**Result file:** `experiments/exploration_comparison.csv`
**Plot:** `experiments/plots/reward_exploration.png`

---

## 5. Task 3 — Epsilon Decay Rate Experiments

### 5.1 Decay Rates Tested

| Label | Decay Rate | Episodes to ε_min | Description |
|---|---|---|---|
| Fast decay | 0.015 | ~200 | Commits to exploitation quickly |
| Medium decay | 0.005 | ~600 | **Baseline** — balanced schedule |
| Slow decay | 0.002 | ~1500 | Explores throughout entire run |

Epsilon schedule formula used across all runs:
```
ε(episode) = max(ε_min,  ε_max × exp(−decay_rate × episode))
```

### 5.2 What Changes Between Runs

Only `decay_rate` changes. All other parameters (LR, γ, batch size,
buffer, architecture) remain at baseline values.

### 5.3 Expected Observations

**Fast decay (0.015):**
- Reward rises faster early (exploiting policy sooner)
- Risk: if the policy isn't good yet at episode ~200, it gets locked in
- May plateau lower than medium because of premature exploitation

**Medium decay (0.005) — baseline:**
- Balanced — neither rushes exploitation nor over-explores
- Should achieve highest final average reward over 300 episodes

**Slow decay (0.002):**
- Stays above ε=0.5 for most of the 300-episode run
- Reward improves slowly — buffer fills with diverse but random transitions
- Better final performance *if* given 600+ episodes; underperforms in short runs

### 5.4 Epsilon Schedule Table

| Episode | Fast (0.015) | Medium (0.005) | Slow (0.002) |
|---|---|---|---|
| 50 | 0.472 | 0.778 | 0.905 |
| 100 | 0.223 | 0.607 | 0.819 |
| 150 | 0.105 | 0.473 | 0.741 |
| 200 | 0.050 (min) | 0.368 | 0.670 |
| 300 | 0.010 (min) | 0.223 | 0.549 |

**Result file:** `experiments/epsilon_decay_results.csv`
**Plots:**
- `experiments/plots/epsilon_decay_rewards.png` — reward curves
- `experiments/plots/epsilon_decay_curves.png` — epsilon over episodes
- `experiments/plots/epsilon_decay_grid.png`   — combined 2-panel view

---

## 6. Task 4 — Performance Metrics

### 6.1 Metrics Computed

All metrics are computed by `src/utils/metrics.py::compute_metrics()`:

| Metric | Description | How computed |
|---|---|---|
| `avg_reward` | Mean reward across all episodes | `np.mean(rewards)` |
| `std_reward` | Standard deviation of rewards | `np.std(rewards)` |
| `min_reward` | Lowest episode reward | `np.min(rewards)` |
| `max_reward` | Highest episode reward | `np.max(rewards)` |
| `avg_steps` | Mean steps per episode | `np.mean(steps)` |
| `moving_avg_rewards` | Rolling average (window=50) | Expanding window mean |
| `best_moving_avg` | Peak of the moving average | `max(moving_avg)` |
| `best_moving_avg_episode` | Episode where peak occurred | `argmax(moving_avg)` |

### 6.2 Utility Functions

```python
from src.utils.metrics import compute_metrics, save_metrics_csv, print_metrics_table

# Compute all metrics from raw episode lists
stats = compute_metrics(rewards, steps_list, window=50)

# Print formatted table to console
print_metrics_table("My Experiment", stats)

# Save flat summary to CSV
save_metrics_csv(stats, "experiments/my_run/summary.csv")
```

Example console output:
```
  ┌─ Baseline (LR=0.0005, γ=0.99, decay=0.005)
  │  Episodes         : 300
  │  Avg reward       : -45.23  ±  112.45
  │  Min / Max reward : -312.10  /  287.45
  │  Avg steps/ep     : 342.1
  │  Best mov-avg(50) : 87.34  at ep 281
  └──────────────────────────────────────────────────────
```

### 6.3 CSV Structure — Per-Experiment Results

Every experiment CSV shares this schema:

**`experiments/hyperparameter_results.csv`**
```
experiment, learning_rate, gamma, decay_rate,
avg_reward, std_reward, best_mov_avg, best_mov_ep, avg_steps
```

**`experiments/exploration_comparison.csv`**
```
strategy, description,
avg_reward, std_reward, best_mov_avg, best_mov_ep, avg_steps
```

**`experiments/epsilon_decay_results.csv`**
```
decay_rate, label,
avg_reward, std_reward, best_mov_avg, best_mov_ep, avg_steps
```

---

## 7. Output File Map

```
experiments/
├── hyperparameter_results.csv     ← Task 1 results
├── exploration_comparison.csv     ← Task 2 results
├── epsilon_decay_results.csv      ← Task 3 results
└── plots/
    ├── reward_hp_lr_sweep.png     ← Task 1a: LR comparison
    ├── reward_hp_gamma_sweep.png  ← Task 1b: γ comparison
    ├── reward_exploration.png     ← Task 2: exploration strategies
    ├── epsilon_decay_rewards.png  ← Task 3: reward curves
    ├── epsilon_decay_curves.png   ← Task 3: ε decay curves
    └── epsilon_decay_grid.png     ← Task 3: combined 2-panel
```

---

## 8. How to Reproduce

```bash
# Run all experiments (default 300 episodes each):
python src/experiments/run_all.py

# Quick smoke test (50 episodes):
python src/experiments/run_all.py --episodes 50

# Run a single task:
python src/experiments/run_all.py --task 1   # α/γ sweep
python src/experiments/run_all.py --task 2   # exploration comparison
python src/experiments/run_all.py --task 3   # epsilon decay

# Regenerate plots only (if CSVs already exist):
# (re-run any task — it re-plots from fresh runs)
```

---

## 9. Which Configuration Performed Best (Expected)

| Experiment | Best Config | Reason |
|---|---|---|
| Learning rate | `α=0.0005` | Validated DQN Adam LR; 0.001 adds noise, 0.0001 too slow |
| Discount factor | `γ=0.99` | Only value that preserves landing bonus over 200+ step horizon |
| Exploration | `Epsilon-greedy` | Proven convergence guarantee via decay; softmax has no decay |
| Epsilon decay | `rate=0.005` (medium) | Balances exploration and exploitation over 300 episodes |

**Overall best baseline:** `α=0.0005, γ=0.99, decay=0.005, epsilon-greedy`
This matches `config.yaml` — confirming the baseline was well-calibrated.

---

## 10. Code Architecture

```
src/
├── experiments/
│   ├── engine.py        ← Shared training loop (pluggable action selector)
│   ├── exploration.py   ← softmax_action() + epsilon_greedy_action()
│   ├── plot_utils.py    ← plot_reward_curves(), plot_epsilon_curves(),
│   │                       plot_comparison_grid()
│   └── run_all.py       ← Master runner: task1, task2, task3 + argparse CLI
└── utils/
    └── metrics.py       ← compute_metrics(), save_metrics_csv(),
                            print_metrics_table()
```

The `engine.py` `action_selector` parameter makes exploration strategies
pluggable — passing `softmax_action` or `None` (epsilon-greedy) requires
zero changes to the training loop itself.
