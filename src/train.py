"""
src/train.py
-------------
Baseline DQN Training Script — Tasks 5 & 6
============================================

Task 5 — Training Loop:
    Implements a complete episode-based DQN training loop using the
    existing DQN model (src/model.py), ReplayBuffer (src/replay_buffer.py),
    and DQNAgent (src/agent/dqn_agent.py) already present in the repo.

Task 6 — Baseline Run:
    Uses assignment-style baseline parameters, adjusted where the original
    values are unsuitable for Deep Q-Learning with PyTorch + LunarLander-v2.
    See PARAMETER NOTES below for the full rationale.

PARAMETER NOTES — Assignment values vs. what we use here:
┌─────────────────────┬──────────────┬──────────────┬─────────────────────────────────────────────────┐
│ Parameter           │ Assignment   │ Used Here    │ Reason for change (if any)                      │
├─────────────────────┼──────────────┼──────────────┼─────────────────────────────────────────────────┤
│ total_episodes      │ 5000         │ 500          │ 5000 takes ~3 hrs on CPU; 500 shows full curve  │
│ total_test_episodes │ 100          │ 100          │ Kept as-is — used in final evaluation           │
│ max_steps           │ 99           │ 500          │ 99 is too short; lander needs ~200-400 steps    │
│ learning_rate       │ 0.7          │ 0.0005       │ LR=0.7 diverges neural nets; use Adam + 5e-4   │
│ gamma               │ 0.8          │ 0.99         │ γ=0.8 undervalues future reward in dense envs  │
│ epsilon (start)     │ 1.0          │ 1.0          │ Kept — full exploration at start is correct     │
│ max_epsilon         │ 1.0          │ 1.0          │ Kept                                            │
│ min_epsilon         │ 0.01         │ 0.01         │ Kept — 1% random floor                         │
│ decay_rate          │ 0.01         │ 0.005        │ exp(-0.01*ep) decays too fast; 0.005 is gentler│
└─────────────────────┴──────────────┴──────────────┴─────────────────────────────────────────────────┘

Epsilon schedule used: exponential decay per episode
    epsilon = max(min_epsilon, max_epsilon * exp(-decay_rate * episode))
    This matches the assignment's formula style.

Outputs saved to experiments/baseline_run/:
    training_metrics.csv  — per-episode: reward, steps, epsilon, moving avg
    reward_curve.png      — reward + 50-episode moving average plot
    steps_curve.png       — steps per episode plot
    training_summary.json — final summary statistics

Usage:
    # Run with all defaults (from repo root):
    python src/train.py

    # Override individual parameters via CLI:
    python src/train.py --episodes 300 --lr 0.001 --gamma 0.95

    # Run with a different output folder:
    python src/train.py --out-dir experiments/my_run
"""

import sys
import os
import csv
import json
import math
import random
import argparse
import numpy as np
import gymnasium as gym
from collections import deque

# ── Make src/ importable when running from repo root ──────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model         import DQN
from src.replay_buffer import ReplayBuffer

import torch
import torch.nn as nn
import torch.optim as optim

# ═══════════════════════════════════════════════════════════════
# BASELINE HYPERPARAMETERS (Task 6)
# Based on assignment spec — see table in docstring for rationale
# ═══════════════════════════════════════════════════════════════
DEFAULTS = {
    # Environment
    "env_name"        : "LunarLander-v2",
    "seed"            : 42,

    # Training schedule
    "total_episodes"  : 500,       # assignment: 5000 → reduced for CPU runtime
    "test_episodes"   : 100,       # kept from assignment
    "max_steps"       : 500,       # assignment: 99 → too short for LunarLander

    # Learning
    "learning_rate"   : 0.0005,    # assignment: 0.7 → too high for neural nets
    "gamma"           : 0.99,      # assignment: 0.8 → too low for dense rewards

    # Exploration (epsilon-greedy with exponential decay)
    "epsilon_start"   : 1.0,       # kept from assignment
    "epsilon_max"     : 1.0,       # kept from assignment
    "epsilon_min"     : 0.01,      # kept from assignment
    "decay_rate"      : 0.005,     # assignment: 0.01 → adjusted for longer run

    # Neural network + replay buffer
    "hidden_size"     : 128,
    "batch_size"      : 64,
    "buffer_capacity" : 50000,
    "target_sync_freq": 10,        # sync target net every N episodes

    # Output
    "out_dir"         : "experiments/baseline_run",
    "log_freq"        : 10,        # print to console every N episodes
}


# ═══════════════════════════════════════════════════════════════
# EPSILON DECAY
# ═══════════════════════════════════════════════════════════════
def compute_epsilon(episode: int, epsilon_max: float,
                    epsilon_min: float, decay_rate: float) -> float:
    """
    Exponential epsilon decay — matches the assignment formula style.

        epsilon = max(epsilon_min, epsilon_max * exp(-decay_rate * episode))

    Why exponential instead of multiplicative?
        The assignment uses decay_rate=0.01 with this formula. It gives
        a fast early drop in exploration followed by a long tail near
        epsilon_min, which is appropriate for learning schedules.

    Args:
        episode    : Current episode number (0-indexed for clean math).
        epsilon_max: Starting epsilon value (e.g. 1.0).
        epsilon_min: Floor value (e.g. 0.01).
        decay_rate : Controls how fast epsilon falls. Higher = faster.

    Returns:
        float: Current epsilon clamped to [epsilon_min, epsilon_max].
    """
    epsilon = epsilon_max * math.exp(-decay_rate * episode)
    return max(epsilon_min, epsilon)


# ═══════════════════════════════════════════════════════════════
# ACTION SELECTION
# ═══════════════════════════════════════════════════════════════
def select_action(state: np.ndarray, online_net: DQN,
                  epsilon: float, action_dim: int,
                  device: torch.device) -> int:
    """
    Epsilon-greedy action selection.

    Two-branch logic:
      - EXPLORE (probability = epsilon):
          Pick a uniformly random action.
          Critical early in training when Q-values are meaningless.

      - EXPLOIT (probability = 1 - epsilon):
          Ask the Q-network for Q(s, a) for all actions, pick argmax.
          As epsilon decays, the agent relies more on its learned policy.

    Args:
        state      : Current observation. Shape: (state_dim,)
        online_net : The Q-network being trained.
        epsilon    : Current exploration rate in [0, 1].
        action_dim : Number of discrete actions.
        device     : CPU or CUDA device.

    Returns:
        int: Chosen action index.
    """
    if random.random() < epsilon:
        # ── EXPLORE: random action ───────────────────────────
        return random.randint(0, action_dim - 1)

    # ── EXPLOIT: greedy Q-value action ───────────────────────
    # Convert state to tensor: (state_dim,) → (1, state_dim)
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

    online_net.eval()
    with torch.no_grad():
        q_values = online_net(state_tensor)   # shape: (1, action_dim)
    online_net.train()

    return int(q_values.argmax(dim=1).item())


# ═══════════════════════════════════════════════════════════════
# LEARNING STEP (Bellman Update)
# ═══════════════════════════════════════════════════════════════
def learning_step(
    online_net:  DQN,
    target_net:  DQN,
    replay_buf:  ReplayBuffer,
    optimizer:   optim.Optimizer,
    loss_fn:     nn.Module,
    batch_size:  int,
    gamma:       float,
    device:      torch.device,
) -> float:
    """
    Sample a mini-batch from the replay buffer and perform one
    gradient descent step on the online Q-network.

    Bellman Target:
        target_Q(s, a) = r  +  gamma * max_a' Q_target(s', a') * (1 - done)

        Intuition:
          - If done=1 (terminal): target = r only (no future reward possible)
          - If done=0 (non-terminal): target = r + discounted max future Q

    Loss:
        MSE(Q_online(s, a_taken),  target_Q(s, a))

        We only update Q for the action that was actually taken,
        not all 4 actions simultaneously.

    Args:
        online_net : The network being trained (gradients flow through this).
        target_net : Frozen copy used for stable Bellman bootstrap values.
        replay_buf : Experience replay buffer to sample from.
        optimizer  : Adam optimiser attached to online_net.
        loss_fn    : MSELoss instance.
        batch_size : Number of transitions to sample.
        gamma      : Discount factor.
        device     : CPU or CUDA.

    Returns:
        float: Scalar loss value for this update step.
    """
    # ── Sample random mini-batch from replay buffer ───────────
    states, actions, rewards, next_states, dones = replay_buf.sample(batch_size)

    # ── Convert NumPy arrays → PyTorch tensors ────────────────
    states_t      = torch.FloatTensor(states).to(device)       # (B, state_dim)
    actions_t     = torch.LongTensor(actions).to(device)       # (B,)
    rewards_t     = torch.FloatTensor(rewards).to(device)      # (B,)
    next_states_t = torch.FloatTensor(next_states).to(device)  # (B, state_dim)
    dones_t       = torch.FloatTensor(dones).to(device)        # (B,) — 0.0 or 1.0

    # ── Current Q-values: Q_online(s, a_taken) ───────────────
    # online_net gives Q for ALL actions → gather selects only the taken action
    # gather(dim=1, index) picks column `action` from each row
    all_q      = online_net(states_t)                              # (B, action_dim)
    q_current  = all_q.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # (B,)

    # ── Bellman target: r + gamma * max_a' Q_target(s', a') ──
    # Use target_net (not online_net) for stable bootstrap values.
    # no_grad: targets are fixed labels — don't backprop through them.
    with torch.no_grad():
        q_next_all = target_net(next_states_t)          # (B, action_dim)
        q_next_max = q_next_all.max(dim=1)[0]           # (B,) — best next Q
        # Mask terminal states: done=1 → no future reward
        q_target   = rewards_t + gamma * q_next_max * (1.0 - dones_t)  # (B,)

    # ── Compute MSE loss and backpropagate ────────────────────
    loss = loss_fn(q_current, q_target)

    optimizer.zero_grad()      # clear gradients from previous step
    loss.backward()            # compute gradients via autograd
    # Clip gradients to prevent exploding updates in early training
    nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=1.0)
    optimizer.step()           # apply gradient update

    return loss.item()


# ═══════════════════════════════════════════════════════════════
# METRICS — CSV writer + in-memory tracker
# ═══════════════════════════════════════════════════════════════
class MetricsTracker:
    """
    Tracks per-episode metrics and writes them to a CSV file.

    Columns written:
        episode, reward, steps, epsilon, moving_avg_50, moving_avg_100
    """

    def __init__(self, csv_path: str):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self._file    = open(csv_path, "w", newline="")
        self._writer  = csv.writer(self._file)
        self._writer.writerow(
            ["episode", "reward", "steps", "epsilon",
             "moving_avg_50", "moving_avg_100"]
        )
        self._file.flush()
        self._rewards = []   # full history for rolling averages

    def record(self, episode: int, reward: float,
               steps: int, epsilon: float) -> dict:
        """Write one CSV row. Returns the row dict for console printing."""
        self._rewards.append(reward)
        avg50  = float(np.mean(self._rewards[-50:]))
        avg100 = float(np.mean(self._rewards[-100:]))
        self._writer.writerow([
            episode, round(reward, 3), steps,
            round(epsilon, 5), round(avg50, 3), round(avg100, 3)
        ])
        self._file.flush()
        return {"avg50": avg50, "avg100": avg100}

    def close(self) -> list:
        """Close the file and return the full rewards list."""
        self._file.close()
        return self._rewards

    def best_avg100(self) -> float:
        if len(self._rewards) < 100:
            return float(np.mean(self._rewards))
        return max(
            float(np.mean(self._rewards[i:i+100]))
            for i in range(len(self._rewards) - 99)
        )


# ═══════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════
def save_plots(rewards: list, out_dir: str) -> None:
    """
    Save reward curve and steps curve PNG files to out_dir.
    Uses matplotlib with a non-interactive backend (safe for headless).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plot] matplotlib not available — skipping plots.")
        return

    episodes = list(range(1, len(rewards) + 1))

    # ── Compute 50-episode moving average ────────────────────
    moving_avg = []
    for i in range(len(rewards)):
        window = rewards[max(0, i - 49): i + 1]
        moving_avg.append(float(np.mean(window)))

    # ── Reward curve ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards, color="steelblue", alpha=0.4,
            linewidth=0.8, label="Episode reward")
    ax.plot(episodes, moving_avg, color="darkorange", linewidth=2.0,
            label="Moving avg (50 ep)")
    ax.axhline(y=200, color="green", linestyle="--", linewidth=1.2,
               label="Solve threshold (200)", alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("DQN Baseline Training — LunarLander-v2\nReward per Episode")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    reward_path = os.path.join(out_dir, "reward_curve.png")
    plt.savefig(reward_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Reward curve  → {reward_path}")

    # ── Also save a copy to docs/figures/ ────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(episodes, rewards, color="steelblue", alpha=0.4,
             linewidth=0.8, label="Episode reward")
    ax2.plot(episodes, moving_avg, color="darkorange", linewidth=2.0,
             label="Moving avg (50 ep)")
    ax2.axhline(y=200, color="green", linestyle="--", linewidth=1.2,
                label="Solve threshold (200)", alpha=0.8)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Total Reward")
    ax2.set_title("DQN Baseline Training — LunarLander-v2")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    docs_path = os.path.join(ROOT, "docs", "figures", "baseline_reward_curve.png")
    os.makedirs(os.path.dirname(docs_path), exist_ok=True)
    plt.savefig(docs_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[Plot] Docs copy     → {docs_path}")


# ═══════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION (Task 5 — Training Loop)
# ═══════════════════════════════════════════════════════════════
def train(args: argparse.Namespace) -> None:
    """
    Full DQN training loop — Task 5 implementation.

    Episode loop structure:
        for episode in range(total_episodes):
            reset env → get initial state
            compute epsilon (exponential decay)
            for step in range(max_steps):
                select action (epsilon-greedy)   ← Task 5 requirement
                step env → (next_state, reward, done)
                store (s, a, r, s', done) in replay buffer
                if buffer ready:
                    sample mini-batch
                    compute Bellman target       ← Task 5 requirement
                    MSE loss + optimizer step    ← Task 5 requirement
                if done: break
            sync target net every target_sync_freq episodes
            record metrics to CSV
    """
    # ── Reproducibility ──────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Device ───────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  DQN Baseline Training Run")
    print(f"  Environment : {args.env_name}")
    print(f"  Device      : {device}")
    print(f"  Episodes    : {args.total_episodes}  |  Max steps: {args.max_steps}")
    print(f"  LR={args.lr}  γ={args.gamma}  ε_decay={args.decay_rate}")
    print(f"  Output dir  : {args.out_dir}")
    print(f"{'='*60}\n")

    # ── Environment ──────────────────────────────────────────
    env        = gym.make(args.env_name, render_mode=None)
    state_dim  = env.observation_space.shape[0]   # 8 for LunarLander-v2
    action_dim = env.action_space.n               # 4 for LunarLander-v2
    print(f"[Env] {args.env_name}  obs_dim={state_dim}  action_dim={action_dim}\n")

    # ── Networks: online (trained) + target (stable bootstrap) ─
    online_net = DQN(state_dim, action_dim, args.hidden_size).to(device)
    target_net = DQN(state_dim, action_dim, args.hidden_size).to(device)
    # Initialise target net with identical weights as online net
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()   # target net is never directly trained

    # ── Optimizer + Loss ─────────────────────────────────────
    # Adam optimizer: adaptive LR per parameter, works well for DQN
    optimizer = optim.Adam(online_net.parameters(), lr=args.lr)
    loss_fn   = nn.MSELoss()   # MSE on Q-value residuals (Bellman error)

    # ── Replay buffer ─────────────────────────────────────────
    replay_buf = ReplayBuffer(capacity=args.buffer_capacity)

    # ── Metrics tracker ───────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "training_metrics.csv")
    tracker  = MetricsTracker(csv_path)

    best_avg100     = -float("inf")
    best_ep         = 0
    recent_rewards  = deque(maxlen=100)

    # ═══════════════════════════════════════════════════════════
    # EPISODE LOOP — Task 5 core
    # ═══════════════════════════════════════════════════════════
    for episode in range(args.total_episodes):

        # Reset environment and get initial observation
        state, _ = env.reset(seed=args.seed + episode)
        state    = np.array(state, dtype=np.float32)

        # Compute epsilon for this episode using exponential decay
        # (matches assignment decay formula: ε = ε_max * exp(-rate * ep))
        epsilon = compute_epsilon(
            episode, args.epsilon_max, args.epsilon_min, args.decay_rate
        )

        episode_reward = 0.0
        episode_losses = []
        steps          = 0

        # ── STEP LOOP ────────────────────────────────────────
        for step in range(args.max_steps):

            # 1. Epsilon-greedy action selection
            action = select_action(
                state, online_net, epsilon, action_dim, device
            )

            # 2. Step the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)

            # done = True when episode ends (crash, landing, or step limit)
            done = terminated or truncated

            # 3. Store transition in replay buffer
            replay_buf.push(state, action, reward, next_state, done)

            # 4. Learn — only once buffer has enough samples for a batch
            if replay_buf.is_ready(args.batch_size):
                loss_val = learning_step(
                    online_net, target_net, replay_buf,
                    optimizer, loss_fn,
                    args.batch_size, args.gamma, device
                )
                episode_losses.append(loss_val)

            episode_reward += reward
            state           = next_state
            steps          += 1

            if done:
                break   # episode finished naturally

        # ── POST-EPISODE BOOKKEEPING ──────────────────────────

        # Sync target network every N episodes
        # (copies online_net weights → target_net for stable future targets)
        if (episode + 1) % args.target_sync_freq == 0:
            target_net.load_state_dict(online_net.state_dict())

        # Record metrics to CSV and get rolling averages
        recent_rewards.append(episode_reward)
        stats = tracker.record(episode + 1, episode_reward, steps, epsilon)

        # Track best 100-episode avg
        if stats["avg100"] > best_avg100 and len(recent_rewards) == 100:
            best_avg100 = stats["avg100"]
            best_ep     = episode + 1

        # Console progress every log_freq episodes
        if (episode + 1) % args.log_freq == 0:
            mean_loss = np.mean(episode_losses) if episode_losses else 0.0
            print(
                f"  Ep {episode+1:>4}/{args.total_episodes} | "
                f"Reward: {episode_reward:>8.2f} | "
                f"Avg50: {stats['avg50']:>7.2f} | "
                f"Avg100: {stats['avg100']:>7.2f} | "
                f"ε: {epsilon:.4f} | "
                f"Steps: {steps:>4} | "
                f"Loss: {mean_loss:.5f}"
            )

        # Early stop if environment is solved
        if stats["avg100"] >= 200.0 and len(recent_rewards) == 100:
            print(f"\n🎉  Solved at episode {episode+1}! "
                  f"Avg100 = {stats['avg100']:.2f}\n")
            break

    # ── WRAP UP ──────────────────────────────────────────────
    env.close()
    rewards_history = tracker.close()

    # Save plots
    save_plots(rewards_history, args.out_dir)

    # Save JSON summary
    summary = {
        "environment"          : args.env_name,
        "total_episodes_run"   : len(rewards_history),
        "best_avg_reward_100ep": round(best_avg100, 3),
        "best_avg_at_episode"  : best_ep,
        "final_epsilon"        : round(compute_epsilon(
                                    len(rewards_history),
                                    args.epsilon_max,
                                    args.epsilon_min,
                                    args.decay_rate), 5),
        "hyperparameters": {
            "learning_rate" : args.lr,
            "gamma"         : args.gamma,
            "epsilon_min"   : args.epsilon_min,
            "decay_rate"    : args.decay_rate,
            "batch_size"    : args.batch_size,
            "buffer_capacity": args.buffer_capacity,
            "hidden_size"   : args.hidden_size,
            "target_sync_freq": args.target_sync_freq,
        }
    }
    summary_path = os.path.join(args.out_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  ✅  Training complete")
    print(f"  Episodes run         : {len(rewards_history)}")
    print(f"  Best avg100 reward   : {best_avg100:.2f}  (at ep {best_ep})")
    print(f"  Metrics CSV          : {csv_path}")
    print(f"  Summary JSON         : {summary_path}")
    print(f"  Reward curve PNG     : {args.out_dir}/reward_curve.png")
    print(f"  Docs figure          : docs/figures/baseline_reward_curve.png")
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════
# CLI ARGUMENT PARSER
# ═══════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments with baseline defaults from DEFAULTS dict.
    All parameters can be overridden without editing source code.
    """
    p = argparse.ArgumentParser(
        description="DQN Baseline Training — LunarLander-v2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    D = DEFAULTS  # shorthand

    # Environment
    p.add_argument("--env",        dest="env_name",       default=D["env_name"],
                   help="Gymnasium environment ID")
    p.add_argument("--seed",                              default=D["seed"],        type=int)

    # Training schedule
    p.add_argument("--episodes",   dest="total_episodes", default=D["total_episodes"], type=int,
                   help="Total training episodes")
    p.add_argument("--max-steps",  dest="max_steps",      default=D["max_steps"],   type=int,
                   help="Max steps per episode")

    # Learning
    p.add_argument("--lr",                                default=D["learning_rate"], type=float,
                   help="Adam learning rate")
    p.add_argument("--gamma",                             default=D["gamma"],       type=float,
                   help="Discount factor")

    # Exploration
    p.add_argument("--epsilon-min", dest="epsilon_min",   default=D["epsilon_min"], type=float)
    p.add_argument("--epsilon-max", dest="epsilon_max",   default=D["epsilon_max"], type=float)
    p.add_argument("--decay-rate",  dest="decay_rate",    default=D["decay_rate"],  type=float,
                   help="Exponential epsilon decay rate")

    # Network + buffer
    p.add_argument("--hidden-size", dest="hidden_size",   default=D["hidden_size"], type=int)
    p.add_argument("--batch-size",  dest="batch_size",    default=D["batch_size"],  type=int)
    p.add_argument("--buffer-cap",  dest="buffer_capacity", default=D["buffer_capacity"], type=int)
    p.add_argument("--sync-freq",   dest="target_sync_freq", default=D["target_sync_freq"], type=int,
                   help="Target network sync frequency (episodes)")

    # Output
    p.add_argument("--out-dir",    dest="out_dir",        default=D["out_dir"],
                   help="Directory to save CSV, plots, and JSON")
    p.add_argument("--log-freq",   dest="log_freq",       default=D["log_freq"],    type=int,
                   help="Console print frequency (episodes)")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    args = parse_args()
    train(args)
