"""
src/experiments/engine.py
--------------------------
Shared training engine used by all experiment scripts.

Provides run_experiment() — a single function that:
  1. Creates the environment
  2. Builds online + target DQN networks
  3. Runs the full episode loop
  4. Returns per-episode rewards, steps, and epsilon values

All experiment scripts (hyperparameter sweep, exploration comparison,
epsilon decay study) call this function with different configs, keeping
the experiment logic DRY and the training loop in one place.

Key design decisions:
  - action_selector parameter makes exploration strategy pluggable
  - No side-effects: caller decides where to save results
  - Reuses src/model.py and src/replay_buffer.py from main branch
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from typing import Callable, List, Tuple

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.model         import DQN
from src.replay_buffer import ReplayBuffer


# ── Type alias for action selector functions ──────────────────
# Signature: (state, net, extra_param, action_dim, device) → int
ActionSelector = Callable[
    [np.ndarray, DQN, float, int, torch.device], int
]


def run_experiment(
    env_name:        str   = "LunarLander-v3",
    total_episodes:  int   = 300,
    max_steps:       int   = 500,
    learning_rate:   float = 0.0005,
    gamma:           float = 0.99,
    epsilon_min:     float = 0.01,
    epsilon_max:     float = 1.0,
    decay_rate:      float = 0.005,
    hidden_size:     int   = 128,
    batch_size:      int   = 64,
    buffer_capacity: int   = 50000,
    target_sync_freq:int   = 10,
    seed:            int   = 42,
    action_selector: ActionSelector = None,  # None → epsilon-greedy
    exploration_param: float = None,         # epsilon or temperature
    log_freq:        int   = 50,
    label:           str   = "experiment",
) -> Tuple[List[float], List[int], List[float]]:
    """
    Run a complete DQN training loop and return raw episode metrics.

    Args:
        env_name        : Gymnasium environment ID.
        total_episodes  : Number of training episodes.
        max_steps       : Max steps per episode.
        learning_rate   : Adam optimizer LR.
        gamma           : Discount factor.
        epsilon_min     : Minimum epsilon (used by epsilon-greedy).
        epsilon_max     : Starting epsilon.
        decay_rate      : Exponential decay rate for epsilon.
        hidden_size     : Hidden layer width for DQN.
        batch_size      : Replay buffer sample size.
        buffer_capacity : Replay buffer max size.
        target_sync_freq: Episodes between target net syncs.
        seed            : Random seed.
        action_selector : Pluggable action selection function.
                          If None, uses built-in epsilon-greedy.
        exploration_param: Passed to custom action_selector (e.g. temperature).
        log_freq        : Console print frequency.
        label           : Human-readable experiment label for printing.

    Returns:
        Tuple of three lists, one value per episode:
          rewards  : total reward per episode
          steps    : steps taken per episode
          epsilons : epsilon value used that episode
    """
    # ── Seed ─────────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Environment ──────────────────────────────────────────
    env        = gym.make(env_name, render_mode=None)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # ── Networks ─────────────────────────────────────────────
    online_net = DQN(state_dim, action_dim, hidden_size).to(device)
    target_net = DQN(state_dim, action_dim, hidden_size).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer  = optim.Adam(online_net.parameters(), lr=learning_rate)
    loss_fn    = nn.MSELoss()
    replay_buf = ReplayBuffer(capacity=buffer_capacity)

    # ── History collectors ───────────────────────────────────
    rewards_hist:  List[float] = []
    steps_hist:    List[int]   = []
    epsilons_hist: List[float] = []

    print(f"\n  [{label}]  LR={learning_rate}  γ={gamma}  "
          f"decay={decay_rate}  episodes={total_episodes}")

    # ── Episode loop ─────────────────────────────────────────
    for episode in range(total_episodes):

        state, _ = env.reset(seed=seed + episode)
        state    = np.array(state, dtype=np.float32)

        # Compute epsilon for this episode
        epsilon = max(epsilon_min,
                      epsilon_max * math.exp(-decay_rate * episode))

        episode_reward = 0.0
        steps          = 0

        for _ in range(max_steps):

            # ── Action selection ──────────────────────────────
            if action_selector is not None:
                # Use pluggable selector (e.g. softmax)
                param = exploration_param if exploration_param is not None \
                        else epsilon
                action = action_selector(state, online_net, param,
                                         action_dim, device)
            else:
                # Default: epsilon-greedy
                if random.random() < epsilon:
                    action = random.randint(0, action_dim - 1)
                else:
                    st = torch.FloatTensor(state).unsqueeze(0).to(device)
                    online_net.eval()
                    with torch.no_grad():
                        action = int(online_net(st).argmax(dim=1).item())
                    online_net.train()

            # ── Step ──────────────────────────────────────────
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            done       = terminated or truncated

            replay_buf.push(state, action, reward, next_state, done)

            # ── Learn ─────────────────────────────────────────
            if replay_buf.is_ready(batch_size):
                states_t, actions_t, rewards_t, next_states_t, dones_t = \
                    [torch.FloatTensor(x).to(device) if i != 1
                     else torch.LongTensor(x).to(device)
                     for i, x in enumerate(replay_buf.sample(batch_size))]

                q_cur = online_net(states_t)\
                        .gather(1, actions_t.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_tgt = rewards_t + gamma * \
                            target_net(next_states_t).max(1)[0] * \
                            (1.0 - dones_t)
                loss = loss_fn(q_cur, q_tgt)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online_net.parameters(), 1.0)
                optimizer.step()

            episode_reward += reward
            state           = next_state
            steps          += 1
            if done:
                break

        # ── Post-episode ──────────────────────────────────────
        if (episode + 1) % target_sync_freq == 0:
            target_net.load_state_dict(online_net.state_dict())

        rewards_hist.append(episode_reward)
        steps_hist.append(steps)
        epsilons_hist.append(epsilon)

        if (episode + 1) % log_freq == 0:
            avg = float(np.mean(rewards_hist[-50:]))
            print(f"    ep {episode+1:>4} | reward {episode_reward:>8.2f}"
                  f" | avg50 {avg:>7.2f} | ε {epsilon:.4f}")

    env.close()
    return rewards_hist, steps_hist, epsilons_hist
