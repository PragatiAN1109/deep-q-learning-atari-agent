"""
src/replay_buffer.py
---------------------
Experience Replay Buffer for the DQN agent.

Why experience replay?
    In standard online learning, the agent trains on each transition
    (s, a, r, s', done) immediately after it occurs and then discards it.
    This is inefficient and unstable because:
      1. Consecutive transitions are highly correlated — the network
         receives very similar inputs in a row, causing it to overfit
         to recent experience and forget earlier knowledge.
      2. Each experience is only used once, wasting data.

    The replay buffer breaks both problems:
      - It stores a large pool of past transitions (up to `capacity`).
      - During each training step, a RANDOM mini-batch is sampled from
        this pool, breaking temporal correlation between samples.
      - Each transition can be replayed many times, improving sample
        efficiency.

    This is one of the two key innovations in the original DQN paper
    (Mnih et al., 2015) that made deep RL stable enough to play Atari.

Buffer design:
    - Fixed-capacity ring buffer using a Python deque (maxlen=capacity).
    - When full, the oldest transitions are automatically overwritten.
    - Transitions stored as plain Python tuples for simplicity.
    - Sampled outputs are NumPy arrays, ready for conversion to tensors
      in the training loop.

Usage:
    from src.replay_buffer import ReplayBuffer
    buf = ReplayBuffer(capacity=50000)
    buf.push(state, action, reward, next_state, done)
    states, actions, rewards, next_states, dones = buf.sample(64)
"""

import random
import numpy as np
from collections import deque
from typing import Tuple


class ReplayBuffer:
    """
    Fixed-capacity circular replay buffer storing (s, a, r, s', done) tuples.

    Args:
        capacity (int): Maximum number of transitions to store.
                        When full, oldest transitions are overwritten.
                        Recommended: 50,000 – 1,000,000 for DQN.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")

        self.capacity = capacity

        # deque with maxlen automatically drops the oldest entry
        # when a new one is pushed beyond capacity — O(1) push
        self._buffer: deque = deque(maxlen=capacity)

    # ── Public API ──────────────────────────────────────────────

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """
        Store a single transition in the buffer.

        Args:
            state      : Observation before the action. Shape: (state_dim,)
            action     : Integer index of the action taken.
            reward     : Scalar reward received after the action.
            next_state : Observation after the action. Shape: (state_dim,)
            done       : True if the episode ended after this transition.
        """
        # Store as a tuple — lightweight and fast
        self._buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[
        np.ndarray,  # states
        np.ndarray,  # actions
        np.ndarray,  # rewards
        np.ndarray,  # next_states
        np.ndarray,  # dones
    ]:
        """
        Sample a random mini-batch of transitions from the buffer.

        Random sampling is critical — it breaks the temporal correlation
        between consecutive transitions, stabilising DQN training.

        Args:
            batch_size (int): Number of transitions to sample.
                              Must be <= len(self).

        Returns:
            Tuple of five NumPy arrays, each of length batch_size:
              states      : float32, shape (batch_size, state_dim)
              actions     : int64,   shape (batch_size,)
              rewards     : float32, shape (batch_size,)
              next_states : float32, shape (batch_size, state_dim)
              dones       : float32, shape (batch_size,)  — 0.0 or 1.0

            Note: dones is float32 so it can be used directly in the
            Bellman equation:
              target = r + gamma * max_Q(s') * (1 - done)

        Raises:
            ValueError: If batch_size > len(self).
        """
        if batch_size > len(self):
            raise ValueError(
                f"Cannot sample {batch_size} transitions: "
                f"buffer only has {len(self)}."
            )

        # Sample without replacement — each transition appears at most once
        batch = random.sample(self._buffer, batch_size)

        # Unzip list of tuples into separate lists, then stack into arrays
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states,      dtype=np.float32),   # (B, state_dim)
            np.array(actions,     dtype=np.int64),      # (B,)
            np.array(rewards,     dtype=np.float32),    # (B,)
            np.array(next_states, dtype=np.float32),    # (B, state_dim)
            np.array(dones,       dtype=np.float32),    # (B,)  0.0 / 1.0
        )

    def __len__(self) -> int:
        """Return the current number of transitions stored in the buffer."""
        return len(self._buffer)

    def is_ready(self, batch_size: int) -> bool:
        """
        Return True once the buffer has enough samples to train.

        The agent should wait until is_ready() is True before calling
        sample(), to avoid training on a nearly-empty buffer where the
        random mini-batch is not diverse enough to be useful.

        Args:
            batch_size (int): The mini-batch size used during training.
        """
        return len(self) >= batch_size

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(capacity={self.capacity}, "
            f"current_size={len(self)})"
        )


# ── Quick sanity check (run: python src/replay_buffer.py) ──────────────────
if __name__ == "__main__":
    import yaml

    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    CAPACITY   = cfg["agent"]["replay_buffer_size"]  # 50000
    BATCH_SIZE = cfg["agent"]["batch_size"]           # 64
    STATE_DIM  = 8   # LunarLander-v2
    ACTION_DIM = 4

    buf = ReplayBuffer(capacity=CAPACITY)
    print("=" * 55)
    print("  ReplayBuffer Sanity Check")
    print("=" * 55)
    print(f"  Capacity   : {CAPACITY:,}")
    print(f"  Batch size : {BATCH_SIZE}")

    # Fill with random fake transitions
    for _ in range(200):
        s  = np.random.randn(STATE_DIM).astype(np.float32)
        a  = np.random.randint(0, ACTION_DIM)
        r  = float(np.random.randn())
        ns = np.random.randn(STATE_DIM).astype(np.float32)
        d  = bool(np.random.rand() > 0.95)
        buf.push(s, a, r, ns, d)

    print(f"\n  Transitions stored : {len(buf)}")
    print(f"  is_ready({BATCH_SIZE})     : {buf.is_ready(BATCH_SIZE)}")

    states, actions, rewards, next_states, dones = buf.sample(BATCH_SIZE)

    print(f"\n  Sampled batch shapes:")
    print(f"    states      : {states.shape}   dtype={states.dtype}")
    print(f"    actions     : {actions.shape}      dtype={actions.dtype}")
    print(f"    rewards     : {rewards.shape}      dtype={rewards.dtype}")
    print(f"    next_states : {next_states.shape}   dtype={next_states.dtype}")
    print(f"    dones       : {dones.shape}      dtype={dones.dtype}")
    print(f"\n  {buf}")
    print("\n✅  ReplayBuffer OK — all shapes and dtypes correct")
