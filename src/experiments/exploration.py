"""
src/experiments/exploration.py
--------------------------------
Alternative exploration strategies for DQN.

Implements:
  1. softmax_action()   — Boltzmann/softmax exploration
  2. epsilon_greedy_action() — the existing strategy, mirrored here
     for fair side-by-side comparison

Both share the same function signature so they can be passed as
`action_selector` to engine.run_experiment() without any other
changes to the training loop.

Why Softmax / Boltzmann exploration?
    Epsilon-greedy treats all non-greedy actions equally — any random
    action is chosen with uniform probability (1/action_dim each).
    This ignores the Q-value information for sub-optimal actions.

    Boltzmann exploration converts Q-values into a probability
    distribution using the softmax function with temperature τ:

        P(a | s) = exp(Q(s,a) / τ) / Σ_a' exp(Q(s,a') / τ)

    High τ → near-uniform distribution (lots of exploration)
    Low  τ → near-greedy distribution (mostly exploitation)
    τ → 0  → identical to greedy argmax

    Advantage: even when exploring, the agent prefers actions with
    higher Q-values rather than sampling completely at random.

Usage:
    from src.experiments.exploration import softmax_action
    from src.experiments.engine      import run_experiment

    rewards, steps, eps = run_experiment(
        action_selector   = softmax_action,
        exploration_param = 1.0,   # temperature τ
    )
"""

import random
import numpy as np
import torch
from src.model import DQN


def softmax_action(
    state:      np.ndarray,
    net:        DQN,
    temperature: float,
    action_dim: int,
    device:     torch.device,
) -> int:
    """
    Boltzmann (softmax) action selection.

    Converts Q-values to a probability distribution and samples from it.

        P(a | s) = exp(Q(s,a) / τ)  /  Σ_a' exp(Q(s,a') / τ)

    Args:
        state      : Current observation. Shape: (state_dim,)
        net        : Online Q-network.
        temperature: τ — controls exploration/exploitation trade-off.
                     Typical range: 0.1 (near-greedy) to 5.0 (near-uniform).
        action_dim : Number of discrete actions.
        device     : CPU or CUDA.

    Returns:
        int: Sampled action index.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be > 0, got {temperature}")

    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    net.eval()
    with torch.no_grad():
        q_vals = net(state_t).squeeze(0).cpu().numpy()   # shape: (action_dim,)
    net.train()

    # ── Numerically stable softmax ────────────────────────────
    # Subtract max before exp() to prevent overflow (standard trick)
    scaled = q_vals / temperature
    scaled -= scaled.max()                  # numerical stability
    probs   = np.exp(scaled)
    probs  /= probs.sum()                   # normalise to sum=1

    # Sample action according to probability distribution
    return int(np.random.choice(action_dim, p=probs))


def epsilon_greedy_action(
    state:      np.ndarray,
    net:        DQN,
    epsilon:    float,
    action_dim: int,
    device:     torch.device,
) -> int:
    """
    Standard epsilon-greedy action selection.
    Mirrored here so it can be used as a drop-in selector alongside softmax.

    Args:
        state     : Current observation.
        net       : Online Q-network.
        epsilon   : Exploration rate in [0, 1].
        action_dim: Number of discrete actions.
        device    : CPU or CUDA.

    Returns:
        int: Selected action index.
    """
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)

    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    net.eval()
    with torch.no_grad():
        action = int(net(state_t).argmax(dim=1).item())
    net.train()
    return action
