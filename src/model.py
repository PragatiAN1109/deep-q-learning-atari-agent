"""
src/model.py
------------
Deep Q-Network (DQN) model definition.

Architecture choice — MLP (Multi-Layer Perceptron):
    The environment is LunarLander-v3, which provides an 8-dimensional
    continuous vector as the observation (position, velocity, angle, etc.).
    There are NO raw pixel frames involved, so a CNN is NOT needed here.

    A fully-connected MLP is the correct and efficient choice:
        Input  : 8-dim state vector
        Hidden : Two layers of 128 neurons each (ReLU activations)
        Output : 4 Q-values, one per discrete action

    The output Q-values represent the expected cumulative reward for
    taking each action from the current state under the learned policy.

Usage:
    from src.model import DQN
    model = DQN(state_dim=8, action_dim=4, hidden_size=128)
    q_values = model(state_tensor)  # shape: (batch, 4)
"""

import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Deep Q-Network using a fully-connected MLP architecture.

    Suitable for environments with low-dimensional vector observations
    such as LunarLander-v3 (obs_dim=8, action_dim=4).

    Args:
        state_dim   (int): Dimensionality of the input observation vector.
        action_dim  (int): Number of discrete actions in the action space.
        hidden_size (int): Number of neurons in each hidden layer.
                           Default: 128 (matches config.yaml agent.hidden_size)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128):
        super(DQN, self).__init__()

        # ── Store dims for reference (useful in logging/debugging) ──
        self.state_dim   = state_dim
        self.action_dim  = action_dim
        self.hidden_size = hidden_size

        # ── Network architecture ────────────────────────────────
        #
        #  Layer 1: state_dim  → hidden_size   (e.g. 8   → 128)
        #  Layer 2: hidden_size → hidden_size  (e.g. 128 → 128)
        #  Layer 3: hidden_size → action_dim   (e.g. 128 → 4  )
        #
        #  ReLU activations after hidden layers introduce non-linearity,
        #  allowing the network to approximate complex Q-value functions.
        #  No activation on the output — Q-values can be any real number.
        #
        self.network = nn.Sequential(
            # Layer 1 — input projection
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),

            # Layer 2 — feature extraction
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),

            # Layer 3 — Q-value head (one value per action, no activation)
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: maps a batch of states to Q-values.

        Args:
            state (torch.Tensor): Observation tensor.
                                  Shape: (batch_size, state_dim)
                                  e.g.   (64, 8) during training
                                  e.g.   (1, 8)  during inference

        Returns:
            torch.Tensor: Q-value estimates for every action.
                          Shape: (batch_size, action_dim)
                          e.g.   (64, 4)
                          q_values[i][a] = estimated return for action a
                          in the i-th sample of the batch.
        """
        return self.network(state)  # shape: (batch_size, action_dim)


    def __repr__(self) -> str:
        """Human-readable summary of the model."""
        return (
            f"DQN(state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, "
            f"hidden_size={self.hidden_size})\n"
            f"{self.network}"
        )


# ── Quick sanity check (run: python src/model.py) ──────────────────────────
if __name__ == "__main__":
    import yaml

    # Load config so dims stay in sync with the rest of the project
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    hidden_size = cfg["agent"]["hidden_size"]

    # LunarLander-v3 has obs_dim=8, action_dim=4
    STATE_DIM  = 8
    ACTION_DIM = 4

    model = DQN(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_size=hidden_size)
    print("=" * 55)
    print("  DQN Model Sanity Check")
    print("=" * 55)
    print(model)

    # Test with a random batch (batch_size=4) to verify shapes
    dummy_batch = torch.randn(4, STATE_DIM)          # (4, 8)
    q_vals      = model(dummy_batch)                  # (4, 4)

    print(f"\nInput  shape : {dummy_batch.shape}  → (batch, state_dim)")
    print(f"Output shape : {q_vals.shape}   → (batch, action_dim)")
    print(f"\nSample Q-values (batch item 0): {q_vals[0].detach().numpy()}")
    print("\n✅  Model forward pass OK")

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅  Trainable parameters      : {total_params:,}")
