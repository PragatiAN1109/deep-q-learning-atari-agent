"""
src/agent/dqn_agent.py
-----------------------
DQN Agent — ties together the Q-network, target network,
replay buffer, and epsilon-greedy action selection.

Responsibilities:
    - Holds TWO networks: online (trained every step) and
      target (synced every N episodes for stable Bellman targets)
    - Selects actions via epsilon-greedy exploration
    - Stores transitions into the replay buffer
    - Samples mini-batches and runs one gradient update (learn())
    - Decays epsilon after each episode
    - Saves / loads model checkpoints

The two-network trick (online + target) is the second key
innovation of the original DQN paper (Mnih et al., 2015).
Without it, the Bellman target moves every step, making
training unstable — like chasing a moving target.

Usage:
    agent = DQNAgent(state_dim=8, action_dim=4, cfg=config)
    action = agent.select_action(state)
    agent.store(state, action, reward, next_state, done)
    loss   = agent.learn()
    agent.decay_epsilon()
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.model import DQN
from src.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Network agent with experience replay and a target network.

    Args:
        state_dim  (int) : Observation vector size (e.g. 8 for LunarLander).
        action_dim (int) : Number of discrete actions (e.g. 4).
        cfg        (dict): Full config dict loaded from config.yaml.
        device     (str) : 'cpu' or 'cuda'. Auto-detected if not provided.
    """

    def __init__(
        self,
        state_dim:  int,
        action_dim: int,
        cfg:        dict,
        device:     str = None,
    ):
        self.state_dim  = state_dim
        self.action_dim = action_dim

        # ── Hyperparameters from config ─────────────────────────
        a = cfg["agent"]
        self.lr                 = a["learning_rate"]
        self.gamma              = a["gamma"]
        self.epsilon            = a["epsilon_start"]
        self.epsilon_min        = a["epsilon_end"]
        self.epsilon_decay      = a["epsilon_decay"]
        self.batch_size         = a["batch_size"]
        self.target_update_freq = a["target_update_freq"]
        hidden_size             = a["hidden_size"]
        buffer_capacity         = a["replay_buffer_size"]

        # ── Device ──────────────────────────────────────────────
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── Networks ────────────────────────────────────────────
        # Online network  : updated every training step via gradient descent
        # Target network  : copy of online net, updated every N episodes
        #                   provides stable Bellman targets
        self.online_net = DQN(state_dim, action_dim, hidden_size).to(self.device)
        self.target_net = DQN(state_dim, action_dim, hidden_size).to(self.device)

        # Initialise target net with the same weights as online net
        self.target_net.load_state_dict(self.online_net.state_dict())

        # Target net is never directly trained — inference only
        self.target_net.eval()

        # ── Optimiser & loss ────────────────────────────────────
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.lr)
        self.loss_fn   = nn.MSELoss()   # Mean Squared Error on Q-value residuals

        # ── Replay buffer ───────────────────────────────────────
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # ── Training state ──────────────────────────────────────
        self.episode_count = 0   # incremented by caller after each episode

        print(f"[DQNAgent] Device      : {self.device}")
        print(f"[DQNAgent] Online net  : {self.online_net}")
        print(f"[DQNAgent] Buffer cap  : {buffer_capacity:,}")

    # ── Action selection ────────────────────────────────────────

    def select_action(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection.

        With probability epsilon  → random action  (explore)
        With probability 1-epsilon → greedy action (exploit)

        Exploration is critical early in training when the Q-network
        has random weights and its predictions are meaningless.
        As training progresses, epsilon decays so the agent exploits
        its improving policy more often.

        Args:
            state (np.ndarray): Current environment observation. Shape: (state_dim,)

        Returns:
            int: Selected action index in [0, action_dim).
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)   # explore

        # Exploit: pick action with highest Q-value
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, state_dim)
        self.online_net.eval()
        with torch.no_grad():
            q_values = self.online_net(state_t)              # (1, action_dim)
        self.online_net.train()
        return int(q_values.argmax(dim=1).item())

    # ── Memory management ───────────────────────────────────────

    def store(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """Push one (s, a, r, s', done) transition into the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    # ── Learning step ───────────────────────────────────────────

    def learn(self) -> float | None:
        """
        Sample a mini-batch and perform one gradient update.

        Bellman update rule (DQN):
            target = r  +  gamma * max_a[ Q_target(s', a) ]  *  (1 - done)

            When done=1 (terminal state) the future term is masked to zero
            because there is no next state to bootstrap from.

        Loss:
            MSE( Q_online(s, a),  target )

        Returns:
            float: The MSE loss value for this step, or None if the buffer
                   doesn't have enough samples yet.
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return None   # not enough experience yet — skip update

        # ── Sample mini-batch ────────────────────────────────────
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        # Convert NumPy arrays → PyTorch tensors on the correct device
        states_t      = torch.FloatTensor(states).to(self.device)       # (B, state_dim)
        actions_t     = torch.LongTensor(actions).to(self.device)       # (B,)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)      # (B,)
        next_states_t = torch.FloatTensor(next_states).to(self.device)  # (B, state_dim)
        dones_t       = torch.FloatTensor(dones).to(self.device)        # (B,)

        # ── Compute current Q-values ─────────────────────────────
        # online_net outputs Q(s, a) for ALL actions → select the one taken
        # gather(1, actions_t.unsqueeze(1)) picks Q(s, action_taken)
        q_current = self.online_net(states_t)                           # (B, action_dim)
        q_current = q_current.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # (B,)

        # ── Compute Bellman target ───────────────────────────────
        # target_net provides stable bootstrap values
        # no_grad: target values are fixed labels, not part of the graph
        with torch.no_grad():
            q_next    = self.target_net(next_states_t).max(dim=1)[0]   # (B,)
            q_target  = rewards_t + self.gamma * q_next * (1.0 - dones_t)  # (B,)

        # ── Gradient update ──────────────────────────────────────
        loss = self.loss_fn(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — prevents exploding gradients in early training
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    # ── Episode bookkeeping ─────────────────────────────────────

    def decay_epsilon(self) -> None:
        """
        Multiply epsilon by epsilon_decay after each episode.
        Epsilon is clipped at epsilon_min so the agent never stops exploring.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_count += 1

    def sync_target_network(self) -> None:
        """
        Hard-copy online network weights into the target network.
        Called every `target_update_freq` episodes.
        """
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ── Checkpoint I/O ──────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save online network weights + training state to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "online_net_state":  self.online_net.state_dict(),
            "target_net_state":  self.target_net.state_dict(),
            "optimizer_state":   self.optimizer.state_dict(),
            "epsilon":           self.epsilon,
            "episode_count":     self.episode_count,
        }, path)
        print(f"[DQNAgent] Checkpoint saved → {path}")

    def load(self, path: str) -> None:
        """Load a previously saved checkpoint and resume training."""
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net_state"])
        self.target_net.load_state_dict(ckpt["target_net_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.epsilon       = ckpt["epsilon"]
        self.episode_count = ckpt["episode_count"]
        print(f"[DQNAgent] Checkpoint loaded ← {path}  "
              f"(episode {self.episode_count}, ε={self.epsilon:.4f})")
