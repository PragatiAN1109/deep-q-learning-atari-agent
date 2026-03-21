"""
src/training/evaluate.py
-------------------------
Evaluate a saved DQN model — no training, no exploration.

Loads a checkpoint, runs N deterministic episodes (epsilon=0),
prints per-episode rewards, and reports mean ± std.

Usage:
    python src/training/evaluate.py --model models/dqn_best.pth
    python src/training/evaluate.py --model models/dqn_best.pth --episodes 20
    python src/training/evaluate.py --model models/dqn_best.pth --render
"""

import sys
import os
import argparse
import yaml
import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.agent.dqn_agent import DQNAgent
from src.utils.seed      import set_global_seed


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def evaluate(model_path: str, num_episodes: int, render: bool, cfg: dict) -> None:
    """
    Run deterministic evaluation episodes with a loaded DQN checkpoint.

    Args:
        model_path   : Path to saved .pth checkpoint.
        num_episodes : Number of evaluation episodes to run.
        render       : If True, open a live render window.
        cfg          : Config dictionary.
    """
    set_global_seed(cfg["environment"]["seed"])

    render_mode = "human" if render else None
    env = gym.make(cfg["environment"]["name"], render_mode=render_mode)

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Load agent and set epsilon=0 (pure exploitation, no exploration)
    agent         = DQNAgent(state_dim=state_dim, action_dim=action_dim, cfg=cfg)
    agent.load(model_path)
    agent.epsilon = 0.0    # greedy policy only during evaluation

    print(f"\n{'='*55}")
    print(f"  Evaluating: {model_path}")
    print(f"  Episodes  : {num_episodes}")
    print(f"  Epsilon   : {agent.epsilon}  (pure exploitation)")
    print(f"{'='*55}\n")

    rewards = []

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset(seed=cfg["environment"]["seed"] + ep)
        state    = np.array(state, dtype=np.float32)

        total_reward = 0.0
        steps        = 0

        while True:
            action                              = agent.select_action(state)
            next_state, reward, term, trunc, _  = env.step(action)
            state        = np.array(next_state, dtype=np.float32)
            total_reward += reward
            steps        += 1
            if term or trunc:
                break

        rewards.append(total_reward)
        print(f"  Episode {ep:>3} | Reward: {total_reward:>8.2f} | Steps: {steps}")

    env.close()

    mean_r = float(np.mean(rewards))
    std_r  = float(np.std(rewards))
    print(f"\n  Mean reward : {mean_r:.2f}  ±  {std_r:.2f}")
    solved = "✅  SOLVED" if mean_r >= cfg["training"]["solve_score"] else "❌  Not yet solved"
    print(f"  Status      : {solved}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN agent")
    parser.add_argument("--model",    type=str, required=True,
                        help="Path to .pth checkpoint (e.g. models/dqn_best.pth)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes (default: 10)")
    parser.add_argument("--render",   action="store_true",
                        help="Render the environment visually")
    parser.add_argument("--config",   type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(args.model, args.episodes, args.render, cfg)
