"""
src/training/train.py
----------------------
Main DQN training loop for LunarLander-v2.

What this script does, step by step:
    1. Load config.yaml (hyperparameters, paths, env name)
    2. Set global random seeds for reproducibility
    3. Create the Gymnasium environment
    4. Instantiate the DQNAgent (online net + target net + replay buffer)
    5. Run N episodes:
         a. Reset env, get initial state
         b. At each step: select action (ε-greedy), step env,
            store transition, call agent.learn()
         c. After episode: decay epsilon, sync target net every K eps,
            log metrics, save checkpoint every M eps
    6. Plot reward curves from the CSV log
    7. Save the final model

Usage:
    # From repo root:
    python src/training/train.py
    python src/training/train.py --config experiments/exp_low_lr.yaml

Outputs:
    models/dqn_checkpoint_ep<N>.pth  — periodic checkpoints
    models/dqn_best.pth              — best model so far
    experiments/training_log.csv     — per-episode metrics
    experiments/reward_curve.png     — reward curve plot
"""

import sys
import os
import argparse
import yaml
import numpy as np
import gymnasium as gym
from collections import deque

# ── Make sure src/ is importable when running from repo root ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.agent.dqn_agent import DQNAgent
from src.utils.seed      import set_global_seed
from src.utils.logger    import TrainingLogger
from src.utils.plot      import plot_training_rewards


# ─────────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    """Load YAML config file and return as dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env(cfg: dict) -> gym.Env:
    """Create and return the Gymnasium environment from config."""
    env_name    = cfg["environment"]["name"]
    render_mode = cfg["environment"].get("render_mode", None)
    env = gym.make(env_name, render_mode=render_mode)
    print(f"[Env] Created: {env_name}  |  "
          f"obs={env.observation_space.shape}  |  "
          f"actions={env.action_space.n}")
    return env


# ─────────────────────────────────────────────────────────────
def train(cfg: dict) -> None:
    """
    Full DQN training loop.

    Args:
        cfg: Parsed config dictionary from config.yaml.
    """
    # ── 1. Seeds ─────────────────────────────────────────────
    set_global_seed(cfg["environment"]["seed"])

    # ── 2. Environment ───────────────────────────────────────
    env = make_env(cfg)

    state_dim  = env.observation_space.shape[0]   # 8
    action_dim = env.action_space.n               # 4

    # ── 3. Agent ─────────────────────────────────────────────
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, cfg=cfg)

    # ── 4. Logger + paths ────────────────────────────────────
    log_path   = cfg["paths"]["log_file"]
    model_dir  = cfg["paths"]["model_dir"]
    exp_dir    = cfg["paths"]["experiment_dir"]
    logger     = TrainingLogger(log_path, log_freq=cfg["training"]["log_freq"])

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(exp_dir,   exist_ok=True)

    # ── 5. Training config ───────────────────────────────────
    max_episodes    = cfg["training"]["max_episodes"]
    max_steps       = cfg["training"]["max_steps_per_episode"]
    solve_score     = cfg["training"]["solve_score"]
    ckpt_freq       = cfg["training"]["checkpoint_freq"]
    target_upd_freq = cfg["agent"]["target_update_freq"]

    # Rolling window for solve detection
    recent_rewards: deque = deque(maxlen=100)
    best_avg_reward = -float("inf")

    print("\n" + "=" * 60)
    print(f"  DQN Training — {cfg['environment']['name']}")
    print(f"  Episodes : {max_episodes}  |  Max steps/ep : {max_steps}")
    print(f"  Solve at : avg reward >= {solve_score} over 100 eps")
    print("=" * 60 + "\n")

    # ── 6. Episode loop ──────────────────────────────────────
    for episode in range(1, max_episodes + 1):

        state, _ = env.reset(seed=cfg["environment"]["seed"] + episode)
        state    = np.array(state, dtype=np.float32)

        episode_reward = 0.0
        episode_loss   = []
        steps          = 0

        # ── Inner step loop ──────────────────────────────────
        for step in range(max_steps):
            # a) Select action (ε-greedy)
            action = agent.select_action(state)

            # b) Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            done       = terminated or truncated

            # c) Store transition
            agent.store(state, action, reward, next_state, done)

            # d) Learn (returns None until buffer has enough samples)
            loss = agent.learn()
            if loss is not None:
                episode_loss.append(loss)

            episode_reward += reward
            state           = next_state
            steps          += 1

            if done:
                break

        # ── Post-episode bookkeeping ─────────────────────────

        # Decay epsilon
        agent.decay_epsilon()

        # Sync target network every N episodes
        if episode % target_upd_freq == 0:
            agent.sync_target_network()

        # Rolling average over last 100 episodes
        recent_rewards.append(episode_reward)
        avg_reward = float(np.mean(recent_rewards))
        mean_loss  = float(np.mean(episode_loss)) if episode_loss else None

        # Log to CSV + console
        logger.log(
            episode=episode,
            reward=episode_reward,
            avg_reward=avg_reward,
            epsilon=agent.epsilon,
            steps=steps,
            loss=mean_loss,
        )

        # Save best model
        if avg_reward > best_avg_reward and len(recent_rewards) == 100:
            best_avg_reward = avg_reward
            agent.save(os.path.join(model_dir, "dqn_best.pth"))

        # Periodic checkpoint
        if episode % ckpt_freq == 0:
            ckpt_path = os.path.join(model_dir, f"dqn_checkpoint_ep{episode}.pth")
            agent.save(ckpt_path)

        # Solve check
        if avg_reward >= solve_score and len(recent_rewards) == 100:
            print(f"\n🎉  Solved at episode {episode}! "
                  f"Avg reward = {avg_reward:.2f}\n")
            break

    # ── 7. Cleanup ───────────────────────────────────────────
    env.close()
    logger.close()

    # Save final model
    agent.save(os.path.join(model_dir, "dqn_final.pth"))

    # Plot reward curves
    plot_training_rewards(
        log_path  = log_path,
        save_path = os.path.join(exp_dir, "reward_curve.png"),
    )

    print("\n✅  Training complete.")
    print(f"   Best avg reward (100 ep) : {best_avg_reward:.2f}")
    print(f"   Models saved in          : {model_dir}")
    print(f"   Log + plot saved in      : {exp_dir}\n")


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on LunarLander-v2")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config YAML file (default: config.yaml)"
    )
    args = parser.parse_args()
    cfg  = load_config(args.config)
    train(cfg)
