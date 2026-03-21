"""
src/test_env.py
---------------
Smoke test for the Gymnasium environment.

Purpose:
    Verifies the environment loads correctly, runs one full episode
    using random actions, and prints key diagnostics. Run this before
    any training to confirm your setup is working end-to-end.

Usage:
    python src/test_env.py

Environment Choice — LunarLander-v2:
    - Observation : 8-dimensional continuous vector
                    (x pos, y pos, x vel, y vel, angle, angular vel,
                     left leg contact, right leg contact)
    - Action space: Discrete(4)
                    0 = do nothing
                    1 = fire left orientation engine
                    2 = fire main engine
                    3 = fire right orientation engine
    - Reward      : +100 to +140 for successful landing,
                    -100 for crashing, small penalties for fuel use
    - Solved when : average reward >= 200 over 100 consecutive episodes

    Why LunarLander-v2 over raw Atari ROM envs?
    * No ROM licensing issues (Atari ROMs require separate install)
    * Trains on CPU in minutes vs hours for pixel-based Atari
    * Clean vector observation — focuses on DQN logic, not preprocessing
    * Visually compelling for demos and video recordings
    * Well-understood benchmark with published DQN results to compare
"""

import gymnasium as gym
import numpy as np
import yaml
import sys
import os


def load_config(path: str = "config.yaml") -> dict:
    """
    Load the central YAML configuration file.

    Args:
        path: Relative or absolute path to config.yaml

    Returns:
        dict: Parsed configuration dictionary
    """
    if not os.path.exists(path):
        print(f"[WARN] config.yaml not found at '{path}', using defaults.")
        return {
            "environment": {"name": "LunarLander-v2", "seed": 42}
        }
    with open(path, "r") as f:
        return yaml.safe_load(f)


def print_space_info(env: gym.Env) -> None:
    """Print detailed observation and action space metadata."""
    obs_space = env.observation_space
    act_space = env.action_space

    print(f"\n{'─'*60}")
    print(f"  Observation Space : {obs_space}")
    print(f"  Observation Shape : {obs_space.shape}")
    print(f"  Obs Low  (first4) : {obs_space.low[:4]}")
    print(f"  Obs High (first4) : {obs_space.high[:4]}")
    print(f"  Action Space      : {act_space}")
    print(f"  Number of Actions : {act_space.n}")
    print(f"{'─'*60}\n")


def run_random_episode(env_name: str, seed: int = 42) -> None:
    """
    Run a single episode with fully random actions and print diagnostics.

    This validates:
      - Environment loads without errors
      - Reset returns correct observation shape
      - Step returns (obs, reward, terminated, truncated, info)
      - Episode terminates or truncates correctly

    Args:
        env_name: Gymnasium environment ID (e.g. 'LunarLander-v2')
        seed    : Random seed for reproducibility
    """
    print("=" * 60)
    print(f"  Environment Smoke Test")
    print(f"  Env  : {env_name}")
    print(f"  Seed : {seed}")
    print("=" * 60)

    # ── Create environment ──────────────────────────────────────
    env = gym.make(env_name, render_mode=None)
    print_space_info(env)

    # ── Reset with seed for reproducibility ────────────────────
    obs, info = env.reset(seed=seed)
    print(f"[RESET] Initial observation:")
    print(f"        Full vector : {np.round(obs, 4)}")
    print(f"        Shape       : {obs.shape}")
    print(f"        Info        : {info}\n")

    # ── Run one episode with random actions ─────────────────────
    total_reward = 0.0
    step         = 0
    terminated   = False
    truncated    = False

    print(f"{'Step':<6} {'Action':<8} {'Reward':>9} {'Total':>9} {'Done':<6} {'Trunc':<6}")
    print("─" * 50)

    while True:
        action = env.action_space.sample()  # Uniform random action
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        done = terminated or truncated

        # Print every 100 steps and on final step to avoid log flood
        if step % 100 == 0 or done:
            print(
                f"{step:<6} {action:<8} {reward:>9.3f} "
                f"{total_reward:>9.3f} {str(terminated):<6} {str(truncated):<6}"
            )

        step += 1
        if done:
            break

    print("─" * 50)
    print(f"\n[RESULT] Episode finished after {step} steps")
    print(f"[RESULT] Total cumulative reward : {total_reward:.3f}")
    print(f"[RESULT] Terminated (crash/land) : {terminated}")
    print(f"[RESULT] Truncated (step limit)  : {truncated}")

    if total_reward >= 200:
        print("\n🎉  Random policy achieved solve threshold (>=200) — lucky run!")
    elif total_reward >= 0:
        print("\n✅  Environment working correctly. Random policy scored >= 0.")
    else:
        print("\n✅  Environment working correctly. (Negative score is normal for random policy.)")

    env.close()
    print("\n🚀  Setup verified. Ready to train the DQN agent.\n")


if __name__ == "__main__":
    # Support optional config path as CLI arg: python src/test_env.py path/to/config.yaml
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = load_config(config_path)

    env_name = cfg["environment"]["name"]
    seed     = cfg["environment"]["seed"]

    run_random_episode(env_name, seed)
