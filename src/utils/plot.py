"""
src/utils/plot.py
------------------
Reward curve plotter — reads the training CSV and saves a PNG.

Produces two subplots:
  1. Per-episode total reward (noisy but shows raw learning signal)
  2. Rolling average reward over 100 episodes (smoothed trend)

Usage:
    python src/utils/plot.py
    # or from code:
    from src.utils.plot import plot_training_rewards
    plot_training_rewards("experiments/training_log.csv",
                          "experiments/reward_curve.png")
"""

import os
import csv
import sys
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt


def plot_training_rewards(
    log_path:  str,
    save_path: str,
    show:      bool = False,
) -> None:
    """
    Read a training CSV log and save a reward curve PNG.

    Args:
        log_path  : Path to CSV produced by TrainingLogger.
        save_path : Where to save the output PNG.
        show      : If True, also display the plot interactively.
    """
    if not os.path.exists(log_path):
        print(f"[Plot] Log file not found: {log_path}")
        return

    episodes, rewards, avg_rewards = [], [], []

    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["total_reward"]))
            avg_rewards.append(float(row["avg_reward_100"]))

    if not episodes:
        print("[Plot] Log file is empty — nothing to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("DQN Training — LunarLander-v2", fontsize=14, fontweight="bold")

    # ── Top subplot: raw episode rewards ────────────────────────
    ax1.plot(episodes, rewards, color="steelblue", alpha=0.5, linewidth=0.8,
             label="Episode reward")
    ax1.axhline(y=200, color="green", linestyle="--", linewidth=1.2,
                label="Solve threshold (200)")
    ax1.set_ylabel("Total Reward")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Bottom subplot: rolling average ─────────────────────────
    ax2.plot(episodes, avg_rewards, color="darkorange", linewidth=1.5,
             label="Avg reward (100 ep)")
    ax2.axhline(y=200, color="green", linestyle="--", linewidth=1.2,
                label="Solve threshold (200)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Avg Reward (100 ep)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Reward curve saved → {save_path}")

    if show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    log  = sys.argv[1] if len(sys.argv) > 1 else "experiments/training_log.csv"
    out  = sys.argv[2] if len(sys.argv) > 2 else "experiments/reward_curve.png"
    plot_training_rewards(log, out, show=False)
