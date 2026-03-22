"""
src/training/run_experiments.py
--------------------------------
Automated experiment runner — trains the DQN agent across multiple
config files and produces a side-by-side comparison plot.

This script:
  1. Accepts a list of YAML config files (or a glob pattern)
  2. Runs full training for each config sequentially
  3. Reads the resulting CSV logs
  4. Produces a single comparison plot:
       - Top    : smoothed avg reward (100 ep) per experiment
       - Bottom : epsilon decay curves per experiment
  5. Saves the comparison PNG to experiments/comparison.png

Usage:
    # Run all predefined experiments:
    python src/training/run_experiments.py

    # Run specific configs:
    python src/training/run_experiments.py \\
        --configs experiments/exp_baseline.yaml \\
                  experiments/exp_low_lr.yaml \\
                  experiments/exp_fast_epsilon.yaml

    # Skip training (just re-plot from existing CSV logs):
    python src/training/run_experiments.py --plot-only
"""

import sys
import os
import argparse
import csv
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Make src/ importable from repo root ───────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.training.train import train, load_config

# ── Default experiment suite ──────────────────────────────────
DEFAULT_CONFIGS = [
    "experiments/exp_baseline.yaml",
    "experiments/exp_low_lr.yaml",
    "experiments/exp_fast_epsilon.yaml",
    "experiments/exp_slow_epsilon.yaml",
]

# Human-readable labels for the plot legend
LABELS = {
    "exp_baseline.yaml":      "Baseline (LR=5e-4, ε-decay=0.995)",
    "exp_low_lr.yaml":        "Low LR  (LR=1e-4, ε-decay=0.995)",
    "exp_fast_epsilon.yaml":  "Fast ε  (LR=5e-4, ε-decay=0.990)",
    "exp_slow_epsilon.yaml":  "Slow ε  (LR=5e-4, ε-decay=0.998)",
}


def read_log(log_path: str):
    """
    Read a training CSV log and return (episodes, avg_rewards, epsilons).

    Returns:
        Tuple of three lists, or (None, None, None) if file missing/empty.
    """
    if not os.path.exists(log_path):
        print(f"  [WARN] Log not found: {log_path}")
        return None, None, None

    episodes, avg_rewards, epsilons = [], [], []
    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            avg_rewards.append(float(row["avg_reward_100"]))
            epsilons.append(float(row["epsilon"]))

    if not episodes:
        print(f"  [WARN] Log is empty: {log_path}")
        return None, None, None

    return episodes, avg_rewards, epsilons


def plot_comparison(configs: list, output_path: str) -> None:
    """
    Read all experiment logs and produce a side-by-side comparison plot.

    Args:
        configs     : List of YAML config file paths.
        output_path : Where to save the PNG.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        "DQN Hyperparameter Comparison — LunarLander-v3",
        fontsize=14, fontweight="bold"
    )

    colors = ["steelblue", "darkorange", "green", "red", "purple", "brown"]

    for i, cfg_path in enumerate(configs):
        if not os.path.exists(cfg_path):
            print(f"  [SKIP] Config not found: {cfg_path}")
            continue

        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        log_path = cfg["paths"]["log_file"]
        label    = LABELS.get(os.path.basename(cfg_path), os.path.basename(cfg_path))
        color    = colors[i % len(colors)]

        episodes, avg_rewards, epsilons = read_log(log_path)
        if episodes is None:
            continue

        # Top plot: smoothed avg reward
        ax1.plot(episodes, avg_rewards, color=color, linewidth=1.5, label=label)

        # Bottom plot: epsilon decay curve
        ax2.plot(episodes, epsilons, color=color, linewidth=1.5, label=label)

    # Formatting
    ax1.axhline(y=200, color="black", linestyle="--", linewidth=1,
                label="Solve threshold (200)", alpha=0.6)
    ax1.set_ylabel("Avg Reward (100 episodes)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Epsilon (exploration rate)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Plot] Comparison chart saved → {output_path}")


def run_all(configs: list, plot_only: bool) -> None:
    """
    Train each config sequentially, then plot all results together.

    Args:
        configs   : List of YAML config paths to run.
        plot_only : If True, skip training and go straight to plotting.
    """
    print("\n" + "=" * 65)
    print(f"  Running {len(configs)} experiment(s)")
    print("=" * 65)

    for i, cfg_path in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Config: {cfg_path}")

        if not os.path.exists(cfg_path):
            print(f"  [ERROR] Config not found — skipping.")
            continue

        cfg = load_config(cfg_path)

        if plot_only:
            log = cfg["paths"]["log_file"]
            print(f"  [SKIP training] Will read log: {log}")
            continue

        print(f"  Starting training...")
        try:
            train(cfg)
        except Exception as e:
            print(f"  [ERROR] Training failed for {cfg_path}: {e}")
            continue

    # Always produce the comparison plot at the end
    plot_comparison(configs, output_path="experiments/comparison.png")

    print("\n✅  All experiments complete.")
    print("   Comparison plot → experiments/comparison.png\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run DQN hyperparameter experiments and compare results"
    )
    parser.add_argument(
        "--configs", nargs="+", default=DEFAULT_CONFIGS,
        help="Paths to YAML config files (default: all 4 preset experiments)"
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Skip training; just re-generate comparison plot from existing logs"
    )
    args = parser.parse_args()
    run_all(args.configs, args.plot_only)
