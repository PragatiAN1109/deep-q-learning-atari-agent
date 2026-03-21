"""
src/experiments/plot_utils.py
-------------------------------
Shared plotting utilities for experiment result visualisation.

All plots use matplotlib with the Agg backend (safe for headless/CI).
Plots are saved to experiments/plots/.

Functions:
  plot_reward_curves()   — multi-line reward vs episode comparison
  plot_epsilon_curves()  — epsilon decay curves per experiment
  plot_comparison_grid() — 2-panel grid (reward top, epsilon bottom)
"""

import os
import numpy as np
from typing import Dict, List

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


COLORS = ["steelblue", "darkorange", "green", "red",
          "purple", "brown", "teal", "olive"]
PLOT_DIR = "experiments/plots"


def _moving_avg(values: List[float], window: int = 30) -> List[float]:
    result = []
    for i in range(len(values)):
        result.append(float(np.mean(values[max(0, i-window+1): i+1])))
    return result


def plot_reward_curves(
    results: Dict[str, List[float]],
    title:   str  = "Reward vs Episode",
    fname:   str  = "reward_curves.png",
    window:  int  = 30,
    out_dir: str  = PLOT_DIR,
) -> str:
    """
    Plot smoothed reward curves for multiple experiment runs.

    Args:
        results : {label: [reward per episode]} dict.
        title   : Plot title.
        fname   : Output filename (saved inside out_dir).
        window  : Moving average window size.
        out_dir : Directory to save the plot.

    Returns:
        str: Full path to the saved PNG.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[Plot] matplotlib not available — skipping.")
        return ""

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (label, rewards) in enumerate(results.items()):
        episodes = list(range(1, len(rewards) + 1))
        color    = COLORS[i % len(COLORS)]
        # Raw rewards — faint background line
        ax.plot(episodes, rewards, color=color, alpha=0.15, linewidth=0.7)
        # Smoothed moving average — prominent line
        ax.plot(episodes, _moving_avg(rewards, window),
                color=color, linewidth=2.0, label=label)

    ax.axhline(y=200, color="black", linestyle="--",
               linewidth=1.0, alpha=0.6, label="Solve threshold (200)")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel(f"Reward (mov-avg {window})", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Saved → {out_path}")
    return out_path


def plot_epsilon_curves(
    results: Dict[str, List[float]],
    title:   str = "Epsilon Decay vs Episode",
    fname:   str = "epsilon_curves.png",
    out_dir: str = PLOT_DIR,
) -> str:
    """
    Plot epsilon decay curves for multiple experiment configurations.

    Args:
        results : {label: [epsilon per episode]} dict.
        title   : Plot title.
        fname   : Output filename.
        out_dir : Directory to save the plot.

    Returns:
        str: Full path to the saved PNG.
    """
    if not MATPLOTLIB_AVAILABLE:
        return ""

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4))

    for i, (label, epsilons) in enumerate(results.items()):
        episodes = list(range(1, len(epsilons) + 1))
        ax.plot(episodes, epsilons,
                color=COLORS[i % len(COLORS)],
                linewidth=2.0, label=label)

    ax.set_xlabel("Episode",      fontsize=11)
    ax.set_ylabel("Epsilon (ε)",  fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Saved → {out_path}")
    return out_path


def plot_comparison_grid(
    reward_results:  Dict[str, List[float]],
    epsilon_results: Dict[str, List[float]],
    title:  str = "DQN Experiment Comparison",
    fname:  str = "comparison_grid.png",
    window: int = 30,
    out_dir:str = PLOT_DIR,
) -> str:
    """
    Two-panel plot: reward curves (top) + epsilon curves (bottom).

    Args:
        reward_results  : {label: [reward per episode]}.
        epsilon_results : {label: [epsilon per episode]}.
        title   : Overall figure title.
        fname   : Output filename.
        window  : Moving average window for rewards.
        out_dir : Directory to save the plot.

    Returns:
        str: Full path to the saved PNG.
    """
    if not MATPLOTLIB_AVAILABLE:
        return ""

    os.makedirs(out_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ── Top: reward curves ────────────────────────────────────
    for i, (label, rewards) in enumerate(reward_results.items()):
        episodes = list(range(1, len(rewards) + 1))
        color    = COLORS[i % len(COLORS)]
        ax1.plot(episodes, rewards, color=color, alpha=0.15, linewidth=0.7)
        ax1.plot(episodes, _moving_avg(rewards, window),
                 color=color, linewidth=2.0, label=label)
    ax1.axhline(y=200, color="black", linestyle="--",
                linewidth=1.0, alpha=0.6, label="Solve (200)")
    ax1.set_ylabel(f"Reward (mov-avg {window})", fontsize=10)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.25)

    # ── Bottom: epsilon curves ────────────────────────────────
    for i, (label, epsilons) in enumerate(epsilon_results.items()):
        episodes = list(range(1, len(epsilons) + 1))
        ax2.plot(episodes, epsilons,
                 color=COLORS[i % len(COLORS)],
                 linewidth=2.0, label=label)
    ax2.set_xlabel("Episode",     fontsize=10)
    ax2.set_ylabel("Epsilon (ε)", fontsize=10)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Saved → {out_path}")
    return out_path
