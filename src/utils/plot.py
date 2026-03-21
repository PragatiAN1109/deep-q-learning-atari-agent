"""
src/utils/plot.py
------------------
Matplotlib-based training visualisation utilities.

Reads per-episode CSV logs produced by :class:`src.utils.logger.TrainingLogger`
and saves publication-quality PNG figures to ``experiments/plots/``.

All plots use the ``Agg`` backend so they work safely on headless servers
(no display required).

Functions
---------
plot_training_rewards
    Two-panel plot: raw episode rewards (top) + 100-ep rolling average (bottom).
plot_loss_curve
    Single-panel plot of mean training loss per episode.
plot_full_dashboard
    Three-panel dashboard: rewards, rolling average, and loss in one figure.

Usage::

    from src.utils.plot import plot_training_rewards, plot_full_dashboard

    plot_training_rewards(
        log_path  = "experiments/metrics/training_log.csv",
        save_path = "experiments/plots/reward_curve.png",
    )
"""

import os
import csv
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt

from src.utils.logger import get_logger

_log = get_logger(__name__)

# ── Shared style constants ────────────────────────────────────
_SOLVE_THRESHOLD = 200.0        # LunarLander-v2 solve score
_FIG_DPI         = 150
_COLOR_RAW       = "steelblue"
_COLOR_AVG       = "darkorange"
_COLOR_LOSS      = "firebrick"
_COLOR_SOLVE     = "green"


def _read_log(log_path: str) -> dict:
    """
    Parse a CSV training log into column lists.

    Args:
        log_path: Path to CSV file written by ``TrainingLogger``.

    Returns:
        Dict with keys ``episodes``, ``rewards``, ``avg_rewards``, ``losses``.
        Returns empty lists if the file does not exist or is empty.
    """
    result = {"episodes": [], "rewards": [], "avg_rewards": [], "losses": []}
    if not os.path.exists(log_path):
        _log.warning("Log file not found: %s", log_path)
        return result

    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result["episodes"].append(int(row["episode"]))
            result["rewards"].append(float(row["total_reward"]))
            result["avg_rewards"].append(float(row["avg_reward_100"]))
            loss_val = row.get("loss", "")
            result["losses"].append(float(loss_val) if loss_val else None)

    return result


def plot_training_rewards(
    log_path:  str,
    save_path: str,
    show:      bool = False,
) -> None:
    """
    Two-panel reward figure: raw per-episode rewards and 100-ep rolling average.

    The solve threshold line (reward = 200) is drawn on both panels for
    easy visual confirmation of whether the agent has converged.

    Args:
        log_path:  Path to training CSV log.
        save_path: Output PNG path. Parent dirs are created automatically.
        show:      If ``True``, display interactively after saving.
    """
    data = _read_log(log_path)
    if not data["episodes"]:
        _log.warning("Empty log — skipping reward plot.")
        return

    eps, rews, avgs = data["episodes"], data["rewards"], data["avg_rewards"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("DQN Training — LunarLander-v2", fontsize=14, fontweight="bold")

    # ── Top: raw episode rewards ──────────────────────────────
    ax1.plot(eps, rews, color=_COLOR_RAW, alpha=0.5,
             linewidth=0.8, label="Episode reward")
    ax1.axhline(_SOLVE_THRESHOLD, color=_COLOR_SOLVE, linestyle="--",
                linewidth=1.2, label=f"Solve threshold ({_SOLVE_THRESHOLD:.0f})")
    ax1.set_ylabel("Total Reward")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Bottom: rolling average ───────────────────────────────
    ax2.plot(eps, avgs, color=_COLOR_AVG, linewidth=1.5,
             label="Avg reward (100 ep)")
    ax2.axhline(_SOLVE_THRESHOLD, color=_COLOR_SOLVE, linestyle="--",
                linewidth=1.2, label=f"Solve threshold ({_SOLVE_THRESHOLD:.0f})")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Avg Reward (100 ep)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    _save_fig(fig, save_path, show)


def plot_loss_curve(
    log_path:  str,
    save_path: str,
    show:      bool = False,
) -> None:
    """
    Single-panel plot of mean Bellman MSE loss per episode.

    Episodes before the replay buffer is ready (loss = None) are skipped.
    A horizontal guide line marks loss = 1.0 as a rough convergence reference.

    Args:
        log_path:  Path to training CSV log.
        save_path: Output PNG path.
        show:      If ``True``, display interactively after saving.
    """
    data = _read_log(log_path)
    if not data["episodes"]:
        return

    # Filter out None-loss episodes (buffer not yet ready)
    pairs = [(e, l) for e, l in zip(data["episodes"], data["losses"])
             if l is not None]
    if not pairs:
        _log.warning("No loss values in log — skipping loss plot.")
        return

    eps, losses = zip(*pairs)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eps, losses, color=_COLOR_LOSS, linewidth=1.0, alpha=0.8,
            label="Mean MSE loss")
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6,
               label="Loss = 1.0 (reference)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("MSE Loss")
    ax.set_title("DQN Training Loss — LunarLander-v2",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    _save_fig(fig, save_path, show)


def plot_full_dashboard(
    log_path:  str,
    save_path: str,
    show:      bool = False,
) -> None:
    """
    Three-panel training dashboard: episode rewards, rolling average, and loss.

    Combines all key training signals in one figure, suitable for including
    in project reports or the README.

    Args:
        log_path:  Path to training CSV log.
        save_path: Output PNG path.
        show:      If ``True``, display interactively after saving.
    """
    data = _read_log(log_path)
    if not data["episodes"]:
        return

    eps  = data["episodes"]
    rews = data["rewards"]
    avgs = data["avg_rewards"]
    loss_pairs = [(e, l) for e, l in zip(eps, data["losses"]) if l is not None]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=False)
    fig.suptitle("DQN Training Dashboard — LunarLander-v2",
                 fontsize=14, fontweight="bold")

    # Panel 1: raw rewards
    axes[0].plot(eps, rews, color=_COLOR_RAW, alpha=0.5,
                 linewidth=0.8, label="Episode reward")
    axes[0].axhline(_SOLVE_THRESHOLD, color=_COLOR_SOLVE,
                    linestyle="--", linewidth=1.0, alpha=0.8)
    axes[0].set_ylabel("Episode Reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    # Panel 2: rolling average
    axes[1].plot(eps, avgs, color=_COLOR_AVG, linewidth=1.5,
                 label="Avg reward (100 ep)")
    axes[1].axhline(_SOLVE_THRESHOLD, color=_COLOR_SOLVE,
                    linestyle="--", linewidth=1.0,
                    label=f"Solve ({_SOLVE_THRESHOLD:.0f})", alpha=0.8)
    axes[1].set_ylabel("Avg Reward (100 ep)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    # Panel 3: loss
    if loss_pairs:
        le, ll = zip(*loss_pairs)
        axes[2].plot(le, ll, color=_COLOR_LOSS, linewidth=1.0,
                     alpha=0.8, label="MSE loss")
        axes[2].set_ylabel("Training Loss")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "No loss data yet",
                     ha="center", va="center", transform=axes[2].transAxes)

    axes[2].set_xlabel("Episode")
    plt.tight_layout()
    _save_fig(fig, save_path, show)


def _save_fig(fig: plt.Figure, save_path: str, show: bool) -> None:
    """Save figure to disk and optionally display it."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    _log.info("Plot saved → %s", save_path)
    if show:
        plt.show()
    plt.close(fig)


# ── CLI entry point ───────────────────────────────────────────
if __name__ == "__main__":
    import sys
    log  = sys.argv[1] if len(sys.argv) > 1 else "experiments/metrics/training_log.csv"
    out  = sys.argv[2] if len(sys.argv) > 2 else "experiments/plots/reward_curve.png"
    plot_training_rewards(log, out, show=False)
    plot_full_dashboard(log, out.replace("reward_curve", "dashboard"), show=False)
