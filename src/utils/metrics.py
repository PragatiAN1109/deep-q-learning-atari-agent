"""
src/utils/metrics.py
---------------------
Reusable metrics utility for DQN experiment analysis.

Provides:
  - compute_metrics()      : compute avg reward, avg steps, moving avg
  - save_metrics_csv()     : write results dict to CSV
  - print_metrics_table()  : formatted console summary

Used by src/experiments/run_all.py and any training script.

Usage:
    from src.utils.metrics import compute_metrics, save_metrics_csv
    stats = compute_metrics(rewards, steps_list, window=50)
    save_metrics_csv(stats, "experiments/my_run/metrics_summary.csv")
"""

import os
import csv
import numpy as np
from typing import List, Dict


def moving_average(values: List[float], window: int) -> List[float]:
    """
    Compute a rolling/moving average over a list of values.

    Args:
        values : List of scalar values (e.g. per-episode rewards).
        window : Number of past episodes to average over.

    Returns:
        List of moving averages, same length as input.
        Early entries use whatever data is available (expanding window).
    """
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(float(np.mean(values[start: i + 1])))
    return result


def compute_metrics(
    rewards:    List[float],
    steps_list: List[int],
    window:     int = 50,
) -> Dict:
    """
    Compute standard DQN performance metrics from episode history.

    Args:
        rewards    : Per-episode total reward list.
        steps_list : Per-episode step count list.
        window     : Window size for moving average (default 50).

    Returns:
        dict with keys:
          total_episodes, avg_reward, std_reward,
          avg_steps,  std_steps,
          min_reward, max_reward,
          moving_avg_rewards (list),
          best_moving_avg,   best_moving_avg_episode
    """
    rewards_arr = np.array(rewards, dtype=np.float32)
    steps_arr   = np.array(steps_list, dtype=np.float32)
    mov_avg     = moving_average(rewards, window)

    best_idx = int(np.argmax(mov_avg))

    return {
        "total_episodes"         : len(rewards),
        "avg_reward"             : round(float(np.mean(rewards_arr)), 4),
        "std_reward"             : round(float(np.std(rewards_arr)),  4),
        "min_reward"             : round(float(np.min(rewards_arr)),  4),
        "max_reward"             : round(float(np.max(rewards_arr)),  4),
        "avg_steps"              : round(float(np.mean(steps_arr)),   4),
        "std_steps"              : round(float(np.std(steps_arr)),    4),
        "moving_avg_rewards"     : [round(v, 4) for v in mov_avg],
        "best_moving_avg"        : round(mov_avg[best_idx], 4),
        "best_moving_avg_episode": best_idx + 1,
    }


def save_metrics_csv(metrics: Dict, csv_path: str) -> None:
    """
    Write a flat metrics summary dict to a single-row CSV file.
    Multi-value fields (like moving_avg_rewards list) are excluded.

    Args:
        metrics  : Dict from compute_metrics().
        csv_path : Output CSV file path.
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    flat = {k: v for k, v in metrics.items()
            if not isinstance(v, list)}
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)


def print_metrics_table(label: str, metrics: Dict) -> None:
    """
    Print a clean, human-readable metrics summary to console.

    Args:
        label   : Experiment name / label string.
        metrics : Dict from compute_metrics().
    """
    print(f"\n  ┌─ {label}")
    print(f"  │  Episodes         : {metrics['total_episodes']}")
    print(f"  │  Avg reward       : {metrics['avg_reward']:.2f}"
          f"  ±  {metrics['std_reward']:.2f}")
    print(f"  │  Min / Max reward : {metrics['min_reward']:.2f}"
          f"  /  {metrics['max_reward']:.2f}")
    print(f"  │  Avg steps/ep     : {metrics['avg_steps']:.1f}")
    print(f"  │  Best mov-avg({50}): {metrics['best_moving_avg']:.2f}"
          f"  at ep {metrics['best_moving_avg_episode']}")
    print(f"  └{'─'*50}")
