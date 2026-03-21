"""
src/utils/logger.py
--------------------
Lightweight CSV + console logger for training metrics.

Logs per-episode: episode, total_reward, avg_reward_100,
epsilon, steps, loss.

The CSV file can be imported into matplotlib (see utils/plot.py)
to produce reward curves for the report.

Usage:
    logger = TrainingLogger("experiments/training_log.csv")
    logger.log(episode=1, reward=120.0, avg_reward=50.0,
               epsilon=0.99, steps=300, loss=0.042)
    logger.close()
"""

import csv
import os
from typing import Optional


FIELDNAMES = ["episode", "total_reward", "avg_reward_100",
              "epsilon", "steps", "loss"]


class TrainingLogger:
    """
    Writes one CSV row per episode and optionally prints to console.

    Args:
        filepath (str) : Path to the output CSV file.
        log_freq (int) : Print to console every N episodes. 0 = silent.
    """

    def __init__(self, filepath: str, log_freq: int = 10):
        self.filepath  = filepath
        self.log_freq  = log_freq
        self._episode  = 0

        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        self._file   = open(filepath, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDNAMES)
        self._writer.writeheader()
        self._file.flush()
        print(f"[Logger] Writing training log → {filepath}")

    def log(
        self,
        episode:      int,
        reward:       float,
        avg_reward:   float,
        epsilon:      float,
        steps:        int,
        loss:         Optional[float] = None,
    ) -> None:
        """
        Write one row to the CSV and optionally print to console.

        Args:
            episode   : Current episode number (1-indexed).
            reward    : Total reward for this episode.
            avg_reward: Rolling average over the last 100 episodes.
            epsilon   : Current exploration rate.
            steps     : Steps taken in this episode.
            loss      : Mean training loss this episode (None if no updates yet).
        """
        row = {
            "episode":        episode,
            "total_reward":   round(reward, 3),
            "avg_reward_100": round(avg_reward, 3),
            "epsilon":        round(epsilon, 5),
            "steps":          steps,
            "loss":           round(loss, 6) if loss is not None else "",
        }
        self._writer.writerow(row)
        self._file.flush()   # ensure data is written even if training crashes

        if self.log_freq > 0 and episode % self.log_freq == 0:
            loss_str = f"{loss:.5f}" if loss is not None else "N/A"
            print(
                f"  Ep {episode:>5} | "
                f"Reward: {reward:>8.2f} | "
                f"Avg100: {avg_reward:>8.2f} | "
                f"ε: {epsilon:.4f} | "
                f"Steps: {steps:>5} | "
                f"Loss: {loss_str}"
            )

    def close(self) -> None:
        """Flush and close the CSV file handle."""
        self._file.flush()
        self._file.close()
        print(f"[Logger] Log closed → {self.filepath}")
