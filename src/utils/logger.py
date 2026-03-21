"""
src/utils/logger.py
--------------------
Structured training logger combining Python's logging module with CSV metrics.

Design:
    - Python ``logging`` for console output (replaces bare print statements).
      Log level is configurable: INFO by default, DEBUG for verbose output.
    - CSV writer for per-episode numeric metrics, compatible with pandas
      and the plotting utilities in ``src/utils/plot.py``.
    - A single ``TrainingLogger`` instance handles both channels so
      training scripts need only one import.

Columns written to CSV:
    episode, total_reward, avg_reward_100, epsilon, steps, loss

Usage::

    from src.utils.logger import TrainingLogger, get_logger

    # Module-level logger (for non-training modules)
    log = get_logger(__name__)
    log.info("Starting environment: %s", env_name)

    # Training logger (CSV + console together)
    logger = TrainingLogger("experiments/training_log.csv", log_freq=10)
    logger.log(episode=1, reward=120.0, avg_reward=50.0,
               epsilon=0.99, steps=300, loss=0.042)
    logger.close()
"""

import csv
import logging
import os
import sys
from typing import Optional


# ── Module-level log format ───────────────────────────────────
_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_DATE_FORMAT = "%H:%M:%S"

# ── CSV column schema ─────────────────────────────────────────
FIELDNAMES = ["episode", "total_reward", "avg_reward_100",
              "epsilon", "steps", "loss"]


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger with a consistent console handler.

    Call once per module at the top level::

        log = get_logger(__name__)

    If a handler is already attached (e.g., in tests), this is a no-op
    so duplicate messages are not printed.

    Args:
        name:  Logger name — use ``__name__`` for automatic namespacing.
        level: Logging level. Defaults to ``logging.INFO``.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class TrainingLogger:
    """
    Unified training logger: Python ``logging`` to console + CSV to disk.

    Each call to :meth:`log` writes one row to the CSV file and, every
    ``log_freq`` episodes, prints a formatted summary line to the console
    via the Python logging system (not bare ``print``).

    Args:
        filepath: Path to the output CSV file.
                  Parent directories are created automatically.
        log_freq: Console print frequency in episodes. ``0`` disables it.
        level:    Logging level for console output.
    """

    def __init__(
        self,
        filepath: str,
        log_freq: int = 10,
        level:    int = logging.INFO,
    ) -> None:
        self.filepath = filepath
        self.log_freq = log_freq
        self._log     = get_logger("TrainingLogger", level)

        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        self._file   = open(filepath, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDNAMES)
        self._writer.writeheader()
        self._file.flush()
        self._log.info("Training log → %s", filepath)

    def log(
        self,
        episode:    int,
        reward:     float,
        avg_reward: float,
        epsilon:    float,
        steps:      int,
        loss:       Optional[float] = None,
    ) -> None:
        """
        Record one episode's metrics to CSV and optionally to console.

        Args:
            episode:    Current episode number (1-indexed).
            reward:     Total reward accumulated during this episode.
            avg_reward: Rolling average over the last 100 episodes.
            epsilon:    Exploration rate at the end of this episode.
            steps:      Number of environment steps taken.
            loss:       Mean training loss for this episode, or ``None``
                        if the replay buffer was not yet ready.
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
        # Flush after every row so data is not lost on training crashes
        self._file.flush()

        if self.log_freq > 0 and episode % self.log_freq == 0:
            loss_str = f"{loss:.5f}" if loss is not None else "N/A"
            self._log.info(
                "Ep %5d | reward %8.2f | avg100 %8.2f | "
                "ε %.4f | steps %4d | loss %s",
                episode, reward, avg_reward, epsilon, steps, loss_str,
            )

    def close(self) -> None:
        """Flush and close the CSV file handle. Call after training ends."""
        self._file.flush()
        self._file.close()
        self._log.info("Training log closed → %s", self.filepath)
