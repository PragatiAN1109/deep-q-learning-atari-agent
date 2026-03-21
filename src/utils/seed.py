"""
src/utils/seed.py
------------------
Global random seed utility for full reproducibility.

Why reproducibility matters:
    Without fixed seeds, every training run produces different results,
    making it impossible to compare experiments fairly or debug issues.
    Setting seeds across Python, NumPy, and PyTorch ensures identical
    behaviour across runs with the same config.

Usage:
    from src.utils.seed import set_global_seed
    set_global_seed(42)
"""

import os
import random
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch (CPU + CUDA).

    Args:
        seed (int): The seed value. Use the value from config.yaml
                    (environment.seed) so it's tracked in git.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic CUDA operations (small perf cost, worth it)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    # Python hash seed for dict ordering reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[Seed] Global random seed set to {seed}")
