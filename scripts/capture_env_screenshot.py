"""
scripts/capture_env_screenshot.py
-----------------------------------
Captures a screenshot of the LunarLander-v3 environment at reset
and saves it to docs/figures/lunarlander_screenshot.png for use
in reports and the README.

Usage:
    python scripts/capture_env_screenshot.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR  = os.path.join(ROOT, "docs", "figures")
OUT_PATH = os.path.join(OUT_DIR, "lunarlander_screenshot.png")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Gymnasium version : {gym.__version__}")

env = gym.make("LunarLander-v3", render_mode="rgb_array")
state, info = env.reset(seed=42)
frame = env.render()

fig, ax = plt.subplots(figsize=(8, 5))
ax.imshow(frame)
ax.axis("off")
ax.set_title(
    "LunarLander-v3  —  Initial State\n"
    "Observation: 8-dim vector  |  Actions: Discrete(4)  |  Gymnasium 1.1.1",
    fontsize=10, pad=10, color="#222222"
)

obs_labels = [
    f"x={state[0]:.3f}", f"y={state[1]:.3f}",
    f"vx={state[2]:.3f}", f"vy={state[3]:.3f}",
    f"θ={state[4]:.3f}", f"ω={state[5]:.3f}",
    f"L-leg={state[6]:.0f}", f"R-leg={state[7]:.0f}",
]
fig.text(
    0.5, 0.02, "  |  ".join(obs_labels),
    ha="center", va="bottom", fontsize=7.5, color="#444444",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5",
              edgecolor="#cccccc", alpha=0.85)
)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
plt.close(fig)
env.close()

print(f"Screenshot saved  → {OUT_PATH}")
print(f"Frame shape       : {np.array(frame).shape}")
print(f"Initial state     : {np.round(state, 4)}")
