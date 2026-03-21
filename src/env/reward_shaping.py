"""
src/env/reward_shaping.py
--------------------------
Optional reward shaping wrapper for LunarLander-v2.

Strategy 1 — Leg Contact Bonus:
    Adds a small positive bonus proportional to leg-ground contact.
    Encourages the agent to keep legs on the ground, increasing the
    chance of a stable landing.

    Bonus per step = leg_bonus_weight * (obs[6] + obs[7])
    where obs[6] = left leg contact flag (0.0 or 1.0)
          obs[7] = right leg contact flag (0.0 or 1.0)

    Default weight = 2.0 → max bonus = +4.0/step when both legs touch.
    This is intentionally small vs the native +10 per leg contact and
    the +100 landing bonus, so it guides without dominating.

This wrapper is a drop-in — all existing training scripts work unchanged.
It only modifies the reward signal; observations, actions, and termination
conditions are identical to the base environment.

Usage (optional — does NOT break existing code):
    import gymnasium as gym
    from src.env.reward_shaping import ShapedLunarLander

    base_env = gym.make("LunarLander-v2")
    env = ShapedLunarLander(base_env, leg_bonus_weight=2.0)

    # Then use env exactly like the original:
    obs, info = env.reset()
    obs, reward, term, trunc, info = env.step(action)

Note:
    To compare shaped vs unshaped training, run:
        python src/train.py --out-dir experiments/unshaped_run
    Then modify src/train.py's env creation to use ShapedLunarLander and run:
        python src/train.py --out-dir experiments/shaped_run
"""

import numpy as np
import gymnasium as gym
from gymnasium import Env


class ShapedLunarLander(gym.Wrapper):
    """
    Reward shaping wrapper for LunarLander-v2.

    Adds a leg-contact bonus to the native reward signal to provide
    denser feedback for the foot-placement phase of landing.

    Args:
        env              : Base Gymnasium environment instance.
        leg_bonus_weight : Multiplier for the leg contact bonus.
                           0.0 = no shaping (identical to base env).
                           2.0 = default (max +4.0/step both legs).
    """

    def __init__(self, env: Env, leg_bonus_weight: float = 2.0):
        super().__init__(env)
        self.leg_bonus_weight = leg_bonus_weight

    def step(self, action: int):
        """
        Step the environment and add shaped reward component.

        Shaping formula:
            shaped_reward = raw_reward
                          + leg_bonus_weight * (obs[6] + obs[7])

        obs[6] = left  leg contact (0.0 or 1.0)
        obs[7] = right leg contact (0.0 or 1.0)

        Returns:
            Same (obs, reward, terminated, truncated, info) tuple as base env,
            with reward replaced by shaped_reward.
        """
        obs, raw_reward, terminated, truncated, info = self.env.step(action)

        # ── Leg contact bonus ─────────────────────────────────
        # obs[6] and obs[7] are binary flags: 1.0 if leg touching ground
        left_leg_contact  = float(obs[6])
        right_leg_contact = float(obs[7])
        leg_bonus = self.leg_bonus_weight * (left_leg_contact + right_leg_contact)

        shaped_reward = raw_reward + leg_bonus

        # Store original reward in info dict for diagnostics
        info["raw_reward"]    = raw_reward
        info["leg_bonus"]     = leg_bonus
        info["shaped_reward"] = shaped_reward

        return obs, shaped_reward, terminated, truncated, info
