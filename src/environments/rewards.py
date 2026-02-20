"""Reward functions for the Flappy Bird RL environment.

All reward functions inherit from RewardFunction ABC and implement the
compute() method, which transforms the raw environment reward into a
custom training signal.

Observation indices (use_lidar=False, 12 features):
    obs[0]: last pipe horizontal position
    obs[1]: last pipe top y
    obs[2]: last pipe bottom y
    obs[3]: next pipe horizontal position
    obs[4]: next pipe top y
    obs[5]: next pipe bottom y
    obs[6]: next-next pipe horizontal position
    obs[7]: next-next pipe top y
    obs[8]: next-next pipe bottom y
    obs[9]: player y position
    obs[10]: player velocity
    obs[11]: player rotation
"""

from abc import ABC, abstractmethod

import numpy as np


class RewardFunction(ABC):
    """Abstract base class for reward functions."""

    @abstractmethod
    def compute(
        self,
        obs: np.ndarray,
        raw_reward: float,
        terminated: bool,
        truncated: bool,
    ) -> float:
        """Compute the shaped reward.

        Args:
            obs: The current observation (12 features with use_lidar=False).
            raw_reward: The original reward from the environment.
            terminated: Whether the episode ended (death).
            truncated: Whether the episode was truncated (e.g. time limit).

        Returns:
            The shaped reward value.
        """
        pass


class BasicReward(RewardFunction):
    """Simple reward: +1.0 per step alive, -1000.0 on death.

    This provides a clear survival incentive and a large penalty for dying,
    which helps agents learn to avoid obstacles.
    """

    def compute(
        self,
        obs: np.ndarray,
        raw_reward: float,
        terminated: bool,
        truncated: bool,
    ) -> float:
        if terminated:
            return -1000.0
        return 1.0


class DistanceReward(RewardFunction):
    """Reward proportional to proximity to the next pipe.

    Supports both observation formats:
      - Custom engine (4 features): obs[2] = dist_next (normalized)
      - Gymnasium (12 features): obs[3] = next pipe horizontal position
    Returns -1000.0 on death.
    """

    def compute(
        self,
        obs: np.ndarray,
        raw_reward: float,
        terminated: bool,
        truncated: bool,
    ) -> float:
        if terminated:
            return -1000.0
        if len(obs) <= 4:
            # Custom engine: [player_y, vel, dist_next, gap_center]
            return 1.0 - float(obs[2])
        # Gymnasium 12-feature obs
        next_pipe_x = float(obs[3])
        return 1.0 - next_pipe_x


class CenteredReward(RewardFunction):
    """Bonus for staying centered in the pipe gap.

    Supports both observation formats:
      - Custom engine (4 features): obs[0] = player_y, obs[3] = gap_center
      - Gymnasium (12 features): obs[4]/obs[5] = pipe gap, obs[9] = player_y
    Returns -1000.0 on death.
    """

    def compute(
        self,
        obs: np.ndarray,
        raw_reward: float,
        terminated: bool,
        truncated: bool,
    ) -> float:
        if terminated:
            return -1000.0

        if len(obs) <= 4:
            # Custom engine: [player_y, vel, dist_next, gap_center]
            player_y = float(obs[0])
            gap_center = float(obs[3])
        else:
            # Gymnasium 12-feature obs
            gap_center = (float(obs[4]) + float(obs[5])) / 2.0
            player_y = float(obs[9])

        distance = abs(player_y - gap_center)

        # Bonus decays with distance; max bonus = 2.0 when distance = 0.
        # Using exponential decay so the bonus is always in [0, 2].
        bonus = 2.0 * np.exp(-5.0 * distance)

        return 1.0 + bonus
