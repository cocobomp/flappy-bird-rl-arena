"""Reward functions for the Flappy Bird RL environment.

All reward functions inherit from RewardFunction ABC and implement the
compute() method, which transforms the raw environment reward into a
custom training signal.

Custom engine observation (8 features):
    obs[0]: player_y          - player y position (normalized)
    obs[1]: velocity           - player vertical velocity (normalized)
    obs[2]: dist_pipe1         - horizontal distance to nearest pipe
    obs[3]: top1               - upper edge of gap (nearest pipe)
    obs[4]: bottom1            - lower edge of gap (nearest pipe)
    obs[5]: dist_pipe2         - horizontal distance to second pipe
    obs[6]: top2               - upper edge of gap (second pipe)
    obs[7]: bottom2            - lower edge of gap (second pipe)

Gymnasium observation (use_lidar=False, 12 features):
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
      - Custom engine (8 features): obs[2] = dist_pipe1 (normalized)
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
        if len(obs) != 12:
            # Custom engine: [player_y, vel, dist_pipe1, top1, bottom1, ...]
            return 1.0 - float(obs[2])
        # Gymnasium 12-feature obs
        next_pipe_x = float(obs[3])
        return 1.0 - next_pipe_x


class CenteredReward(RewardFunction):
    """Bonus for staying centered in the pipe gap.

    Supports both observation formats:
      - Custom engine (8 features): obs[0] = player_y,
        gap_center = (obs[3] + obs[4]) / 2
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

        if len(obs) != 12:
            # Custom engine: [player_y, vel, dist_pipe1, top1, bottom1, ...]
            player_y = float(obs[0])
            gap_center = (float(obs[3]) + float(obs[4])) / 2.0
        else:
            # Gymnasium 12-feature obs
            gap_center = (float(obs[4]) + float(obs[5])) / 2.0
            player_y = float(obs[9])

        distance = abs(player_y - gap_center)

        # Bonus decays with distance; max bonus = 2.0 when distance = 0.
        # Using exponential decay so the bonus is always in [0, 2].
        bonus = 2.0 * np.exp(-5.0 * distance)

        return 1.0 + bonus


class SmartReward(RewardFunction):
    """Combined reward: centering + velocity direction + progress + survival.

    Provides the richest learning signal by rewarding:
      - Centering: exponential bonus for being near gap center (0 to 2.0)
      - Direction: bonus for moving toward gap (+0.3), penalty for away (-0.1)
      - Progress: bonus for being close to next pipe (0 to 0.3)
      - Survival: small constant bonus (+0.1)
      - Death: configurable penalty (default -5.0)

    Supports both 8-feature (custom engine) and 12-feature (gymnasium) obs.
    """

    def __init__(self, death_penalty: float = 5.0):
        self.death_penalty = death_penalty

    def compute(
        self,
        obs: np.ndarray,
        raw_reward: float,
        terminated: bool,
        truncated: bool,
    ) -> float:
        if terminated:
            return -self.death_penalty

        if len(obs) != 12:
            # Custom engine: [player_y, vel, dist_pipe1, top1, bottom1, ...]
            player_y = float(obs[0])
            velocity = float(obs[1])
            dist_next = float(obs[2])
            gap_center = (float(obs[3]) + float(obs[4])) / 2.0
        else:
            gap_center = (float(obs[4]) + float(obs[5])) / 2.0
            player_y = float(obs[9])
            velocity = float(obs[10]) if len(obs) > 10 else 0.0
            dist_next = float(obs[3])

        # Centering: max 2.0 when perfectly centered in gap
        centering = 2.0 * np.exp(-5.0 * abs(player_y - gap_center))

        # Direction: reward moving toward the gap center
        if player_y > gap_center:
            # Bird below gap → reward going up (negative velocity)
            direction = 0.3 if velocity < 0 else -0.1
        else:
            # Bird above gap → reward going down (positive velocity)
            direction = 0.3 if velocity > 0 else -0.1

        # Progress: closer to pipe = more reward
        progress = (1.0 - max(0.0, dist_next)) * 0.3

        return 0.1 + centering + direction + progress
