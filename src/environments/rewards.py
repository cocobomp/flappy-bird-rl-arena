"""Reward functions for the Flappy Bird RL environment.

All reward functions inherit from RewardFunction ABC and implement the
compute() method, which transforms the raw environment reward into a
custom training signal.

Custom engine observation (5D relative):
    obs[0]: delta_y1           - player_y - gap_center (positive = below gap)
    obs[1]: velocity           - player vertical velocity (normalized)
    obs[2]: dist_pipe1         - horizontal distance to nearest pipe
    obs[3]: delta_y2           - offset to second pipe gap center
    obs[4]: dist_pipe2         - horizontal distance to second pipe

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
      - Custom engine (5D relative): obs[2] = dist_pipe1 (normalized)
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
            # Custom engine: [delta_y1, vel, dist_pipe1, delta_y2, dist_pipe2]
            return 1.0 - float(obs[2])
        # Gymnasium 12-feature obs
        next_pipe_x = float(obs[3])
        return 1.0 - next_pipe_x


class CenteredReward(RewardFunction):
    """Bonus for staying centered in the pipe gap.

    Supports both observation formats:
      - Custom engine (5D relative): obs[0] = delta_y1 (player_y - gap_center)
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
            # Custom engine: [delta_y1, vel, dist_pipe1, delta_y2, dist_pipe2]
            distance = abs(float(obs[0]))
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
    """Combined reward: scaled centering + progress + survival.

    Reward signal (3 components, scaled so pipe bonus dominates):
      - Centering: clipped linear bonus, max 0.3 when centered, 0.0 when
        >= 0.15 away from gap center (no reward leakage outside gap).
        Coefficient reduced from 1.0 to 0.3 so cumulative shaping between
        pipes does not drown out the discrete pipe bonus (old ratio was
        3:1 shaping:pipe; new ratio is ~0.8:1).
      - Progress: small bonus for being close to next pipe (0 to 0.1).
        Reduced from 0.2 to 0.1 for the same scaling reason.
      - Survival: small constant bonus (+0.1)
      - Death: configurable penalty (default -5.0, overridden by BirdEntry)

    Direction reward was removed (caused oscillation near gap center).
    Centering uses clipped linear for tighter signal.

    Supports both 5D-relative (custom engine) and 12-feature (gymnasium) obs.
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
            # Custom engine: obs[0] = delta_y1 (player_y - gap_center)
            distance = abs(float(obs[0]))
            dist_next = float(obs[2])
        else:
            # Gymnasium 12-feature obs
            gap_center = (float(obs[4]) + float(obs[5])) / 2.0
            player_y = float(obs[9])
            distance = abs(player_y - gap_center)
            dist_next = float(obs[3])

        # Clipped linear centering: max 0.3 when centered, 0.0 when >= 0.15 away
        # (scaled down from 1.0 so pipe bonus remains the dominant signal)
        centering = max(0.0, 1.0 - distance / 0.15) * 0.3

        # Small progress bonus (scaled down from 0.2)
        progress = (1.0 - max(0.0, dist_next)) * 0.1

        return 0.1 + centering + progress
