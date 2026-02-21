"""Reward functions for the Flappy Bird RL environment.

All reward functions inherit from RewardFunction ABC and implement the
compute() method, which transforms the raw environment reward into a
custom training signal.

Custom engine observation (8D relative):
    obs[0]: delta_y1           - player_y - gap_center (positive = below gap)
    obs[1]: velocity           - player vertical velocity (normalized)
    obs[2]: dist_pipe1         - horizontal distance to nearest pipe
    obs[3]: delta_y2           - offset to second pipe gap center
    obs[4]: dist_pipe2         - horizontal distance to second pipe
    obs[5]: gap_position       - -1 above gap, 0 inside gap, +1 below gap
    obs[6]: proximity_danger   - urgency signal when close to pipe
    obs[7]: vertical_speed_dir - sign(vel) * vel^2, amplified velocity

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
        if len(obs) < 12:
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

        if len(obs) < 12:
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
    """Combined reward: centering + progress + survival + velocity control.

    Reward signal (4 components, scaled so pipe bonus dominates):
      - Centering: clipped linear bonus, max 0.3 when centered, 0.0 when
        >= 0.15 away from gap center (no reward leakage outside gap).
      - Progress: small bonus for being close to next pipe (0 to 0.1).
      - Survival: small constant bonus (+0.1)
      - Velocity penalty: penalizes extreme vertical velocities when near
        pipes (within 30% screen width). Encourages smooth flight through
        gaps rather than wild oscillation. Max penalty -0.15.
      - Death: configurable penalty (default -5.0, overridden by BirdEntry)

    Supports both 5D-relative (custom engine) and 12-feature (gymnasium) obs.
    """

    def __init__(self, death_penalty: float = 5.0, velocity_coef: float = 0.15):
        self.death_penalty = death_penalty
        self.velocity_coef = velocity_coef

    def compute(
        self,
        obs: np.ndarray,
        raw_reward: float,
        terminated: bool,
        truncated: bool,
    ) -> float:
        if terminated:
            return -self.death_penalty

        if len(obs) < 12:
            # Custom engine: obs[0]=delta_y1, obs[1]=velocity, obs[2]=dist_pipe1
            distance = abs(float(obs[0]))
            velocity = float(obs[1])
            dist_next = float(obs[2])
        else:
            # Gymnasium 12-feature obs
            gap_center = (float(obs[4]) + float(obs[5])) / 2.0
            player_y = float(obs[9])
            distance = abs(player_y - gap_center)
            velocity = float(obs[10])
            dist_next = float(obs[3])

        # Clipped linear centering: max 0.3 when centered, 0.0 when >= 0.15 away
        centering = max(0.0, 1.0 - distance / 0.15) * 0.3

        # Small progress bonus
        progress = (1.0 - max(0.0, dist_next)) * 0.1

        # Velocity penalty: penalize extreme speeds when near a pipe.
        # Only active within 30% screen distance to avoid penalizing
        # necessary corrections far from pipes.
        vel_penalty = 0.0
        if dist_next < 0.3:
            vel_penalty = -self.velocity_coef * min(1.0, velocity ** 2)

        return 0.1 + centering + progress + vel_penalty


class CurriculumReward(RewardFunction):
    """Adaptive reward that starts generous and becomes stricter over time.

    Designed for curriculum learning: early in training the agent gets
    large survival bonuses and forgiving centering, making it easy to
    learn basic flight. As episodes progress, the reward tightens:
    stricter centering, velocity control, and smaller survival bonus.

    Phases (controlled by episode_count):
      - Early (< warmup):  survival=0.5, centering_radius=0.25, no vel penalty
      - Mid (warmup..2*warmup): linear interpolation between early and late
      - Late (>= 2*warmup): survival=0.05, centering_radius=0.10, vel penalty

    Supports both 5D-relative (custom engine) and 12-feature (gymnasium) obs.
    """

    def __init__(self, death_penalty: float = 5.0, warmup: int = 50):
        self.death_penalty = death_penalty
        self.warmup = max(1, warmup)
        self.episode_count = 0

    def advance_episode(self):
        """Call at the start of each new episode to update the curriculum."""
        self.episode_count += 1

    @property
    def progress(self) -> float:
        """Curriculum progress in [0, 1]. 0=early, 1=late."""
        return min(1.0, self.episode_count / (2.0 * self.warmup))

    def compute(
        self,
        obs: np.ndarray,
        raw_reward: float,
        terminated: bool,
        truncated: bool,
    ) -> float:
        if terminated:
            return -self.death_penalty

        p = self.progress  # 0 = early, 1 = late

        if len(obs) < 12:
            distance = abs(float(obs[0]))
            velocity = float(obs[1])
            dist_next = float(obs[2])
        else:
            gap_center = (float(obs[4]) + float(obs[5])) / 2.0
            player_y = float(obs[9])
            distance = abs(player_y - gap_center)
            velocity = float(obs[10])
            dist_next = float(obs[3])

        # Interpolate survival bonus: 0.5 (early) -> 0.05 (late)
        survival = 0.5 - 0.45 * p

        # Interpolate centering radius: 0.25 (forgiving) -> 0.10 (strict)
        radius = 0.25 - 0.15 * p
        centering = max(0.0, 1.0 - distance / radius) * 0.3

        # Progress bonus (constant across curriculum)
        progress_bonus = (1.0 - max(0.0, dist_next)) * 0.1

        # Velocity penalty: ramps in during late phase
        vel_penalty = 0.0
        if p > 0.5 and dist_next < 0.3:
            vel_scale = (p - 0.5) * 2.0  # 0..1 during second half
            vel_penalty = -0.15 * vel_scale * min(1.0, velocity ** 2)

        return survival + centering + progress_bonus + vel_penalty
