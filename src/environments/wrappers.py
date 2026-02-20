"""Gymnasium wrappers for the Flappy Bird RL environment.

These wrappers transform observations and rewards from the base
FlappyBird-v0 environment (with use_lidar=False, 12-feature obs).

Observation indices:
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

from __future__ import annotations

import gymnasium
import numpy as np

from src.environments.rewards import RewardFunction


class SimpleObsWrapper(gymnasium.ObservationWrapper):
    """Extract 4 features from the 12-feature FlappyBird observation.

    Output features (shape (4,), float32):
        [0] player_y      = obs[9]
        [1] velocity       = obs[10]
        [2] dist_next_pipe = obs[3]
        [3] gap_center     = (obs[4] + obs[5]) / 2
    """

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        player_y = observation[9]
        velocity = observation[10]
        dist_next_pipe = observation[3]
        gap_center = (observation[4] + observation[5]) / 2.0
        return np.array(
            [player_y, velocity, dist_next_pipe, gap_center],
            dtype=np.float32,
        )


class EnrichedObsWrapper(gymnasium.ObservationWrapper):
    """Extract 7 features from the 12-feature FlappyBird observation.

    Output features (shape (7,), float32):
        [0] player_y        = obs[9]
        [1] velocity         = obs[10]
        [2] dist_next        = obs[3]
        [3] gap_center       = (obs[4] + obs[5]) / 2
        [4] dist_2nd         = obs[6]
        [5] gap_2nd_center   = (obs[7] + obs[8]) / 2
        [6] delta_y          = player_y - gap_center
    """

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        player_y = observation[9]
        velocity = observation[10]
        dist_next = observation[3]
        gap_center = (observation[4] + observation[5]) / 2.0
        dist_2nd = observation[6]
        gap_2nd_center = (observation[7] + observation[8]) / 2.0
        delta_y = player_y - gap_center
        return np.array(
            [player_y, velocity, dist_next, gap_center, dist_2nd, gap_2nd_center, delta_y],
            dtype=np.float32,
        )


class CustomRewardWrapper(gymnasium.Wrapper):
    """Replace the environment reward with a custom reward function.

    Overrides step() directly (rather than inheriting from RewardWrapper
    and using reward()) because the RewardFunction.compute() method needs
    access to the full observation, terminated, and truncated signals --
    not just the scalar reward.
    """

    def __init__(self, env: gymnasium.Env, reward_fn: RewardFunction):
        super().__init__(env)
        self._reward_fn = reward_fn

    def step(self, action):
        obs, raw_reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = self._reward_fn.compute(obs, raw_reward, terminated, truncated)
        return obs, shaped_reward, terminated, truncated, info
