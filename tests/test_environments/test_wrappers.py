"""Tests for environment wrappers."""

import gymnasium
import numpy as np
import pytest

from src.environments.rewards import BasicReward, CenteredReward
from src.environments.wrappers import (
    SimpleObsWrapper,
    EnrichedObsWrapper,
    CustomRewardWrapper,
)


# --- Fake environment for deterministic testing ---

class FakeFlappyEnv(gymnasium.Env):
    """Minimal fake of FlappyBird-v0 (use_lidar=False) for testing wrappers.

    Produces a fixed 12-feature observation and allows controlling
    terminated/truncated signals.
    """

    metadata = {"render_modes": []}

    def __init__(self, obs: np.ndarray | None = None):
        super().__init__()
        self.action_space = gymnasium.spaces.Discrete(2)
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32,
        )
        self._obs = obs if obs is not None else self._default_obs()
        self._terminated = False
        self._truncated = False
        self._raw_reward = 0.1

    @staticmethod
    def _default_obs() -> np.ndarray:
        return np.array([
            0.5,   # obs[0]: last_pipe_x
            0.6,   # obs[1]: last_pipe_top_y
            0.4,   # obs[2]: last_pipe_bottom_y
            0.8,   # obs[3]: next_pipe_x
            0.6,   # obs[4]: next_pipe_top_y
            0.4,   # obs[5]: next_pipe_bottom_y
            1.2,   # obs[6]: next_next_pipe_x
            0.7,   # obs[7]: next_next_pipe_top_y
            0.3,   # obs[8]: next_next_pipe_bottom_y
            0.5,   # obs[9]: player_y
            0.0,   # obs[10]: player_velocity
            0.0,   # obs[11]: player_rotation
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._obs.copy(), {}

    def step(self, action):
        return (
            self._obs.copy(),
            self._raw_reward,
            self._terminated,
            self._truncated,
            {"score": 0},
        )

    def set_terminated(self, val: bool):
        self._terminated = val

    def set_raw_reward(self, val: float):
        self._raw_reward = val


# ===================================================================
# SimpleObsWrapper
# ===================================================================

class TestSimpleObsWrapper:
    """Tests for SimpleObsWrapper: extracts 4 features."""

    def setup_method(self):
        self.base_env = FakeFlappyEnv()
        self.env = SimpleObsWrapper(self.base_env)

    def test_observation_shape(self):
        obs, _ = self.env.reset()
        assert obs.shape == (4,)

    def test_observation_dtype(self):
        obs, _ = self.env.reset()
        assert obs.dtype == np.float32

    def test_observation_values(self):
        obs, _ = self.env.reset()
        expected_player_y = 0.5        # obs[9]
        expected_velocity = 0.0         # obs[10]
        expected_dist_next = 0.8        # obs[3]
        expected_gap_center = 0.5       # (obs[4]+obs[5])/2 = (0.6+0.4)/2
        np.testing.assert_array_almost_equal(
            obs,
            [expected_player_y, expected_velocity, expected_dist_next, expected_gap_center],
        )

    def test_step_observation(self):
        self.env.reset()
        obs, _, _, _, _ = self.env.step(0)
        assert obs.shape == (4,)

    def test_observation_space_updated(self):
        assert self.env.observation_space.shape == (4,)
        assert self.env.observation_space.dtype == np.float32

    def test_custom_obs_values(self):
        custom_obs = np.array([
            0.1, 0.2, 0.3, 0.4, 0.8, 0.2, 0.9, 0.7, 0.3, 0.75, -0.5, 0.1,
        ], dtype=np.float32)
        env = SimpleObsWrapper(FakeFlappyEnv(obs=custom_obs))
        obs, _ = env.reset()
        expected = np.array([0.75, -0.5, 0.4, 0.5], dtype=np.float32)
        np.testing.assert_array_almost_equal(obs, expected)


# ===================================================================
# EnrichedObsWrapper
# ===================================================================

class TestEnrichedObsWrapper:
    """Tests for EnrichedObsWrapper: extracts 7 features."""

    def setup_method(self):
        self.base_env = FakeFlappyEnv()
        self.env = EnrichedObsWrapper(self.base_env)

    def test_observation_shape(self):
        obs, _ = self.env.reset()
        assert obs.shape == (7,)

    def test_observation_dtype(self):
        obs, _ = self.env.reset()
        assert obs.dtype == np.float32

    def test_observation_values(self):
        obs, _ = self.env.reset()
        player_y = 0.5
        velocity = 0.0
        dist_next = 0.8
        gap_center = 0.5       # (0.6+0.4)/2
        dist_2nd = 1.2
        gap_2nd_center = 0.5   # (0.7+0.3)/2
        delta_y = player_y - gap_center  # 0.0

        np.testing.assert_array_almost_equal(
            obs,
            [player_y, velocity, dist_next, gap_center, dist_2nd, gap_2nd_center, delta_y],
        )

    def test_step_observation(self):
        self.env.reset()
        obs, _, _, _, _ = self.env.step(1)
        assert obs.shape == (7,)

    def test_observation_space_updated(self):
        assert self.env.observation_space.shape == (7,)
        assert self.env.observation_space.dtype == np.float32

    def test_delta_y_computed_correctly(self):
        custom_obs = np.array([
            0.1, 0.2, 0.3, 0.4, 0.8, 0.2, 0.9, 0.7, 0.3, 0.75, -0.5, 0.1,
        ], dtype=np.float32)
        env = EnrichedObsWrapper(FakeFlappyEnv(obs=custom_obs))
        obs, _ = env.reset()
        gap_center = (0.8 + 0.2) / 2  # 0.5
        expected_delta = 0.75 - gap_center  # 0.25
        assert obs[6] == pytest.approx(expected_delta)


# ===================================================================
# CustomRewardWrapper
# ===================================================================

class TestCustomRewardWrapper:
    """Tests for CustomRewardWrapper: replaces env reward with custom reward."""

    def test_basic_reward_alive(self):
        base_env = FakeFlappyEnv()
        env = CustomRewardWrapper(base_env, reward_fn=BasicReward())
        env.reset()
        _, reward, _, _, _ = env.step(0)
        assert reward == pytest.approx(1.0)

    def test_basic_reward_death(self):
        base_env = FakeFlappyEnv()
        base_env.set_terminated(True)
        env = CustomRewardWrapper(base_env, reward_fn=BasicReward())
        env.reset()
        _, reward, terminated, _, _ = env.step(0)
        assert reward == pytest.approx(-1000.0)
        assert terminated is True

    def test_centered_reward_integration(self):
        base_env = FakeFlappyEnv()
        env = CustomRewardWrapper(base_env, reward_fn=CenteredReward())
        env.reset()
        _, reward, _, _, _ = env.step(0)
        # Player at 0.5, gap center = 0.5 => perfectly centered => 3.0
        assert reward == pytest.approx(3.0)

    def test_info_passed_through(self):
        base_env = FakeFlappyEnv()
        env = CustomRewardWrapper(base_env, reward_fn=BasicReward())
        env.reset()
        _, _, _, _, info = env.step(0)
        assert "score" in info

    def test_observation_unchanged(self):
        base_env = FakeFlappyEnv()
        env = CustomRewardWrapper(base_env, reward_fn=BasicReward())
        obs_reset, _ = env.reset()
        obs_step, _, _, _, _ = env.step(0)
        # Observations should still be the original 12-feature shape
        assert obs_step.shape == (12,)

    def test_raw_reward_forwarded_to_reward_fn(self):
        """Verify the wrapper passes the original raw reward to the reward function."""
        base_env = FakeFlappyEnv()
        base_env.set_raw_reward(42.0)

        class RawEchoReward:
            """Echoes back the raw reward for testing."""
            def compute(self, obs, raw_reward, terminated, truncated):
                return raw_reward

        # Technically RawEchoReward doesn't subclass RewardFunction,
        # but CustomRewardWrapper should still work with duck typing.
        env = CustomRewardWrapper(base_env, reward_fn=RawEchoReward())
        env.reset()
        _, reward, _, _, _ = env.step(0)
        assert reward == pytest.approx(42.0)

    def test_works_with_obs_wrapper_stacked(self):
        """Verify CustomRewardWrapper stacks correctly with an obs wrapper."""
        base_env = FakeFlappyEnv()
        env = SimpleObsWrapper(base_env)
        env = CustomRewardWrapper(env, reward_fn=BasicReward())
        obs, _ = env.reset()
        assert obs.shape == (4,)
        obs, reward, _, _, _ = env.step(0)
        assert obs.shape == (4,)
        assert reward == pytest.approx(1.0)
