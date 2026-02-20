"""Tests for reward functions."""

import numpy as np
import pytest

from src.environments.rewards import (
    RewardFunction,
    BasicReward,
    DistanceReward,
    CenteredReward,
)


# --- Helpers ---

def _make_obs(**overrides) -> np.ndarray:
    """Create a 12-feature observation array with sensible defaults.

    Default values:
        obs[0]: last_pipe_x = 0.5
        obs[1]: last_pipe_top_y = 0.6
        obs[2]: last_pipe_bottom_y = 0.4
        obs[3]: next_pipe_x = 0.8
        obs[4]: next_pipe_top_y = 0.6
        obs[5]: next_pipe_bottom_y = 0.4
        obs[6]: next_next_pipe_x = 1.2
        obs[7]: next_next_pipe_top_y = 0.7
        obs[8]: next_next_pipe_bottom_y = 0.3
        obs[9]: player_y = 0.5
        obs[10]: player_velocity = 0.0
        obs[11]: player_rotation = 0.0
    """
    defaults = {
        "last_pipe_x": 0.5,
        "last_pipe_top_y": 0.6,
        "last_pipe_bottom_y": 0.4,
        "next_pipe_x": 0.8,
        "next_pipe_top_y": 0.6,
        "next_pipe_bottom_y": 0.4,
        "next_next_pipe_x": 1.2,
        "next_next_pipe_top_y": 0.7,
        "next_next_pipe_bottom_y": 0.3,
        "player_y": 0.5,
        "player_velocity": 0.0,
        "player_rotation": 0.0,
    }
    defaults.update(overrides)
    return np.array([
        defaults["last_pipe_x"],
        defaults["last_pipe_top_y"],
        defaults["last_pipe_bottom_y"],
        defaults["next_pipe_x"],
        defaults["next_pipe_top_y"],
        defaults["next_pipe_bottom_y"],
        defaults["next_next_pipe_x"],
        defaults["next_next_pipe_top_y"],
        defaults["next_next_pipe_bottom_y"],
        defaults["player_y"],
        defaults["player_velocity"],
        defaults["player_rotation"],
    ], dtype=np.float32)


# ===================================================================
# RewardFunction ABC
# ===================================================================

class TestRewardFunctionABC:
    """Test that RewardFunction is a proper abstract base class."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            RewardFunction()

    def test_subclass_must_implement_compute(self):
        class IncompleteReward(RewardFunction):
            pass

        with pytest.raises(TypeError):
            IncompleteReward()


# ===================================================================
# BasicReward
# ===================================================================

class TestBasicReward:
    """Tests for BasicReward: +1.0 alive, -1000.0 on death."""

    def setup_method(self):
        self.reward_fn = BasicReward()

    def test_alive_reward(self):
        obs = _make_obs()
        result = self.reward_fn.compute(obs, raw_reward=0.1, terminated=False, truncated=False)
        assert result == pytest.approx(1.0)

    def test_alive_reward_ignores_raw_reward(self):
        obs = _make_obs()
        result = self.reward_fn.compute(obs, raw_reward=5.0, terminated=False, truncated=False)
        assert result == pytest.approx(1.0)

    def test_death_penalty(self):
        obs = _make_obs()
        result = self.reward_fn.compute(obs, raw_reward=-1.0, terminated=True, truncated=False)
        assert result == pytest.approx(-1000.0)

    def test_truncated_not_death(self):
        obs = _make_obs()
        result = self.reward_fn.compute(obs, raw_reward=0.0, terminated=False, truncated=True)
        assert result == pytest.approx(1.0)

    def test_is_reward_function_subclass(self):
        assert isinstance(self.reward_fn, RewardFunction)


# ===================================================================
# DistanceReward
# ===================================================================

class TestDistanceReward:
    """Tests for DistanceReward: reward proportional to proximity to next pipe."""

    def setup_method(self):
        self.reward_fn = DistanceReward()

    def test_death_penalty(self):
        obs = _make_obs()
        result = self.reward_fn.compute(obs, raw_reward=-1.0, terminated=True, truncated=False)
        assert result == pytest.approx(-1000.0)

    def test_closer_pipe_gives_higher_reward(self):
        obs_far = _make_obs(next_pipe_x=1.0)
        obs_close = _make_obs(next_pipe_x=0.2)
        r_far = self.reward_fn.compute(obs_far, raw_reward=0.1, terminated=False, truncated=False)
        r_close = self.reward_fn.compute(obs_close, raw_reward=0.1, terminated=False, truncated=False)
        assert r_close > r_far

    def test_alive_reward_is_positive(self):
        obs = _make_obs(next_pipe_x=0.5)
        result = self.reward_fn.compute(obs, raw_reward=0.1, terminated=False, truncated=False)
        assert result > 0.0

    def test_is_reward_function_subclass(self):
        assert isinstance(self.reward_fn, RewardFunction)


# ===================================================================
# CenteredReward
# ===================================================================

class TestCenteredReward:
    """Tests for CenteredReward: bonus for staying centered in pipe gap."""

    def setup_method(self):
        self.reward_fn = CenteredReward()

    def test_death_penalty(self):
        obs = _make_obs()
        result = self.reward_fn.compute(obs, raw_reward=-1.0, terminated=True, truncated=False)
        assert result == pytest.approx(-1000.0)

    def test_perfectly_centered_gives_max_bonus(self):
        # gap center = (0.6 + 0.4) / 2 = 0.5, player_y = 0.5 => perfect
        obs = _make_obs(
            next_pipe_top_y=0.6,
            next_pipe_bottom_y=0.4,
            player_y=0.5,
        )
        result = self.reward_fn.compute(obs, raw_reward=0.1, terminated=False, truncated=False)
        # Should be base (1.0) + max bonus (up to 2.0) = 3.0
        assert result == pytest.approx(3.0)

    def test_off_center_gives_lower_reward(self):
        # Perfectly centered
        obs_center = _make_obs(
            next_pipe_top_y=0.6,
            next_pipe_bottom_y=0.4,
            player_y=0.5,
        )
        # Off-center
        obs_off = _make_obs(
            next_pipe_top_y=0.6,
            next_pipe_bottom_y=0.4,
            player_y=0.8,
        )
        r_center = self.reward_fn.compute(obs_center, raw_reward=0.1, terminated=False, truncated=False)
        r_off = self.reward_fn.compute(obs_off, raw_reward=0.1, terminated=False, truncated=False)
        assert r_center > r_off

    def test_alive_reward_at_least_base(self):
        obs = _make_obs(player_y=0.9)  # far from center
        result = self.reward_fn.compute(obs, raw_reward=0.1, terminated=False, truncated=False)
        assert result >= 1.0

    def test_alive_reward_at_most_three(self):
        obs = _make_obs(player_y=0.5)
        result = self.reward_fn.compute(obs, raw_reward=0.1, terminated=False, truncated=False)
        assert result <= 3.0

    def test_is_reward_function_subclass(self):
        assert isinstance(self.reward_fn, RewardFunction)
