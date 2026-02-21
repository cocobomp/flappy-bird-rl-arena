"""Tests for exploration strategies."""

import numpy as np
import pytest

from src.game.strategies import (
    RandomStrategy,
    GravityAwareStrategy,
    HeuristicStrategy,
    GuidedStrategy,
    BoltzmannStrategy,
    cosine_epsilon_schedule,
    STRATEGY_MAP,
    STRATEGY_OPTIONS,
)


def _obs(delta_y=0.0, velocity=0.0, dist_pipe1=0.5,
         delta_y2=0.0, dist_pipe2=0.8) -> np.ndarray:
    gap_position = -1.0 if delta_y < -0.098 else (1.0 if delta_y > 0.098 else 0.0)
    proximity_danger = max(0.0, 1.0 - dist_pipe1) if dist_pipe1 < 0.2 else 0.0
    vertical_speed_dir = float(np.sign(velocity)) * (velocity ** 2)
    return np.array([delta_y, velocity, dist_pipe1, delta_y2, dist_pipe2,
                     gap_position, proximity_danger, vertical_speed_dir],
                    dtype=np.float32)


# ===================================================================
# BoltzmannStrategy
# ===================================================================

class TestBoltzmannStrategy:
    """Tests for temperature-based action selection."""

    def test_without_q_values_falls_back_to_heuristic(self):
        strat = BoltzmannStrategy(temperature=1.0)
        # Without set_q_values, should use fallback heuristic
        action = strat.explore(_obs())
        assert action in (0, 1)

    def test_with_q_values_returns_valid_action(self):
        strat = BoltzmannStrategy(temperature=1.0)
        strat.set_q_values(np.array([1.0, 2.0]))
        action = strat.explore(_obs())
        assert action in (0, 1)

    def test_low_temperature_favors_best_action(self):
        strat = BoltzmannStrategy(temperature=0.01)
        # Q-values strongly favor action 1
        counts = {0: 0, 1: 0}
        for _ in range(200):
            strat.set_q_values(np.array([-5.0, 5.0]))
            counts[strat.explore(_obs())] += 1
        # Action 1 should dominate at low temperature
        assert counts[1] > counts[0] * 5

    def test_high_temperature_is_more_uniform(self):
        strat = BoltzmannStrategy(temperature=100.0)
        counts = {0: 0, 1: 0}
        for _ in range(500):
            strat.set_q_values(np.array([1.0, 1.1]))
            counts[strat.explore(_obs())] += 1
        # Both actions should get significant selection at high temp
        assert counts[0] > 100
        assert counts[1] > 100

    def test_q_values_reset_after_explore(self):
        strat = BoltzmannStrategy(temperature=1.0)
        strat.set_q_values(np.array([1.0, 2.0]))
        strat.explore(_obs())
        # Q-values should be None after explore (reset)
        assert strat._q_values is None

    def test_temperature_annealing(self):
        strat = BoltzmannStrategy(temperature=1.0, temp_decay=0.9,
                                  min_temperature=0.1)
        initial_temp = strat.temperature
        strat.on_episode_end(score=1)
        assert strat.temperature < initial_temp

    def test_temperature_respects_minimum(self):
        strat = BoltzmannStrategy(temperature=0.11, temp_decay=0.5,
                                  min_temperature=0.1)
        strat.on_episode_end(score=1)
        assert strat.temperature >= 0.1

    def test_registered_in_strategy_map(self):
        assert "boltzmann" in STRATEGY_MAP
        assert "boltzmann" in STRATEGY_OPTIONS


# ===================================================================
# GuidedStrategy (adaptive noise)
# ===================================================================

class TestGuidedStrategy:
    """Tests for the guided strategy with adaptive noise decay."""

    def test_explore_returns_valid_action(self):
        strat = GuidedStrategy()
        action = strat.explore(_obs())
        assert action in (0, 1)

    def test_noise_decays_on_episode_end(self):
        strat = GuidedStrategy(noise=0.1, noise_decay=0.9, min_noise=0.001)
        initial_noise = strat._current_noise
        strat.on_episode_end(score=1)
        assert strat._current_noise < initial_noise

    def test_noise_respects_minimum(self):
        strat = GuidedStrategy(noise=0.01, noise_decay=0.1, min_noise=0.005)
        strat.on_episode_end(score=5)
        assert strat._current_noise >= 0.005

    def test_higher_score_decays_noise_faster(self):
        strat1 = GuidedStrategy(noise=0.1, noise_decay=0.99, min_noise=0.001)
        strat2 = GuidedStrategy(noise=0.1, noise_decay=0.99, min_noise=0.001)
        strat1.on_episode_end(score=1)
        strat2.on_episode_end(score=10)
        assert strat2._current_noise < strat1._current_noise


# ===================================================================
# cosine_epsilon_schedule
# ===================================================================

class TestCosineEpsilonSchedule:
    """Tests for the cosine annealing epsilon scheduler."""

    def test_starts_at_epsilon_start(self):
        eps = cosine_epsilon_schedule(0, 100, epsilon_start=1.0, epsilon_end=0.01)
        assert eps == pytest.approx(1.0)

    def test_ends_at_epsilon_end(self):
        eps = cosine_epsilon_schedule(99, 100, epsilon_start=1.0, epsilon_end=0.01)
        assert eps == pytest.approx(0.01)

    def test_midpoint_is_between_start_and_end(self):
        eps = cosine_epsilon_schedule(50, 100, epsilon_start=1.0, epsilon_end=0.01)
        assert 0.01 < eps < 1.0

    def test_monotonically_decreasing(self):
        epsilons = [cosine_epsilon_schedule(i, 100) for i in range(100)]
        for i in range(len(epsilons) - 1):
            assert epsilons[i] >= epsilons[i + 1]

    def test_single_episode_returns_end(self):
        eps = cosine_epsilon_schedule(0, 1, epsilon_start=1.0, epsilon_end=0.01)
        assert eps == pytest.approx(0.01)

    def test_beyond_total_episodes_returns_end(self):
        eps = cosine_epsilon_schedule(200, 100, epsilon_start=1.0, epsilon_end=0.01)
        assert eps == pytest.approx(0.01)

    def test_custom_range(self):
        eps = cosine_epsilon_schedule(0, 50, epsilon_start=0.5, epsilon_end=0.1)
        assert eps == pytest.approx(0.5)
        eps = cosine_epsilon_schedule(49, 50, epsilon_start=0.5, epsilon_end=0.1)
        assert eps == pytest.approx(0.1)
