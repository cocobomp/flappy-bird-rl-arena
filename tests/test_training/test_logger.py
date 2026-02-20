"""Tests for the MetricsLogger."""

import pytest

from src.training.logger import MetricsLogger


class TestMetricsLogger:
    """Test the MetricsLogger class."""

    def test_initial_state(self):
        logger = MetricsLogger()
        assert logger.history == []

    def test_log_single_episode(self):
        logger = MetricsLogger()
        logger.log_episode(episode=0, score=1, reward=10.5, steps=50)
        assert len(logger.history) == 1
        entry = logger.history[0]
        assert entry["episode"] == 0
        assert entry["score"] == 1
        assert entry["reward"] == 10.5
        assert entry["steps"] == 50

    def test_log_multiple_episodes(self):
        logger = MetricsLogger()
        for i in range(5):
            logger.log_episode(episode=i, score=i * 2, reward=float(i), steps=100 + i)
        assert len(logger.history) == 5
        assert logger.history[0]["episode"] == 0
        assert logger.history[4]["episode"] == 4

    def test_log_episode_with_agent_info(self):
        logger = MetricsLogger()
        agent_info = {"epsilon": 0.5, "loss": 0.02}
        logger.log_episode(episode=0, score=3, reward=15.0, steps=80, agent_info=agent_info)
        entry = logger.history[0]
        assert entry["epsilon"] == 0.5
        assert entry["loss"] == 0.02

    def test_log_episode_without_agent_info(self):
        logger = MetricsLogger()
        logger.log_episode(episode=0, score=1, reward=5.0, steps=20)
        entry = logger.history[0]
        assert "episode" in entry
        assert "score" in entry
        assert "reward" in entry
        assert "steps" in entry


class TestGetRecent:
    """Test the get_recent method."""

    def test_get_recent_default(self):
        logger = MetricsLogger()
        for i in range(20):
            logger.log_episode(episode=i, score=i, reward=float(i), steps=50)
        recent = logger.get_recent()
        assert len(recent) == 10
        # Most recent entries
        assert recent[0]["episode"] == 10
        assert recent[-1]["episode"] == 19

    def test_get_recent_custom_n(self):
        logger = MetricsLogger()
        for i in range(20):
            logger.log_episode(episode=i, score=i, reward=float(i), steps=50)
        recent = logger.get_recent(n=5)
        assert len(recent) == 5

    def test_get_recent_less_than_n(self):
        logger = MetricsLogger()
        for i in range(3):
            logger.log_episode(episode=i, score=i, reward=float(i), steps=50)
        recent = logger.get_recent(n=10)
        assert len(recent) == 3

    def test_get_recent_empty(self):
        logger = MetricsLogger()
        recent = logger.get_recent()
        assert recent == []


class TestBestScore:
    """Test the best_score property."""

    def test_best_score_single(self):
        logger = MetricsLogger()
        logger.log_episode(episode=0, score=5, reward=10.0, steps=50)
        assert logger.best_score == 5

    def test_best_score_multiple(self):
        logger = MetricsLogger()
        scores = [3, 7, 1, 10, 4]
        for i, s in enumerate(scores):
            logger.log_episode(episode=i, score=s, reward=float(s), steps=50)
        assert logger.best_score == 10

    def test_best_score_all_zero(self):
        logger = MetricsLogger()
        for i in range(5):
            logger.log_episode(episode=i, score=0, reward=1.0, steps=50)
        assert logger.best_score == 0

    def test_best_score_empty(self):
        logger = MetricsLogger()
        assert logger.best_score == 0


class TestAverageScore:
    """Test the average_score method."""

    def test_average_score_default(self):
        logger = MetricsLogger()
        for i in range(10):
            logger.log_episode(episode=i, score=10, reward=1.0, steps=50)
        assert logger.average_score() == 10.0

    def test_average_score_custom_n(self):
        logger = MetricsLogger()
        # Log 20 episodes with score=i
        for i in range(20):
            logger.log_episode(episode=i, score=i, reward=float(i), steps=50)
        # Last 5 episodes: scores 15,16,17,18,19 => avg = 17.0
        avg = logger.average_score(n=5)
        assert avg == 17.0

    def test_average_score_less_than_n(self):
        logger = MetricsLogger()
        logger.log_episode(episode=0, score=4, reward=1.0, steps=50)
        logger.log_episode(episode=1, score=6, reward=1.0, steps=50)
        # Only 2 episodes, asking for last 100 => average of all = 5.0
        assert logger.average_score(n=100) == 5.0

    def test_average_score_empty(self):
        logger = MetricsLogger()
        assert logger.average_score() == 0.0

    def test_average_score_single(self):
        logger = MetricsLogger()
        logger.log_episode(episode=0, score=7, reward=1.0, steps=50)
        assert logger.average_score(n=1) == 7.0
