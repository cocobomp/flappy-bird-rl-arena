"""Tests for the Trainer class."""

import gymnasium
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.training.config import ExperimentConfig
from src.training.logger import MetricsLogger
from src.training.trainer import Trainer, AGENT_MAP, OBS_WRAPPER_MAP, REWARD_MAP

from src.agents import QLearningAgent, DQNAgent, DoubleDQNAgent, BaseAgent
from src.environments.wrappers import SimpleObsWrapper, EnrichedObsWrapper
from src.environments.rewards import BasicReward, DistanceReward, CenteredReward


# ---------------------------------------------------------------------------
# Fake environment that behaves like FlappyBird-v0 (use_lidar=False)
# ---------------------------------------------------------------------------

class FakeFlappyEnv(gymnasium.Env):
    """Minimal fake of FlappyBird-v0 for deterministic trainer testing.

    Terminates after a fixed number of steps to keep tests fast.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 5, **kwargs):
        super().__init__()
        self.action_space = gymnasium.spaces.Discrete(2)
        self.observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float64,
        )
        self._max_episode_steps = max_episode_steps
        self._step_count = 0

    def _make_obs(self) -> np.ndarray:
        return np.random.uniform(-1, 1, size=(12,)).astype(np.float64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        return self._make_obs(), {"score": 0}

    def step(self, action):
        self._step_count += 1
        terminated = self._step_count >= self._max_episode_steps
        score = 1 if self._step_count % 3 == 0 else 0
        return self._make_obs(), 0.1, terminated, False, {"score": score}


def _make_config(
    agent="dqn",
    observation="simple",
    reward="basic",
    episodes=2,
    max_steps=50,
    **extra_hyperparams,
) -> ExperimentConfig:
    """Helper to create a short-running test config."""
    hyperparams = {
        "lr": 0.001,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.995,
        "hidden_dims": [32, 32],
        "batch_size": 8,
        "buffer_size": 1000,
        "tau": 0.005,
    }
    hyperparams.update(extra_hyperparams)
    return ExperimentConfig(
        agent=agent,
        observation=observation,
        reward=reward,
        hyperparams=hyperparams,
        training={
            "episodes": episodes,
            "max_steps": max_steps,
            "save_every": 100,  # large so we don't trigger saves during short tests
            "log_every": 1,
        },
    )


# ---------------------------------------------------------------------------
# Tests for mapping dictionaries
# ---------------------------------------------------------------------------

class TestMaps:
    """Verify the mapping dicts contain the right entries."""

    def test_agent_map_keys(self):
        assert set(AGENT_MAP.keys()) == {"q_learning", "dqn", "double_dqn"}

    def test_agent_map_values(self):
        assert AGENT_MAP["q_learning"] is QLearningAgent
        assert AGENT_MAP["dqn"] is DQNAgent
        assert AGENT_MAP["double_dqn"] is DoubleDQNAgent

    def test_obs_wrapper_map_keys(self):
        assert set(OBS_WRAPPER_MAP.keys()) == {"simple", "enriched", "raw"}

    def test_obs_wrapper_map_values(self):
        assert OBS_WRAPPER_MAP["simple"] is SimpleObsWrapper
        assert OBS_WRAPPER_MAP["enriched"] is EnrichedObsWrapper
        assert OBS_WRAPPER_MAP["raw"] is None

    def test_reward_map_keys(self):
        assert set(REWARD_MAP.keys()) == {"basic", "distance", "centered"}

    def test_reward_map_values(self):
        assert REWARD_MAP["basic"] is BasicReward
        assert REWARD_MAP["distance"] is DistanceReward
        assert REWARD_MAP["centered"] is CenteredReward


# ---------------------------------------------------------------------------
# Tests for _build_env
# ---------------------------------------------------------------------------

class TestBuildEnv:
    """Test that _build_env produces correctly wrapped environments."""

    @patch("src.training.trainer.gymnasium.make")
    def test_build_env_simple_obs_shape(self, mock_make):
        mock_make.return_value = FakeFlappyEnv()
        config = _make_config(observation="simple", reward="basic")
        trainer = Trainer(config)
        env = trainer._build_env()
        obs, _ = env.reset()
        assert obs.shape == (4,), f"Expected (4,) for simple obs, got {obs.shape}"
        env.close()

    @patch("src.training.trainer.gymnasium.make")
    def test_build_env_enriched_obs_shape(self, mock_make):
        mock_make.return_value = FakeFlappyEnv()
        config = _make_config(observation="enriched", reward="basic")
        trainer = Trainer(config)
        env = trainer._build_env()
        obs, _ = env.reset()
        assert obs.shape == (7,), f"Expected (7,) for enriched obs, got {obs.shape}"
        env.close()

    @patch("src.training.trainer.gymnasium.make")
    def test_build_env_raw_obs_shape(self, mock_make):
        mock_make.return_value = FakeFlappyEnv()
        config = _make_config(observation="raw", reward="basic")
        trainer = Trainer(config)
        env = trainer._build_env()
        obs, _ = env.reset()
        assert obs.shape == (12,), f"Expected (12,) for raw obs, got {obs.shape}"
        env.close()

    @patch("src.training.trainer.gymnasium.make")
    def test_build_env_action_space(self, mock_make):
        mock_make.return_value = FakeFlappyEnv()
        config = _make_config()
        trainer = Trainer(config)
        env = trainer._build_env()
        assert env.action_space.n == 2
        env.close()


# ---------------------------------------------------------------------------
# Tests for _build_agent
# ---------------------------------------------------------------------------

class TestBuildAgent:
    """Test that _build_agent creates the correct agent type."""

    @patch("src.training.trainer.gymnasium.make")
    def test_build_dqn_agent(self, mock_make):
        mock_make.return_value = FakeFlappyEnv()
        config = _make_config(agent="dqn", observation="simple")
        trainer = Trainer(config)
        env = trainer._build_env()
        agent = trainer._build_agent(env)
        assert isinstance(agent, DQNAgent)
        assert agent.state_dim == 4
        assert agent.action_dim == 2
        env.close()

    @patch("src.training.trainer.gymnasium.make")
    def test_build_double_dqn_agent(self, mock_make):
        mock_make.return_value = FakeFlappyEnv()
        config = _make_config(agent="double_dqn", observation="enriched")
        trainer = Trainer(config)
        env = trainer._build_env()
        agent = trainer._build_agent(env)
        assert isinstance(agent, DoubleDQNAgent)
        assert agent.state_dim == 7
        assert agent.action_dim == 2
        env.close()

    @patch("src.training.trainer.gymnasium.make")
    def test_build_q_learning_agent(self, mock_make):
        mock_make.return_value = FakeFlappyEnv()
        config = _make_config(agent="q_learning", observation="simple")
        trainer = Trainer(config)
        env = trainer._build_env()
        agent = trainer._build_agent(env)
        assert isinstance(agent, QLearningAgent)
        assert agent.state_dim == 4
        assert agent.action_dim == 2
        env.close()

    @patch("src.training.trainer.gymnasium.make")
    def test_build_agent_inherits_base(self, mock_make):
        mock_make.return_value = FakeFlappyEnv()
        config = _make_config(agent="dqn")
        trainer = Trainer(config)
        env = trainer._build_env()
        agent = trainer._build_agent(env)
        assert isinstance(agent, BaseAgent)
        env.close()

    @patch("src.training.trainer.gymnasium.make")
    def test_build_agent_raw_obs(self, mock_make):
        mock_make.return_value = FakeFlappyEnv()
        config = _make_config(agent="dqn", observation="raw")
        trainer = Trainer(config)
        env = trainer._build_env()
        agent = trainer._build_agent(env)
        assert agent.state_dim == 12
        env.close()


# ---------------------------------------------------------------------------
# Tests for train()
# ---------------------------------------------------------------------------

class TestTrain:
    """Test the training loop."""

    @patch("src.training.trainer.gymnasium.make")
    def test_train_returns_tuple(self, mock_make):
        mock_make.return_value = FakeFlappyEnv(max_episode_steps=5)
        config = _make_config(agent="dqn", episodes=2, max_steps=50)
        trainer = Trainer(config)
        result = trainer.train()
        assert isinstance(result, tuple)
        assert len(result) == 2

    @patch("src.training.trainer.gymnasium.make")
    def test_train_returns_agent_and_logger(self, mock_make):
        mock_make.return_value = FakeFlappyEnv(max_episode_steps=5)
        config = _make_config(agent="dqn", episodes=2, max_steps=50)
        trainer = Trainer(config)
        agent, logger = trainer.train()
        assert isinstance(agent, BaseAgent)
        assert isinstance(logger, MetricsLogger)

    @patch("src.training.trainer.gymnasium.make")
    def test_train_correct_history_length(self, mock_make):
        mock_make.return_value = FakeFlappyEnv(max_episode_steps=5)
        config = _make_config(agent="dqn", episodes=3, max_steps=50)
        trainer = Trainer(config)
        agent, logger = trainer.train()
        assert len(logger.history) == 3

    @patch("src.training.trainer.gymnasium.make")
    def test_train_history_entries_have_required_keys(self, mock_make):
        mock_make.return_value = FakeFlappyEnv(max_episode_steps=5)
        config = _make_config(agent="dqn", episodes=2, max_steps=50)
        trainer = Trainer(config)
        _, logger = trainer.train()
        for entry in logger.history:
            assert "episode" in entry
            assert "score" in entry
            assert "reward" in entry
            assert "steps" in entry

    @patch("src.training.trainer.gymnasium.make")
    def test_train_q_learning(self, mock_make):
        mock_make.return_value = FakeFlappyEnv(max_episode_steps=5)
        config = _make_config(agent="q_learning", episodes=3, max_steps=50)
        trainer = Trainer(config)
        agent, logger = trainer.train()
        assert isinstance(agent, QLearningAgent)
        assert len(logger.history) == 3

    @patch("src.training.trainer.gymnasium.make")
    def test_train_double_dqn(self, mock_make):
        mock_make.return_value = FakeFlappyEnv(max_episode_steps=5)
        config = _make_config(agent="double_dqn", episodes=2, max_steps=50)
        trainer = Trainer(config)
        agent, logger = trainer.train()
        assert isinstance(agent, DoubleDQNAgent)
        assert len(logger.history) == 2

    @patch("src.training.trainer.gymnasium.make")
    def test_train_respects_max_steps(self, mock_make):
        # FakeFlappyEnv never terminates on its own with very high max
        mock_make.return_value = FakeFlappyEnv(max_episode_steps=999)
        config = _make_config(agent="dqn", episodes=1, max_steps=10)
        trainer = Trainer(config)
        _, logger = trainer.train()
        # Steps should be capped at max_steps
        assert logger.history[0]["steps"] <= 10

    @patch("src.training.trainer.gymnasium.make")
    def test_train_episode_indices_sequential(self, mock_make):
        mock_make.return_value = FakeFlappyEnv(max_episode_steps=5)
        config = _make_config(agent="dqn", episodes=3, max_steps=50)
        trainer = Trainer(config)
        _, logger = trainer.train()
        episodes = [e["episode"] for e in logger.history]
        assert episodes == [0, 1, 2]
