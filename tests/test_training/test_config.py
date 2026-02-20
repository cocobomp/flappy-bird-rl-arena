"""Tests for the experiment configuration system."""

import pytest
from pathlib import Path

from src.training.config import ExperimentConfig, load_config


class TestExperimentConfig:
    """Test the ExperimentConfig dataclass."""

    def test_create_config(self):
        config = ExperimentConfig(
            agent="dqn",
            observation="simple",
            reward="basic",
            hyperparams={"lr": 0.001, "gamma": 0.99},
            training={"episodes": 100, "max_steps": 500, "save_every": 50, "log_every": 10},
        )
        assert config.agent == "dqn"
        assert config.observation == "simple"
        assert config.reward == "basic"
        assert config.hyperparams["lr"] == 0.001
        assert config.training["episodes"] == 100

    def test_experiment_name_dqn_simple_basic(self):
        config = ExperimentConfig(
            agent="dqn",
            observation="simple",
            reward="basic",
            hyperparams={},
            training={},
        )
        assert config.experiment_name == "dqn_simple_basic"

    def test_experiment_name_q_learning_enriched_distance(self):
        config = ExperimentConfig(
            agent="q_learning",
            observation="enriched",
            reward="distance",
            hyperparams={},
            training={},
        )
        assert config.experiment_name == "q_learning_enriched_distance"

    def test_experiment_name_double_dqn_raw_centered(self):
        config = ExperimentConfig(
            agent="double_dqn",
            observation="raw",
            reward="centered",
            hyperparams={},
            training={},
        )
        assert config.experiment_name == "double_dqn_raw_centered"

    def test_hyperparams_is_dict(self):
        config = ExperimentConfig(
            agent="dqn",
            observation="simple",
            reward="basic",
            hyperparams={"lr": 0.0005, "hidden_dims": [128, 128]},
            training={},
        )
        assert isinstance(config.hyperparams, dict)
        assert config.hyperparams["hidden_dims"] == [128, 128]

    def test_training_is_dict(self):
        config = ExperimentConfig(
            agent="dqn",
            observation="simple",
            reward="basic",
            hyperparams={},
            training={"episodes": 1000, "max_steps": 500, "save_every": 100, "log_every": 10},
        )
        assert isinstance(config.training, dict)
        assert config.training["max_steps"] == 500
        assert config.training["save_every"] == 100
        assert config.training["log_every"] == 10


class TestLoadConfig:
    """Test loading configuration from YAML files."""

    def test_load_default_config(self, tmp_path):
        yaml_content = (
            "agent: dqn\n"
            "observation: simple\n"
            "reward: basic\n"
            "hyperparams:\n"
            "  lr: 0.0005\n"
            "  gamma: 0.99\n"
            "  epsilon_start: 1.0\n"
            "  epsilon_end: 0.01\n"
            "  epsilon_decay: 0.995\n"
            "  hidden_dims: [128, 128]\n"
            "  batch_size: 64\n"
            "  buffer_size: 50000\n"
            "  tau: 0.005\n"
            "training:\n"
            "  episodes: 1000\n"
            "  max_steps: 500\n"
            "  save_every: 100\n"
            "  log_every: 10\n"
        )
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)

        config = load_config(yaml_file)

        assert isinstance(config, ExperimentConfig)
        assert config.agent == "dqn"
        assert config.observation == "simple"
        assert config.reward == "basic"
        assert config.hyperparams["lr"] == 0.0005
        assert config.hyperparams["gamma"] == 0.99
        assert config.hyperparams["epsilon_start"] == 1.0
        assert config.hyperparams["epsilon_end"] == 0.01
        assert config.hyperparams["epsilon_decay"] == 0.995
        assert config.hyperparams["hidden_dims"] == [128, 128]
        assert config.hyperparams["batch_size"] == 64
        assert config.hyperparams["buffer_size"] == 50000
        assert config.hyperparams["tau"] == 0.005
        assert config.training["episodes"] == 1000
        assert config.training["max_steps"] == 500
        assert config.training["save_every"] == 100
        assert config.training["log_every"] == 10

    def test_load_config_experiment_name(self, tmp_path):
        yaml_content = (
            "agent: q_learning\n"
            "observation: enriched\n"
            "reward: centered\n"
            "hyperparams:\n"
            "  lr: 0.1\n"
            "training:\n"
            "  episodes: 500\n"
        )
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)

        config = load_config(yaml_file)
        assert config.experiment_name == "q_learning_enriched_centered"

    def test_load_config_returns_correct_types(self, tmp_path):
        yaml_content = (
            "agent: double_dqn\n"
            "observation: raw\n"
            "reward: distance\n"
            "hyperparams:\n"
            "  lr: 0.001\n"
            "  hidden_dims: [64, 32]\n"
            "training:\n"
            "  episodes: 200\n"
            "  max_steps: 300\n"
        )
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)

        config = load_config(yaml_file)
        assert isinstance(config, ExperimentConfig)
        assert isinstance(config.hyperparams, dict)
        assert isinstance(config.training, dict)
        assert config.hyperparams["hidden_dims"] == [64, 32]

    def test_load_config_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/path/config.yaml"))

    def test_load_actual_default_config(self):
        """Test loading the actual default.yaml from the configs directory."""
        config_path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
        config = load_config(config_path)

        assert config.agent == "dqn"
        assert config.observation == "simple"
        assert config.reward == "basic"
        assert config.experiment_name == "dqn_simple_basic"
