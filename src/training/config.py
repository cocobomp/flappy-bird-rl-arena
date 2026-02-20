"""Experiment configuration system for RL Flappy Bird training.

Provides a dataclass for experiment configuration and a YAML loader.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ExperimentConfig:
    """Configuration for a single RL training experiment.

    Attributes:
        agent: Agent type -- one of "q_learning", "dqn", "double_dqn".
        observation: Observation wrapper -- "simple" (4 features),
            "enriched" (7 features), or "raw" (12 features).
        reward: Reward function -- "basic", "distance", or "centered".
        hyperparams: Agent-specific hyperparameters (lr, gamma, epsilon, etc.).
        training: Training loop parameters (episodes, max_steps, save_every,
            log_every).
    """

    agent: str
    observation: str
    reward: str
    hyperparams: dict
    training: dict

    @property
    def experiment_name(self) -> str:
        """Generate a descriptive experiment name from the configuration.

        Returns:
            A string in the form '{agent}_{observation}_{reward}'.
        """
        return f"{self.agent}_{self.observation}_{self.reward}"


def load_config(path: Path) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        An ExperimentConfig instance populated from the YAML data.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return ExperimentConfig(
        agent=data["agent"],
        observation=data["observation"],
        reward=data["reward"],
        hyperparams=data.get("hyperparams", {}),
        training=data.get("training", {}),
    )
