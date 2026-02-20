"""Abstract base class for all RL agents."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseAgent(ABC):
    """Base interface that all agents must implement.

    Provides a common contract for action selection, training,
    serialization, and introspection across different RL algorithms.
    """

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim

    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select an action given the current state.

        Args:
            state: Current environment observation.
            training: If True, the agent may explore (e.g. epsilon-greedy).

        Returns:
            The chosen action index.
        """
        pass

    @abstractmethod
    def train_step(self, state, action, reward, next_state, done) -> dict:
        """Perform one training step with a single transition.

        Args:
            state: State at time t.
            action: Action taken at time t.
            reward: Reward received after taking action.
            next_state: State at time t+1.
            done: Whether the episode terminated.

        Returns:
            A dict of training metrics (e.g. {"loss": 0.5}).
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save the agent's learned parameters to disk.

        Args:
            path: Directory or file path to save to.
        """
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load the agent's learned parameters from disk.

        Args:
            path: Directory or file path to load from.
        """
        pass

    @abstractmethod
    def get_info(self) -> dict:
        """Return agent info for logging purposes.

        Returns:
            A dict of current agent state (e.g. epsilon, lr, table size).
        """
        pass

    @abstractmethod
    def get_weights(self) -> dict:
        """Return a copy of the agent's learned parameters."""
        pass

    @abstractmethod
    def set_weights(self, weights: dict) -> None:
        """Replace the agent's learned parameters with a copy."""
        pass

    @abstractmethod
    def mutate(self, noise_scale: float = 0.1) -> None:
        """Add random noise to the agent's parameters (for evolution)."""
        pass
