"""Tabular Q-Learning agent with state discretization."""

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.agents.base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """Tabular Q-Learning with epsilon-greedy exploration.

    States are discretized into bins for tabular lookup.
    The Q-table is stored as a defaultdict mapping discretized
    state tuples to numpy arrays of Q-values per action.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_bins: int = 10,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        super().__init__(state_dim, action_dim)
        self.n_bins = n_bins
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.q_table: defaultdict = defaultdict(
            lambda: np.zeros(self.action_dim)
        )

        # Pre-compute bin edges for discretization: clips to [-1, 1]
        self._bin_edges = np.linspace(-1.0, 1.0, n_bins + 1)[1:-1]

    def _discretize(self, state: np.ndarray) -> tuple:
        """Clip state to [-1, 1] and digitize into bin indices.

        Args:
            state: Continuous state vector.

        Returns:
            Tuple of bin indices, one per dimension.
        """
        clipped = np.clip(state, -1.0, 1.0)
        indices = np.digitize(clipped, self._bin_edges)
        return tuple(indices)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection.

        Args:
            state: Current environment observation.
            training: If True, uses epsilon-greedy; if False, purely greedy.

        Returns:
            The chosen action index.
        """
        if training and np.random.random() < self.epsilon:
            return int(np.random.randint(self.action_dim))

        key = self._discretize(state)
        q_values = self.q_table[key]
        return int(np.argmax(q_values))

    def train_step(self, state, action, reward, next_state, done) -> dict:
        """Perform one Q-learning update.

        Q(s,a) <- Q(s,a) + lr * (r + gamma * max_a' Q(s',a') - Q(s,a))
        When done=True, the target is simply r.

        Args:
            state: State at time t.
            action: Action taken.
            reward: Reward received.
            next_state: State at time t+1.
            done: Whether the episode ended.

        Returns:
            Dict with training metrics.
        """
        key = self._discretize(state)
        next_key = self._discretize(next_state)

        current_q = self.q_table[key][action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_key])

        # Q-learning update
        self.q_table[key][action] = current_q + self.lr * (target - current_q)

        # Decay epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay,
        )

        return {"td_error": abs(target - current_q)}

    def save(self, path: Path) -> None:
        """Save Q-table and parameters to disk.

        Creates a directory with:
          - q_table.pkl: pickled Q-table (converted to regular dict)
          - params.json: agent hyperparameters and current epsilon

        Args:
            path: Directory to save to.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save Q-table as pickle (convert to regular dict for portability)
        with open(path / "q_table.pkl", "wb") as f:
            pickle.dump(dict(self.q_table), f)

        # Save parameters as JSON
        params = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "n_bins": self.n_bins,
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
        }
        with open(path / "params.json", "w") as f:
            json.dump(params, f, indent=2)

    def load(self, path: Path) -> None:
        """Load Q-table and parameters from disk.

        Args:
            path: Directory to load from.
        """
        path = Path(path)

        # Load Q-table
        with open(path / "q_table.pkl", "rb") as f:
            loaded_dict = pickle.load(f)

        self.q_table = defaultdict(lambda: np.zeros(self.action_dim))
        self.q_table.update(loaded_dict)

        # Load parameters
        with open(path / "params.json", "r") as f:
            params = json.load(f)

        self.epsilon = params["epsilon"]
        self.lr = params["lr"]
        self.gamma = params["gamma"]
        self.epsilon_end = params["epsilon_end"]
        self.epsilon_decay = params["epsilon_decay"]

    def get_info(self) -> dict:
        """Return current agent info for logging.

        Returns:
            Dict with epsilon and Q-table size.
        """
        return {
            "epsilon": self.epsilon,
            "q_table_size": len(self.q_table),
        }

    def get_weights(self) -> dict:
        return {k: v.copy() for k, v in self.q_table.items()}

    def set_weights(self, weights: dict) -> None:
        self.q_table = defaultdict(lambda: np.zeros(self.action_dim))
        for k, v in weights.items():
            self.q_table[k] = v.copy()

    def mutate(self, noise_scale: float = 0.1) -> None:
        for key in list(self.q_table.keys()):
            self.q_table[key] += np.random.normal(0, noise_scale, self.action_dim)
