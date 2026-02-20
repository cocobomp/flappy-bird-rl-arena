"""Supervised learning agents using scikit-learn (behavioral cloning).

These agents learn by imitating expert actions rather than through
reinforcement signals. They collect (state, action) pairs and
periodically retrain a sklearn classifier on the accumulated data.

Observations are 5D: [delta_y1, velocity, dist_pipe1, delta_y2, dist_pipe2]
Actions are binary: 0 (no flap) or 1 (flap).
"""

import pickle
import random
from abc import abstractmethod
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from src.agents.base_agent import BaseAgent


class SklearnAgent(BaseAgent):
    """Abstract base for all scikit-learn behavioral cloning agents.

    Collects (state, action) demonstration pairs in a rolling buffer
    and periodically retrains a sklearn classifier to imitate them.

    Exploration is controlled via epsilon-greedy during training:
    with probability epsilon the agent picks a random action, otherwise
    it follows the learned model's prediction.

    Subclasses must implement ``_create_model()`` to return a
    scikit-learn classifier instance.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_samples: int = 5000,
        retrain_every: int = 200,
        min_samples: int = 50,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.99995,
        **kwargs,
    ):
        """Initialize the sklearn agent.

        Args:
            state_dim: Dimensionality of the observation space.
            action_dim: Number of discrete actions.
            max_samples: Maximum number of (state, action) pairs to keep
                in the rolling buffer.
            retrain_every: Retrain the model every this many train_step calls.
            min_samples: Minimum samples required before training can begin.
            epsilon_start: Initial exploration probability.
            epsilon_end: Minimum exploration probability.
            epsilon_decay: Multiplicative decay applied to epsilon each step.
            **kwargs: Extra keyword arguments (ignored, for subclass compat).
        """
        super().__init__(state_dim, action_dim)

        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Buffer parameters
        self.max_samples = max_samples
        self.retrain_every = retrain_every
        self.min_samples = min_samples

        # Rolling buffer of (state, action) demonstration pairs
        self._buffer: deque = deque(maxlen=max_samples)

        # Step counter for periodic retraining
        self._step_count: int = 0

        # Whether the model has been trained at least once
        self._trained: bool = False

        # Create the underlying sklearn model
        self.model = self._create_model()

    @property
    def trained(self) -> bool:
        """Whether the model has been trained at least once."""
        return self._trained

    # ------------------------------------------------------------------
    # Abstract method for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _create_model(self):
        """Create and return a scikit-learn classifier instance.

        Returns:
            A scikit-learn classifier (e.g. RandomForestClassifier).
        """
        pass

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select an action via the trained model or randomly.

        If the model has not been trained yet, a random action is returned.
        During training, epsilon-greedy exploration is applied.

        Args:
            state: Current environment observation.
            training: If True, applies epsilon-greedy exploration.

        Returns:
            The chosen action index (0 or 1).
        """
        # Epsilon-greedy exploration during training
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        # If model not trained yet, return random action
        if not self._trained:
            return random.randint(0, self.action_dim - 1)

        # Use model prediction
        q_values = self._get_q_values(state)
        return int(np.argmax(q_values))

    def train_step(self, state, action, reward, next_state, done) -> dict:
        """Store a demonstration pair and periodically retrain the model.

        On non-terminal steps, stores (state, action) as a positive
        demonstration. On terminal steps (done=True), stores
        (state, 1-action) since the terminal action was bad and the
        opposite action would have been preferable.

        Epsilon is decayed every step. The model is retrained every
        ``retrain_every`` steps, provided enough samples with both
        classes are available.

        Args:
            state: State at time t.
            action: Action taken at time t.
            reward: Reward received (unused for behavioral cloning).
            next_state: State at time t+1 (unused).
            done: Whether the episode terminated.

        Returns:
            Dict with training metrics.
        """
        # Store demonstration pair
        if not done:
            # Normal step: the action taken was reasonable
            self._buffer.append((np.array(state, dtype=np.float32), int(action)))
        else:
            # Terminal step: the action led to death, store the opposite
            self._buffer.append((np.array(state, dtype=np.float32), 1 - int(action)))

        self._step_count += 1

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Periodic retraining
        retrained = False
        if (
            self._step_count % self.retrain_every == 0
            and len(self._buffer) >= self.min_samples
        ):
            retrained = self._retrain()

        return {
            "retrained": retrained,
            "samples": len(self._buffer),
            "epsilon": self.epsilon,
        }

    def save(self, path: Path) -> None:
        """Save the model and replay buffer to disk.

        Creates a directory containing:
          - model.pkl: the sklearn classifier and buffer data
          - params.pkl: agent hyperparameters and state

        Args:
            path: Directory to save to.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model, buffer, and parameters together
        data = {
            "model": self.model,
            "buffer": list(self._buffer),
            "trained": self._trained,
            "epsilon": self.epsilon,
            "step_count": self._step_count,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "max_samples": self.max_samples,
            "retrain_every": self.retrain_every,
            "min_samples": self.min_samples,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
        }
        with open(path / "sklearn_agent.pkl", "wb") as f:
            pickle.dump(data, f)

    def load(self, path: Path) -> None:
        """Load the model and replay buffer from disk.

        Args:
            path: Directory to load from.
        """
        path = Path(path)

        with open(path / "sklearn_agent.pkl", "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self._buffer = deque(data["buffer"], maxlen=data.get("max_samples", self.max_samples))
        self._trained = data["trained"]
        self.epsilon = data["epsilon"]
        self._step_count = data["step_count"]
        self.epsilon_end = data.get("epsilon_end", self.epsilon_end)
        self.epsilon_decay = data.get("epsilon_decay", self.epsilon_decay)

    def get_info(self) -> dict:
        """Return current agent info for logging.

        Returns:
            Dict with epsilon, trained status, and sample count.
        """
        return {
            "epsilon": self.epsilon,
            "trained": self._trained,
            "samples": len(self._buffer),
        }

    def get_weights(self) -> dict:
        """Sklearn agents do not support weight extraction.

        Returns:
            None.
        """
        return None

    def set_weights(self, weights: dict) -> None:
        """Sklearn agents do not support weight injection (no-op)."""
        pass

    def mutate(self, noise_scale: float = 0.1) -> None:
        """Sklearn agents do not support mutation (no-op)."""
        pass

    # ------------------------------------------------------------------
    # Sklearn-specific methods
    # ------------------------------------------------------------------

    def _get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Return class probabilities as pseudo Q-values for display.

        Uses ``predict_proba`` to get the probability of each action
        class, which serves as a proxy for Q-values in the renderer.

        Args:
            state: Current environment observation.

        Returns:
            Numpy array of shape (action_dim,) with class probabilities.
        """
        if not self._trained:
            return np.full(self.action_dim, 0.5)

        state_2d = np.array(state, dtype=np.float32).reshape(1, -1)
        proba = self.model.predict_proba(state_2d)[0]

        # predict_proba may return fewer columns if only one class was
        # seen during training; pad to action_dim if necessary.
        if len(proba) < self.action_dim:
            full_proba = np.full(self.action_dim, 0.5)
            classes = self.model.classes_
            for i, cls in enumerate(classes):
                if cls < self.action_dim:
                    full_proba[int(cls)] = proba[i]
            return full_proba

        return proba

    def get_feature_importances(self) -> Optional[np.ndarray]:
        """Return feature importances if the model supports them.

        Available for tree-based models (RandomForest, GradientBoosting).
        Returns None for models without feature importances (KNN, SVM).

        Returns:
            Numpy array of feature importances or None.
        """
        if not self._trained:
            return None

        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_

        return None

    def _retrain(self) -> bool:
        """Retrain the model on the current buffer contents.

        Only retrains if both action classes (0 and 1) are present
        in the buffer, since sklearn classifiers require at least
        two classes for predict_proba.

        Returns:
            True if retraining was performed, False otherwise.
        """
        states = np.array([s for s, _ in self._buffer], dtype=np.float32)
        actions = np.array([a for _, a in self._buffer], dtype=np.int64)

        # Ensure both classes are present
        unique_classes = set(actions)
        if len(unique_classes) < 2:
            return False

        # Retrain the model from scratch on the full buffer
        self.model.fit(states, actions)
        self._trained = True
        return True


class RandomForestAgent(SklearnAgent):
    """Behavioral cloning agent using a Random Forest classifier.

    Random forests are an ensemble of decision trees that vote on the
    predicted class. They provide feature importance scores and are
    robust to overfitting.
    """

    def _create_model(self):
        """Create a RandomForestClassifier.

        Returns:
            A RandomForestClassifier instance with 50 trees and max depth 8.
        """
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            random_state=42,
        )


class GradientBoostAgent(SklearnAgent):
    """Behavioral cloning agent using a Gradient Boosting classifier.

    Gradient boosting builds trees sequentially, with each new tree
    correcting the errors of the previous ensemble. Often achieves
    higher accuracy than random forests but can be slower to train.
    """

    def _create_model(self):
        """Create a GradientBoostingClassifier.

        Returns:
            A GradientBoostingClassifier with 50 estimators, depth 4, lr 0.1.
        """
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )


class KNNAgent(SklearnAgent):
    """Behavioral cloning agent using K-Nearest Neighbors.

    KNN classifies new states by finding the K closest states in the
    buffer and taking a majority vote on their actions. Simple and
    non-parametric, but does not provide feature importances.
    """

    def _create_model(self):
        """Create a KNeighborsClassifier.

        Returns:
            A KNeighborsClassifier with 11 neighbors.
        """
        from sklearn.neighbors import KNeighborsClassifier

        return KNeighborsClassifier(n_neighbors=11)


class SVMAgent(SklearnAgent):
    """Behavioral cloning agent using a Support Vector Machine.

    Uses an RBF kernel SVM with probability estimates enabled.
    SVMs find the optimal hyperplane separating action classes and
    can handle non-linear boundaries via the kernel trick.
    """

    def _create_model(self):
        """Create an SVC with probability estimates.

        Returns:
            An SVC with RBF kernel and probability=True.
        """
        from sklearn.svm import SVC

        return SVC(
            kernel="rbf",
            probability=True,
            C=1.0,
            random_state=42,
        )
