"""Deep Q-Network (DQN) agent with experience replay and target network."""

import json
import random
from collections import deque
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.base_agent import BaseAgent


def _init_weights(module: nn.Module) -> None:
    """Apply orthogonal initialization for better gradient flow."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
        nn.init.constant_(module.bias, 0.0)


class QNetwork(nn.Module):
    """Feedforward Q-network with configurable hidden layers.

    Architecture: input -> [hidden_i -> ReLU]* -> output (no activation).
    Uses orthogonal initialization for better gradient flow.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        prev_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ReplayBuffer:
    """Fixed-size deque-based experience replay buffer."""

    def __init__(self, max_size: int = 50000):
        self.buffer: deque = deque(maxlen=max_size)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a random batch of transitions.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as numpy arrays.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class SumTree:
    """Binary sum tree for O(log n) priority-based sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.write_pos = 0
        self.size = 0

    def _propagate(self, idx: int, change: float) -> None:
        """Update parent nodes iteratively."""
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def _retrieve(self, s: float) -> int:
        """Find the leaf node for a given cumulative sum."""
        idx = 0
        while True:
            left = 2 * idx + 1
            if left >= len(self.tree):
                return idx
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1

    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data) -> None:
        idx = self.write_pos + self.capacity - 1
        self.data[self.write_pos] = data
        self.update(idx, priority)
        self.write_pos = (self.write_pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float):
        idx = self._retrieve(s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer using a SumTree.

    Samples transitions proportional to their TD-error priority,
    with importance sampling weights for unbiased gradient updates.
    """

    def __init__(
        self,
        max_size: int = 50000,
        alpha: float = 0.6,
        epsilon: float = 1e-6,
    ):
        self.tree = SumTree(max_size)
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_priority = 1.0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition with max priority (ensures new experiences are sampled)."""
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, (state, action, reward, next_state, done))

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> Tuple[np.ndarray, ...]:
        """Sample a prioritized batch with importance sampling weights.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, is_weights).
        """
        indices = []
        priorities = []
        batch = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(priority)
            batch.append(data)

        states, actions, rewards, next_states, dones = zip(*batch)

        # Importance sampling weights
        probs = np.array(priorities) / self.tree.total()
        weights = (len(self) * probs) ** (-beta)
        weights /= weights.max()

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(indices, dtype=np.int64),
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(
        self, indices: np.ndarray, td_errors: np.ndarray
    ) -> None:
        """Update priorities based on new TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(int(idx), priority)

    def __len__(self) -> int:
        return self.tree.size


class DQNAgent(BaseAgent):
    """Deep Q-Network agent with experience replay and target network.

    Uses an online Q-network for action selection and learning, and a
    separate target network (soft-updated) for stable Q-value targets.

    Supports optional Prioritized Experience Replay (PER) and LR scheduling.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
        lr: float = 3e-4,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.99995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        tau: float = 0.005,
        train_every: int = 4,
        train_intensity: int = 2,
        use_per: bool = False,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 100000,
        lr_schedule: str = None,
        lr_schedule_steps: int = 100000,
    ):
        super().__init__(state_dim, action_dim)

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau
        self.train_every = train_every
        self.train_intensity = train_intensity
        self.lr = lr

        # PER configuration
        self.use_per = use_per
        self.per_alpha = per_alpha
        self._per_beta = per_beta_start
        self._per_beta_start = per_beta_start
        self._per_beta_increment = (1.0 - per_beta_start) / max(per_beta_frames, 1)

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.q_net = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # LR scheduler
        self.lr_schedule = lr_schedule
        self.lr_scheduler = None
        if lr_schedule == "cosine":
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=lr_schedule_steps, eta_min=lr * 0.01
            )

        # Replay buffer
        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                max_size=buffer_size, alpha=per_alpha
            )
        else:
            self.replay_buffer = ReplayBuffer(max_size=buffer_size)

        # Step counter for train_every
        self._step_count = 0

    def _get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Compute Q-values for all actions given a state.

        Args:
            state: A single state observation.

        Returns:
            Numpy array of Q-values, shape (action_dim,).
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return q_values.cpu().numpy().squeeze(0)

    def get_activations(self, state: np.ndarray) -> list[np.ndarray]:
        """Get activations at each layer for visualization."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        activations = [state]  # input layer
        with torch.no_grad():
            x = state_t
            for layer in self.q_net.network:
                x = layer(x)
                if isinstance(layer, nn.ReLU):
                    activations.append(x.cpu().numpy().squeeze(0))
            activations.append(x.cpu().numpy().squeeze(0))  # output
        return activations

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection using Q-values.

        Args:
            state: Current observation.
            training: If True, uses epsilon-greedy; if False, purely greedy.

        Returns:
            The chosen action index.
        """
        if training and np.random.random() < self.epsilon:
            return int(np.random.randint(self.action_dim))

        q_values = self._get_q_values(state)
        return int(np.argmax(q_values))

    def _update_target(self) -> None:
        """Soft update target network: target = tau*online + (1-tau)*target."""
        for q_param, t_param in zip(
            self.q_net.parameters(), self.target_net.parameters()
        ):
            t_param.data.copy_(
                self.tau * q_param.data + (1.0 - self.tau) * t_param.data
            )

    def _compute_loss(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        weights: np.ndarray = None,
    ) -> Tuple[torch.Tensor, float, np.ndarray]:
        """Compute the DQN loss for a batch of transitions.

        Args:
            states, actions, rewards, next_states, dones: Batch arrays.
            weights: Optional importance sampling weights for PER.

        Returns:
            Tuple of (loss tensor, mean Q-value float, per-sample TD errors).
        """
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values for taken actions
        q_values = self.q_net(states_t)
        q_taken = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            target = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        # Per-sample TD errors (for PER priority updates)
        td_errors = (q_taken - target).detach().abs().cpu().numpy()

        # Weighted loss for PER, standard loss otherwise
        if weights is not None:
            weights_t = torch.FloatTensor(weights).to(self.device)
            element_loss = nn.functional.smooth_l1_loss(
                q_taken, target, reduction="none"
            )
            loss = (weights_t * element_loss).mean()
        else:
            loss = nn.functional.smooth_l1_loss(q_taken, target)

        q_mean = q_values.detach().mean().item()

        return loss, q_mean, td_errors

    def train_step(self, state, action, reward, next_state, done) -> dict:
        """Perform one training step: store transition, optionally learn.

        Adds the transition to the replay buffer. Every `train_every` steps
        (if enough samples), samples a batch, computes loss, updates the
        online network, and soft-updates the target network.

        When PER is enabled, samples are drawn proportional to TD-error
        priority and importance sampling weights correct for the bias.

        Args:
            state: State at time t.
            action: Action taken.
            reward: Reward received.
            next_state: State at time t+1.
            done: Whether the episode ended.

        Returns:
            Dict with optional "loss" and "q_mean" keys.
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
        self._step_count += 1

        metrics = {}

        # Train if we have enough samples and it's the right step
        if (
            len(self.replay_buffer) >= self.batch_size
            and self._step_count % self.train_every == 0
        ):
            for _ in range(self.train_intensity):
                if self.use_per:
                    (
                        states, actions, rewards_b, next_states, dones,
                        indices, is_weights,
                    ) = self.replay_buffer.sample(
                        self.batch_size, beta=self._per_beta
                    )
                    loss, q_mean, td_errors = self._compute_loss(
                        states, actions, rewards_b, next_states, dones,
                        weights=is_weights,
                    )
                    self.replay_buffer.update_priorities(indices, td_errors)
                else:
                    batch = self.replay_buffer.sample(self.batch_size)
                    loss, q_mean, _td_errors = self._compute_loss(*batch)

                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping
                nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
                self.optimizer.step()

            # Soft update target network (once per step, outside inner loop)
            self._update_target()

            # Step LR scheduler after each training update
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            metrics["loss"] = loss.item()
            metrics["q_mean"] = q_mean

        # Decay epsilon every step
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay,
        )

        # Anneal PER beta toward 1.0
        if self.use_per:
            self._per_beta = min(
                1.0, self._per_beta + self._per_beta_increment
            )

        return metrics

    def save(self, path: Path) -> None:
        """Save networks and parameters to disk.

        Creates a directory with:
          - q_net.pt: online network state dict
          - target_net.pt: target network state dict
          - params.json: hyperparameters and current epsilon

        Args:
            path: Directory to save to.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.q_net.state_dict(), path / "q_net.pt")
        torch.save(self.target_net.state_dict(), path / "target_net.pt")

        params = {
            "state_dim": int(self.state_dim),
            "action_dim": int(self.action_dim),
            "gamma": float(self.gamma),
            "epsilon": float(self.epsilon),
            "epsilon_end": float(self.epsilon_end),
            "epsilon_decay": float(self.epsilon_decay),
            "batch_size": int(self.batch_size),
            "tau": float(self.tau),
            "train_every": int(self.train_every),
            "train_intensity": int(self.train_intensity),
            "lr": float(self.lr),
            "step_count": int(self._step_count),
            "use_per": self.use_per,
            "per_beta": float(self._per_beta),
        }
        with open(path / "params.json", "w") as f:
            json.dump(params, f, indent=2)

    def load(self, path: Path) -> None:
        """Load networks and parameters from disk.

        Args:
            path: Directory to load from.
        """
        path = Path(path)

        self.q_net.load_state_dict(
            torch.load(path / "q_net.pt", map_location=self.device, weights_only=True)
        )
        self.target_net.load_state_dict(
            torch.load(path / "target_net.pt", map_location=self.device, weights_only=True)
        )
        self.target_net.eval()

        with open(path / "params.json", "r") as f:
            params = json.load(f)

        self.epsilon = params["epsilon"]
        self.gamma = params["gamma"]
        self.lr = params["lr"]
        self.tau = params["tau"]
        self.train_intensity = params.get("train_intensity", 4)
        self._step_count = params["step_count"]
        if "per_beta" in params:
            self._per_beta = params["per_beta"]

    def get_info(self) -> dict:
        """Return current agent info for logging.

        Returns:
            Dict with epsilon and step count.
        """
        info = {
            "epsilon": self.epsilon,
            "step_count": self._step_count,
        }
        if self.lr_scheduler is not None:
            info["lr"] = self.optimizer.param_groups[0]["lr"]
        if self.use_per:
            info["per_beta"] = self._per_beta
        return info

    def get_weights(self) -> dict:
        return {k: v.clone() for k, v in self.q_net.state_dict().items()}

    def set_weights(self, weights: dict) -> None:
        self.q_net.load_state_dict(weights)
        self.target_net.load_state_dict(weights)
        self.target_net.eval()

    def mutate(self, noise_scale: float = 0.1) -> None:
        with torch.no_grad():
            for param in self.q_net.parameters():
                param.add_(torch.randn_like(param) * noise_scale)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
