"""REINFORCE (Policy Gradient) agent with baseline for variance reduction."""

import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.base_agent import BaseAgent


class PolicyNetwork(nn.Module):
    """Feedforward policy network that outputs action probabilities.

    Architecture: input -> [hidden_i -> ReLU]* -> output -> Softmax.
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
        layers.append(nn.Softmax(dim=-1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ReinforceAgent(BaseAgent):
    """REINFORCE with baseline (episode-based policy gradient).

    Buffers transitions during an episode, then trains at episode end.
    Uses mean episode return as baseline for variance reduction.

    Unlike DQN which learns Q-values, REINFORCE directly learns a policy
    (probability distribution over actions) and updates it using the
    policy gradient theorem.
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
    ):
        super().__init__(state_dim, action_dim)

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.lr = lr

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy network and optimizer
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dims).to(
            self.device
        )
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Episode buffer: stores (log_prob, reward) for each step in the episode
        self._log_probs: List[torch.Tensor] = []
        self._rewards: List[float] = []

        # Step counter (for compatibility with training loop info)
        self._step_count = 0

        # Cache the last log_prob from select_action so train_step can use it
        self._last_log_prob: torch.Tensor | None = None

    def _get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Return action probabilities for display compatibility.

        Although REINFORCE does not compute Q-values, the renderer expects
        this method to return per-action values for visualization. We return
        the policy's action probabilities instead.

        Args:
            state: A single state observation.

        Returns:
            Numpy array of action probabilities, shape (action_dim,).
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.policy_net(state_t)
        return probs.cpu().numpy().squeeze(0)

    def get_activations(self, state: np.ndarray) -> list[np.ndarray]:
        """Get activations at each layer for visualization.

        Args:
            state: A single state observation.

        Returns:
            List of numpy arrays, one per layer (input, hidden activations, output).
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        activations = [state]  # input layer
        with torch.no_grad():
            x = state_t
            for layer in self.policy_net.network:
                x = layer(x)
                if isinstance(layer, nn.ReLU):
                    activations.append(x.cpu().numpy().squeeze(0))
            activations.append(x.cpu().numpy().squeeze(0))  # output (probabilities)
        return activations

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select an action by sampling from the policy distribution.

        During training, samples from the categorical distribution defined
        by the policy network output. Also applies epsilon-greedy exploration
        for interface compatibility. During evaluation, picks the action with
        the highest probability (greedy).

        The log-probability of the chosen action is cached internally so
        that train_step can use it when computing the policy gradient.

        Args:
            state: Current environment observation.
            training: If True, sample from distribution (with epsilon-greedy);
                      if False, pick argmax (greedy).

        Returns:
            The chosen action index.
        """
        # Epsilon-greedy exploration (for training only)
        if training and np.random.random() < self.epsilon:
            self._last_log_prob = None
            return int(np.random.randint(self.action_dim))

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy_net(state_t)

        if training:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            self._last_log_prob = dist.log_prob(action)
            return action.item()
        else:
            # Greedy: pick the most probable action
            self._last_log_prob = None
            return probs.argmax(dim=-1).item()

    def train_step(self, state, action, reward, next_state, done) -> dict:
        """Buffer a transition and train at episode end.

        REINFORCE is an episode-based method. Each call appends the reward
        (and the cached log_prob from select_action) to the episode buffer.
        When done=True (episode ended), computes discounted returns, subtracts
        the mean return as a baseline, and performs a single policy gradient
        update over the entire episode.

        Args:
            state: State at time t.
            action: Action taken at time t.
            reward: Reward received after taking action.
            next_state: State at time t+1.
            done: Whether the episode terminated.

        Returns:
            Dict with optional "loss" key (present only when training occurs).
        """
        self._step_count += 1

        # Store the log_prob and reward for this step.
        # If epsilon-greedy exploration was used, log_prob may be None.
        # In that case, compute log_prob now for the action that was taken.
        if self._last_log_prob is not None:
            self._log_probs.append(self._last_log_prob)
        else:
            # Compute log_prob for the random action that was taken
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            probs = self.policy_net(state_t)
            dist = torch.distributions.Categorical(probs)
            action_t = torch.tensor(action, device=self.device)
            self._log_probs.append(dist.log_prob(action_t))

        self._rewards.append(reward)
        self._last_log_prob = None

        metrics = {}

        if done:
            # Compute discounted returns G_t for each timestep
            returns = []
            g = 0.0
            for r in reversed(self._rewards):
                g = r + self.gamma * g
                returns.insert(0, g)

            returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

            # Baseline: subtract mean return for variance reduction
            if len(returns_t) > 1:
                baseline = returns_t.mean()
                advantages = returns_t - baseline
            else:
                advantages = returns_t

            # Compute policy gradient loss: -sum(log_prob * advantage)
            log_probs_t = torch.stack(self._log_probs)
            loss = -(log_probs_t * advantages.detach()).sum()

            # Backprop and update
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()

            metrics["loss"] = loss.item()

            # Clear episode buffer
            self._log_probs.clear()
            self._rewards.clear()

        # Decay epsilon every step
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay,
        )

        return metrics

    def save(self, path: Path) -> None:
        """Save policy network and parameters to disk.

        Creates a directory with:
          - policy_net.pt: policy network state dict
          - params.json: hyperparameters and current epsilon

        Args:
            path: Directory to save to.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.policy_net.state_dict(), path / "policy_net.pt")

        params = {
            "state_dim": int(self.state_dim),
            "action_dim": int(self.action_dim),
            "gamma": float(self.gamma),
            "epsilon": float(self.epsilon),
            "epsilon_end": float(self.epsilon_end),
            "epsilon_decay": float(self.epsilon_decay),
            "lr": float(self.lr),
            "step_count": int(self._step_count),
        }
        with open(path / "params.json", "w") as f:
            json.dump(params, f, indent=2)

    def load(self, path: Path) -> None:
        """Load policy network and parameters from disk.

        Args:
            path: Directory to load from.
        """
        path = Path(path)

        self.policy_net.load_state_dict(
            torch.load(
                path / "policy_net.pt",
                map_location=self.device,
                weights_only=True,
            )
        )

        with open(path / "params.json", "r") as f:
            params = json.load(f)

        self.epsilon = params["epsilon"]
        self.gamma = params["gamma"]
        self.lr = params["lr"]
        self._step_count = params["step_count"]

    def get_info(self) -> dict:
        """Return current agent info for logging.

        Returns:
            Dict with epsilon and step count.
        """
        return {
            "epsilon": self.epsilon,
            "step_count": self._step_count,
        }

    def get_weights(self) -> dict:
        """Return a copy of the policy network's learned parameters."""
        return {k: v.clone() for k, v in self.policy_net.state_dict().items()}

    def set_weights(self, weights: dict) -> None:
        """Replace the policy network's parameters with the given weights."""
        self.policy_net.load_state_dict(weights)

    def mutate(self, noise_scale: float = 0.1) -> None:
        """Add random noise to the policy network's parameters (for evolution)."""
        with torch.no_grad():
            for param in self.policy_net.parameters():
                param.add_(torch.randn_like(param) * noise_scale)
