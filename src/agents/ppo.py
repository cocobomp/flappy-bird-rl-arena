"""Proximal Policy Optimization (PPO) agent with clipped surrogate objective."""

import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.base_agent import BaseAgent


class ActorCriticNetwork(nn.Module):
    """Shared backbone with actor (policy) and critic (value) heads.

    Architecture:
        Shared:  state_dim -> hidden_dims[0] -> ReLU
        Actor:   hidden_dims[0] -> hidden_dims[1] -> ReLU -> action_dim -> Softmax
        Critic:  hidden_dims[0] -> hidden_dims[1] -> ReLU -> 1

    Uses orthogonal initialization for improved training stability.
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

        # Shared backbone: state_dim -> hidden_dims[0] -> ReLU
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
        )

        # Actor head: hidden_dims[0] -> hidden_dims[1] -> ReLU -> action_dim -> Softmax
        self.actor = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim),
            nn.Softmax(dim=-1),
        )

        # Critic head: hidden_dims[0] -> hidden_dims[1] -> ReLU -> 1
        self.critic = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Apply orthogonal initialization to all linear layers."""
        for module_group in [self.shared, self.actor, self.critic]:
            for module in module_group:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                    nn.init.constant_(module.bias, 0.0)
        # Small gain for actor output layer (start near-uniform policy)
        actor_output = self.actor[-2]  # Linear before Softmax
        nn.init.orthogonal_(actor_output.weight, gain=0.01)
        # Small gain for critic output layer (start near-zero values)
        critic_output = self.critic[-1]  # Final Linear
        nn.init.orthogonal_(critic_output.weight, gain=1.0)

    def forward(self, x: torch.Tensor):
        """Forward pass returning action probabilities and state value.

        Args:
            x: State tensor of shape (batch, state_dim).

        Returns:
            Tuple of (action_probs, state_value) with shapes
            (batch, action_dim) and (batch, 1).
        """
        shared_out = self.shared(x)
        action_probs = self.actor(shared_out)
        state_value = self.critic(shared_out)
        return action_probs, state_value


class PPOAgent(BaseAgent):
    """PPO agent with clipped surrogate objective.

    Collects transitions into a rollout buffer, then performs multiple
    epochs of minibatch updates using the clipped PPO objective,
    a value function loss, and an entropy bonus.
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
        clip_eps: float = 0.2,
        value_clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        rollout_size: int = 128,
        n_epochs: int = 4,
        mini_batch_size: int = 32,
        gae_lambda: float = 0.95,
        max_grad_norm: float = 0.5,
    ):
        super().__init__(state_dim, action_dim)

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.clip_eps = clip_eps
        self.value_clip_eps = value_clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.rollout_size = rollout_size
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm
        self.lr = lr

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor-Critic network and optimizer
        self.ac_net = ActorCriticNetwork(state_dim, action_dim, hidden_dims).to(
            self.device
        )
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=lr)

        # Rollout buffer (lists that get cleared after each update)
        self._states: List[np.ndarray] = []
        self._actions: List[int] = []
        self._log_probs: List[float] = []
        self._rewards: List[float] = []
        self._dones: List[bool] = []
        self._values: List[float] = []

        # Step counter
        self._step_count = 0

        # Track whether the last select_action stored a pending transition
        self._pending = False

    def _get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Return action probabilities from the actor head (for display).

        Args:
            state: A single state observation.

        Returns:
            Numpy array of action probabilities, shape (action_dim,).
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.ac_net(state_t)
        return action_probs.cpu().numpy().squeeze(0)

    def get_activations(self, state: np.ndarray) -> list:
        """Capture intermediate activations from the shared backbone.

        Args:
            state: A single state observation.

        Returns:
            List of numpy arrays with activations at each layer.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        activations = [state]  # input layer
        with torch.no_grad():
            x = state_t
            for layer in self.ac_net.shared:
                x = layer(x)
                if isinstance(layer, nn.ReLU):
                    activations.append(x.cpu().numpy().squeeze(0))

            # Actor head output
            actor_out = x
            for layer in self.ac_net.actor:
                actor_out = layer(actor_out)
            activations.append(actor_out.cpu().numpy().squeeze(0))
        return activations

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select an action using the policy network.

        During training with high epsilon, may explore randomly.
        Stores the transition data (state, action, log_prob, value) in the
        rollout buffer for later PPO updates.

        Args:
            state: Current environment observation.
            training: If True, allows epsilon-greedy exploration.

        Returns:
            The chosen action index.
        """
        # Epsilon-greedy exploration (mainly useful early in training)
        if training and np.random.random() < self.epsilon:
            # Still need to compute value for the buffer
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_probs, value = self.ac_net(state_t)
            action = int(np.random.randint(self.action_dim))
            dist = torch.distributions.Categorical(action_probs)
            log_prob = dist.log_prob(torch.tensor(action)).item()

            if training:
                self._states.append(state.copy())
                self._actions.append(action)
                self._log_probs.append(log_prob)
                self._values.append(value.item())
                self._pending = True

            return action

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, value = self.ac_net(state_t)

        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        if training:
            self._states.append(state.copy())
            self._actions.append(action.item())
            self._log_probs.append(log_prob.item())
            self._values.append(value.item())
            self._pending = True

        return action.item()

    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float,
    ) -> tuple:
        """Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Array of rewards, shape (T,).
            values: Array of state values, shape (T,).
            dones: Array of done flags, shape (T,).
            next_value: Bootstrap value V(s_{T+1}).

        Returns:
            Tuple of (advantages, returns) as numpy arrays, each shape (T,).
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def _ppo_update(self, next_state: np.ndarray, done: bool) -> dict:
        """Perform the PPO update on the collected rollout buffer.

        Args:
            next_state: The state after the last transition (for bootstrapping).
            done: Whether the episode ended at the last transition.

        Returns:
            Dict with training metrics (policy_loss, value_loss, entropy, total_loss).
        """
        # Convert buffer to numpy arrays
        states = np.array(self._states, dtype=np.float32)
        actions = np.array(self._actions, dtype=np.int64)
        old_log_probs = np.array(self._log_probs, dtype=np.float32)
        rewards = np.array(self._rewards, dtype=np.float32)
        dones = np.array(self._dones, dtype=np.float32)
        values = np.array(self._values, dtype=np.float32)

        # Bootstrap value for last state
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, next_value = self.ac_net(next_state_t)
        next_value = next_value.item() * (1.0 - float(done))

        # Compute GAE advantages and returns
        advantages, returns = self._compute_gae(rewards, values, dones, next_value)

        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(self.device)
        old_values_t = torch.FloatTensor(values).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        total_samples = len(states)
        metrics = {}

        for _epoch in range(self.n_epochs):
            # Generate random permutation for minibatches
            indices = np.random.permutation(total_samples)

            for start in range(0, total_samples, self.mini_batch_size):
                end = min(start + self.mini_batch_size, total_samples)
                mb_indices = indices[start:end]

                mb_states = states_t[mb_indices]
                mb_actions = actions_t[mb_indices]
                mb_old_log_probs = old_log_probs_t[mb_indices]
                mb_old_values = old_values_t[mb_indices]
                mb_advantages = advantages_t[mb_indices]
                mb_returns = returns_t[mb_indices]

                # Per-minibatch advantage normalization
                if len(mb_advantages) > 1:
                    adv_std = mb_advantages.std()
                    if adv_std > 1e-8:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            adv_std + 1e-8
                        )

                # Forward pass
                action_probs, values_pred = self.ac_net(mb_states)
                values_pred = values_pred.squeeze(-1)

                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # Ratio for PPO clipping
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss with clipping (prevents large value updates)
                value_loss_unclipped = (values_pred - mb_returns) ** 2
                values_clipped = mb_old_values + torch.clamp(
                    values_pred - mb_old_values,
                    -self.value_clip_eps,
                    self.value_clip_eps,
                )
                value_loss_clipped = (values_clipped - mb_returns) ** 2
                value_loss = 0.5 * torch.max(
                    value_loss_unclipped, value_loss_clipped
                ).mean()

                # Total loss
                total_loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                # Gradient step
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.ac_net.parameters(), max_norm=self.max_grad_norm
                )
                self.optimizer.step()

                metrics = {
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy": entropy.item(),
                    "total_loss": total_loss.item(),
                }

        # Clear rollout buffer
        self._states.clear()
        self._actions.clear()
        self._log_probs.clear()
        self._rewards.clear()
        self._dones.clear()
        self._values.clear()
        self._pending = False

        return metrics

    def train_step(self, state, action, reward, next_state, done) -> dict:
        """Perform one training step: store reward/done, update if buffer full.

        The select_action method already stored (state, action, log_prob, value).
        This method completes the transition with (reward, done) and triggers
        a PPO update when the rollout buffer is full or the episode ends.

        Args:
            state: State at time t.
            action: Action taken at time t.
            reward: Reward received after taking action.
            next_state: State at time t+1.
            done: Whether the episode terminated.

        Returns:
            Dict with optional training metrics.
        """
        # If select_action was not called before (e.g. direct train_step usage),
        # we need to add the state/action/log_prob/value now
        if not self._pending:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_probs, value = self.ac_net(state_t)
            dist = torch.distributions.Categorical(action_probs)
            log_prob = dist.log_prob(torch.tensor(action)).item()

            self._states.append(np.array(state, dtype=np.float32))
            self._actions.append(action)
            self._log_probs.append(log_prob)
            self._values.append(value.item())

        self._pending = False

        # Store reward and done
        self._rewards.append(reward)
        self._dones.append(float(done))

        self._step_count += 1
        metrics = {}

        # Perform PPO update when buffer is full or episode ends
        buffer_len = len(self._rewards)
        if buffer_len >= self.rollout_size or done:
            metrics = self._ppo_update(next_state, done)

        # Decay epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay,
        )

        return metrics

    def save(self, path: Path) -> None:
        """Save the actor-critic network and parameters to disk.

        Creates a directory with:
          - ac_net.pt: actor-critic network state dict
          - params.json: hyperparameters and current epsilon

        Args:
            path: Directory to save to.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.ac_net.state_dict(), path / "ac_net.pt")

        params = {
            "state_dim": int(self.state_dim),
            "action_dim": int(self.action_dim),
            "gamma": float(self.gamma),
            "epsilon": float(self.epsilon),
            "epsilon_end": float(self.epsilon_end),
            "epsilon_decay": float(self.epsilon_decay),
            "clip_eps": float(self.clip_eps),
            "value_clip_eps": float(self.value_clip_eps),
            "value_coef": float(self.value_coef),
            "entropy_coef": float(self.entropy_coef),
            "rollout_size": int(self.rollout_size),
            "n_epochs": int(self.n_epochs),
            "mini_batch_size": int(self.mini_batch_size),
            "gae_lambda": float(self.gae_lambda),
            "max_grad_norm": float(self.max_grad_norm),
            "lr": float(self.lr),
            "step_count": int(self._step_count),
        }
        with open(path / "params.json", "w") as f:
            json.dump(params, f, indent=2)

    def load(self, path: Path) -> None:
        """Load the actor-critic network and parameters from disk.

        Args:
            path: Directory to load from.
        """
        path = Path(path)

        self.ac_net.load_state_dict(
            torch.load(
                path / "ac_net.pt", map_location=self.device, weights_only=True
            )
        )

        with open(path / "params.json", "r") as f:
            params = json.load(f)

        self.epsilon = params["epsilon"]
        self.gamma = params["gamma"]
        self.lr = params["lr"]
        self.clip_eps = params["clip_eps"]
        self.value_clip_eps = params.get("value_clip_eps", 0.2)
        self.value_coef = params["value_coef"]
        self.entropy_coef = params["entropy_coef"]
        self.rollout_size = params["rollout_size"]
        self.n_epochs = params["n_epochs"]
        self.mini_batch_size = params["mini_batch_size"]
        self.gae_lambda = params["gae_lambda"]
        self.max_grad_norm = params.get("max_grad_norm", 0.5)
        self._step_count = params["step_count"]

    def get_info(self) -> dict:
        """Return current agent info for logging.

        Returns:
            Dict with epsilon, step count, and PPO-specific parameters.
        """
        return {
            "epsilon": self.epsilon,
            "step_count": self._step_count,
            "clip_eps": self.clip_eps,
            "value_coef": self.value_coef,
            "entropy_coef": self.entropy_coef,
        }

    def get_weights(self) -> dict:
        """Return a copy of the actor-critic network parameters."""
        return {k: v.clone() for k, v in self.ac_net.state_dict().items()}

    def set_weights(self, weights: dict) -> None:
        """Replace the actor-critic network parameters with a copy."""
        self.ac_net.load_state_dict(weights)

    def mutate(self, noise_scale: float = 0.1) -> None:
        """Add random noise to the actor-critic parameters (for evolution)."""
        with torch.no_grad():
            for param in self.ac_net.parameters():
                param.add_(torch.randn_like(param) * noise_scale)
