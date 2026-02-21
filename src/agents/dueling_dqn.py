"""Dueling DQN agent -- separates Q-value into Value and Advantage streams."""

from typing import List

import numpy as np
import torch
import torch.nn as nn

from src.agents.dqn import DQNAgent, _init_weights


class DuelingQNetwork(nn.Module):
    """Dueling network architecture: shared features split into V(s) and A(s,a).

    Architecture:
        input -> shared_layer -> ReLU -> {
            value_stream:     hidden -> ReLU -> 1
            advantage_stream: hidden -> ReLU -> action_dim
        }

    Output: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))

    This decomposition helps the agent learn which states are valuable
    without needing to learn the effect of every action in every state.
    Especially useful for problems like Flappy Bird where sometimes
    the choice of action matters little (bird is far from pipes).
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

        if len(hidden_dims) < 2:
            raise ValueError(
                "Dueling DQN requires at least 2 hidden_dims: "
                "[shared_dim, stream_dim, ...]. Got: {}".format(hidden_dims)
            )

        self.action_dim = action_dim

        # Shared feature layers: state_dim -> hidden_dims[0] -> ReLU
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
        )

        # Value stream: hidden_dims[0] -> hidden_dims[1] -> ReLU -> 1
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )

        # Advantage stream: hidden_dims[0] -> hidden_dims[1] -> ReLU -> action_dim
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim),
        )

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values using dueling decomposition.

        Q(s,a) = V(s) + A(s,a) - mean_a(A(s,:))
        """
        features = self.shared(x)
        value = self.value_stream(features)            # (batch, 1)
        advantage = self.advantage_stream(features)    # (batch, action_dim)
        # Combine: Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class DuelingDQNAgent(DQNAgent):
    """Dueling DQN: overrides network architecture only.

    Replaces the standard QNetwork with a DuelingQNetwork that decomposes
    Q-values into state Value V(s) and action Advantage A(s,a). All
    training logic (replay buffer, epsilon-greedy, target updates) is
    inherited unchanged from DQNAgent.
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
        # Let DQNAgent.__init__ set up everything (including default QNetwork)
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            lr=lr,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=tau,
            train_every=train_every,
            train_intensity=train_intensity,
            use_per=use_per,
            per_alpha=per_alpha,
            per_beta_start=per_beta_start,
            per_beta_frames=per_beta_frames,
            lr_schedule=lr_schedule,
            lr_schedule_steps=lr_schedule_steps,
        )

        if hidden_dims is None:
            hidden_dims = [64, 32]

        # Replace the standard QNetwork with a DuelingQNetwork
        self.q_net = DuelingQNetwork(state_dim, action_dim, hidden_dims).to(
            self.device
        )
        self.target_net = DuelingQNetwork(state_dim, action_dim, hidden_dims).to(
            self.device
        )
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # Re-create optimizer for the new network parameters
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    def get_activations(self, state: np.ndarray) -> list[np.ndarray]:
        """Get activations from the dueling network for visualization.

        Returns activations from:
            1. Input (raw state)
            2. Shared layer (after ReLU)
            3. Output (Q-values from V + A - mean(A))
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        activations = [state]  # input layer

        with torch.no_grad():
            # Shared layers
            x = state_t
            for layer in self.q_net.shared:
                x = layer(x)
                if isinstance(layer, nn.ReLU):
                    activations.append(x.cpu().numpy().squeeze(0))

            # Compute full Q-values as output
            q_values = self.q_net(state_t)
            activations.append(q_values.cpu().numpy().squeeze(0))

        return activations
