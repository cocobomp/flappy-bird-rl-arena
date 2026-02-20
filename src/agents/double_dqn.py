"""Double DQN agent -- reduces overestimation bias of vanilla DQN."""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from src.agents.dqn import DQNAgent


class DoubleDQNAgent(DQNAgent):
    """Double DQN: uses online net to SELECT actions, target net to EVALUATE.

    This decoupling reduces the overestimation bias inherent in vanilla DQN,
    where the same network both selects and evaluates the greedy action.

    Inherits everything from DQNAgent except the loss computation.
    """

    def _compute_loss(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> Tuple[torch.Tensor, float]:
        """Compute Double DQN loss.

        Key difference from DQN:
          - Online network SELECTS best next action: a* = argmax_a Q_online(s', a)
          - Target network EVALUATES that action: Q_target(s', a*)

        Args:
            states, actions, rewards, next_states, dones: Batch arrays.

        Returns:
            Tuple of (loss tensor, mean Q-value float).
        """
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values for taken actions
        q_values = self.q_net(states_t)
        q_taken = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN target:
        # 1. Online network selects best actions for next states
        # 2. Target network evaluates those actions
        with torch.no_grad():
            best_actions = self.q_net(next_states_t).argmax(1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, best_actions).squeeze(1)
            target = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = nn.functional.mse_loss(q_taken, target)
        q_mean = q_values.detach().mean().item()

        return loss, q_mean
