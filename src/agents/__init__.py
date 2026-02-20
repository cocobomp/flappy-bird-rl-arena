"""RL agents for Flappy Bird."""

from src.agents.base_agent import BaseAgent
from src.agents.q_learning import QLearningAgent
from src.agents.dqn import DQNAgent
from src.agents.double_dqn import DoubleDQNAgent

__all__ = [
    "BaseAgent",
    "QLearningAgent",
    "DQNAgent",
    "DoubleDQNAgent",
]
