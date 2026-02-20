"""RL agents for Flappy Bird."""

from src.agents.base_agent import BaseAgent
from src.agents.q_learning import QLearningAgent
from src.agents.dqn import DQNAgent
from src.agents.double_dqn import DoubleDQNAgent
from src.agents.dueling_dqn import DuelingDQNAgent
from src.agents.reinforce import ReinforceAgent
from src.agents.ppo import PPOAgent
from src.agents.sklearn_agent import (
    RandomForestAgent, GradientBoostAgent, KNNAgent, SVMAgent,
)

__all__ = [
    "BaseAgent",
    "QLearningAgent",
    "DQNAgent",
    "DoubleDQNAgent",
    "DuelingDQNAgent",
    "ReinforceAgent",
    "PPOAgent",
    "RandomForestAgent",
    "GradientBoostAgent",
    "KNNAgent",
    "SVMAgent",
]
