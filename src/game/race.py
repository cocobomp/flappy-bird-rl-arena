"""Race manager: orchestrates birds, agents, training, and rendering."""
from dataclasses import dataclass, field
import numpy as np

from src.agents import QLearningAgent, DQNAgent, DoubleDQNAgent, BaseAgent
from src.environments.rewards import (
    BasicReward, DistanceReward, CenteredReward, SmartReward, RewardFunction,
)
from src.game.engine import FlappyBirdEngine, Bird, SCREEN_WIDTH, GROUND_Y, PIPE_GAP
from src.game.ui import BIRD_COLORS, ALGO_DISPLAY, REWARD_DISPLAY, STRATEGY_DISPLAY
from src.game.strategies import STRATEGY_MAP, ExplorationStrategy

AGENT_MAP = {
    "q_learning": QLearningAgent,
    "dqn": DQNAgent,
    "double_dqn": DoubleDQNAgent,
}

REWARD_MAP = {
    "basic": BasicReward,
    "distance": DistanceReward,
    "centered": CenteredReward,
    "smart": SmartReward,
}

STATE_DIM = 4
ACTION_DIM = 2


@dataclass
class BirdEntry:
    """Links a bird, its RL agent, reward function, and exploration strategy."""
    algo: str
    reward: str
    strategy_name: str
    color: tuple[int, int, int]
    state_dim: int = STATE_DIM
    epsilon_start: float = 0.8
    lr: float = 5e-4
    epsilon_decay: float = 0.998
    death_penalty: float = 5.0
    pipe_bonus: float = 10.0
    agent: BaseAgent = field(init=False)
    reward_fn: RewardFunction = field(init=False)
    strategy: ExplorationStrategy = field(init=False)
    bird: Bird | None = field(default=None, init=False)
    best_score: int = 0
    total_pipes: int = 0
    prev_obs: np.ndarray | None = field(default=None, init=False)
    last_exploring: bool = field(default=True, init=False)

    def __post_init__(self):
        agent_cls = AGENT_MAP[self.algo]
        if self.algo == "q_learning":
            self.agent = agent_cls(
                state_dim=self.state_dim, action_dim=ACTION_DIM,
                n_bins=10, lr=self.lr, gamma=0.99,
                epsilon_start=self.epsilon_start, epsilon_end=0.01,
                epsilon_decay=self.epsilon_decay,
            )
        else:
            self.agent = agent_cls(
                state_dim=self.state_dim, action_dim=ACTION_DIM,
                hidden_dims=[64, 64], lr=self.lr, gamma=0.99,
                epsilon_start=self.epsilon_start, epsilon_end=0.01,
                epsilon_decay=self.epsilon_decay,
                buffer_size=20000, batch_size=32, tau=0.005, train_every=4,
            )
        if self.reward == "smart":
            self.reward_fn = SmartReward(death_penalty=self.death_penalty)
        else:
            self.reward_fn = REWARD_MAP[self.reward]()
        strategy_cls = STRATEGY_MAP[self.strategy_name]
        self.strategy = strategy_cls()

    @property
    def display_name(self) -> str:
        return f"{ALGO_DISPLAY[self.algo]} {REWARD_DISPLAY[self.reward]} [{STRATEGY_DISPLAY[self.strategy_name]}]"


class RaceManager:
    """Manages the multi-bird race loop."""

    def __init__(self, render: bool = True):
        self.engine = FlappyBirdEngine()
        self.entries: list[BirdEntry] = []
        self.render_enabled = render
        self.renderer = None
        self.speed = 1
        self.paused = False
        self._color_index = 0

    def add_bird(self, algo: str, reward: str, strategy: str = "guided",
                 epsilon_start: float = 0.8, lr: float = 5e-4,
                 epsilon_decay: float = 0.998, death_penalty: float = 5.0,
                 pipe_bonus: float = 10.0) -> BirdEntry:
        color = BIRD_COLORS[self._color_index % len(BIRD_COLORS)]
        self._color_index += 1
        entry = BirdEntry(
            algo=algo, reward=reward, strategy_name=strategy, color=color,
            epsilon_start=epsilon_start, lr=lr,
            epsilon_decay=epsilon_decay, death_penalty=death_penalty,
            pipe_bonus=pipe_bonus,
        )
        bird = self.engine.add_bird(color=color)
        entry.bird = bird
        self.entries.append(entry)
        return entry

    def remove_bird(self, index: int):
        if 0 <= index < len(self.entries):
            bird_id = self.entries[index].bird.bird_id
            self.engine.remove_bird(bird_id)
            self.entries.pop(index)
            for i, entry in enumerate(self.entries):
                entry.bird = self.engine.birds[i]

    def reset_round(self):
        self.engine.reset()
        for entry in self.entries:
            entry.prev_obs = self.engine.get_observation(entry.bird)

    def step_frame(self) -> bool:
        """Run one frame of the game. Returns True if round is over."""
        if not self.entries:
            return True

        actions = {}
        for entry in self.entries:
            if entry.bird.alive:
                obs = self.engine.get_observation(entry.bird)
                # Exploration with strategy vs exploitation with learned policy
                exploring = np.random.random() < entry.agent.epsilon
                if exploring:
                    action = entry.strategy.explore(obs)
                else:
                    action = entry.agent.select_action(obs, training=False)
                entry.last_exploring = exploring
                actions[entry.bird.bird_id] = action
                entry.prev_obs = obs

        # Track scores before step to detect pipe passing
        prev_scores = {e.bird.bird_id: e.bird.score for e in self.entries}

        obs_dict, _, done_dict, info = self.engine.step(actions)

        for entry in self.entries:
            bid = entry.bird.bird_id
            if entry.prev_obs is not None:
                obs = obs_dict.get(bid, entry.prev_obs)
                terminated = done_dict.get(bid, False)
                raw_reward = 0.1 if not terminated else -1.0
                reward = entry.reward_fn.compute(obs, raw_reward, terminated, False)
                # Pipe passing bonus!
                pipes_passed = entry.bird.score - prev_scores.get(bid, 0)
                if pipes_passed > 0:
                    reward += entry.pipe_bonus * pipes_passed
                    entry.total_pipes += pipes_passed
                action = actions.get(bid, 0)
                entry.agent.train_step(entry.prev_obs, action, reward, obs, terminated)

        for entry in self.entries:
            if entry.bird.score > entry.best_score:
                entry.best_score = entry.bird.score

        return info["round_over"]

    def get_bird_configs(self) -> dict:
        configs = {}
        for entry in self.entries:
            configs[entry.bird.bird_id] = {
                "name": entry.display_name,
                "algo": ALGO_DISPLAY[entry.algo],
                "reward": REWARD_DISPLAY[entry.reward],
                "strategy": STRATEGY_DISPLAY[entry.strategy_name],
                "best_score": entry.best_score,
                "total_pipes": entry.total_pipes,
                "epsilon": round(entry.agent.epsilon, 3),
                "exploring": entry.last_exploring,
            }
        return configs
