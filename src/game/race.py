"""Race manager: orchestrates birds, agents, training, and rendering."""
from dataclasses import dataclass, field
import numpy as np

from src.agents import QLearningAgent, DQNAgent, DoubleDQNAgent, BaseAgent
from src.environments.rewards import BasicReward, DistanceReward, CenteredReward, RewardFunction
from src.game.engine import FlappyBirdEngine, Bird, SCREEN_WIDTH, GROUND_Y, PIPE_GAP
from src.game.ui import BIRD_COLORS, ALGO_DISPLAY, REWARD_DISPLAY

AGENT_MAP = {
    "q_learning": QLearningAgent,
    "dqn": DQNAgent,
    "double_dqn": DoubleDQNAgent,
}

REWARD_MAP = {
    "basic": BasicReward,
    "distance": DistanceReward,
    "centered": CenteredReward,
}

STATE_DIM = 4
ACTION_DIM = 2


@dataclass
class BirdEntry:
    """Links a bird, its RL agent, and its reward function."""
    algo: str
    reward: str
    color: tuple[int, int, int]
    state_dim: int = STATE_DIM
    agent: BaseAgent = field(init=False)
    reward_fn: RewardFunction = field(init=False)
    bird: Bird | None = field(default=None, init=False)
    best_score: int = 0
    prev_obs: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self):
        agent_cls = AGENT_MAP[self.algo]
        if self.algo == "q_learning":
            self.agent = agent_cls(
                state_dim=self.state_dim, action_dim=ACTION_DIM,
                n_bins=10, lr=0.1, gamma=0.99,
                epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.998,
            )
        else:
            self.agent = agent_cls(
                state_dim=self.state_dim, action_dim=ACTION_DIM,
                hidden_dims=[64, 64], lr=5e-4, gamma=0.99,
                epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.998,
                buffer_size=20000, batch_size=32, tau=0.005, train_every=4,
            )
        self.reward_fn = REWARD_MAP[self.reward]()

    @property
    def display_name(self) -> str:
        return f"{ALGO_DISPLAY[self.algo]} + {REWARD_DISPLAY[self.reward]}"


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

    def add_bird(self, algo: str, reward: str) -> BirdEntry:
        color = BIRD_COLORS[self._color_index % len(BIRD_COLORS)]
        self._color_index += 1
        entry = BirdEntry(algo=algo, reward=reward, color=color)
        bird = self.engine.add_bird(color=color)
        entry.bird = bird
        self.entries.append(entry)
        return entry

    def remove_bird(self, index: int):
        if 0 <= index < len(self.entries):
            bird_id = self.entries[index].bird.bird_id
            self.engine.remove_bird(bird_id)
            self.entries.pop(index)
            # Re-link birds after reindex
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

        # Collect actions from all alive birds
        actions = {}
        for entry in self.entries:
            if entry.bird.alive:
                obs = self.engine.get_observation(entry.bird)
                action = entry.agent.select_action(obs, training=True)
                actions[entry.bird.bird_id] = action
                entry.prev_obs = obs

        # Step engine
        obs_dict, _, done_dict, info = self.engine.step(actions)

        # Train each bird's agent
        for entry in self.entries:
            bid = entry.bird.bird_id
            if entry.prev_obs is not None:
                obs = obs_dict.get(bid, entry.prev_obs)
                terminated = done_dict.get(bid, False)
                # Compute custom reward
                raw_reward = 0.1 if not terminated else -1.0
                reward = entry.reward_fn.compute(obs, raw_reward, terminated, False)
                action = actions.get(bid, 0)
                entry.agent.train_step(entry.prev_obs, action, reward, obs, terminated)

        # Update best scores
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
                "best_score": entry.best_score,
            }
        return configs
