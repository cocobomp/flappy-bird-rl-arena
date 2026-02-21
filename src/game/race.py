"""Race manager: orchestrates birds, agents, training, and rendering."""
from dataclasses import dataclass, field
import numpy as np

from src.agents import (
    QLearningAgent, DQNAgent, DoubleDQNAgent, DuelingDQNAgent,
    ReinforceAgent, PPOAgent, BaseAgent,
    RandomForestAgent, GradientBoostAgent, KNNAgent, SVMAgent,
)
from src.environments.rewards import (
    BasicReward, DistanceReward, CenteredReward, SmartReward, RewardFunction,
)
from src.game.engine import FlappyBirdEngine, Bird, SCREEN_WIDTH, GROUND_Y, PIPE_GAP
from src.game.ui import BIRD_COLORS, ALGO_DISPLAY, REWARD_DISPLAY, STRATEGY_DISPLAY
from src.game.strategies import STRATEGY_MAP, STRATEGY_OPTIONS, ExplorationStrategy

AGENT_MAP = {
    "q_learning": QLearningAgent,
    "dqn": DQNAgent,
    "double_dqn": DoubleDQNAgent,
    "dueling_dqn": DuelingDQNAgent,
    "reinforce": ReinforceAgent,
    "ppo": PPOAgent,
    "random_forest": RandomForestAgent,
    "gradient_boost": GradientBoostAgent,
    "knn": KNNAgent,
    "svm": SVMAgent,
}

REWARD_MAP = {
    "basic": BasicReward,
    "distance": DistanceReward,
    "centered": CenteredReward,
    "smart": SmartReward,
}

STATE_DIM = 5
ACTION_DIM = 2


@dataclass
class BirdEntry:
    """Links a bird, its RL agent, reward function, and exploration strategy."""
    algo: str
    reward: str
    strategy_name: str
    color: tuple[int, int, int]
    state_dim: int = STATE_DIM
    epsilon_start: float = 1.0
    lr: float = 3e-4
    epsilon_decay: float = 1.0
    episode_epsilon_decay: float = 0.98
    death_penalty: float = 20.0
    pipe_bonus: float = 15.0
    alive_reward: float = 0.02
    flap_threshold: float = 0.04
    strategy_noise: float = 0.02
    safety_filter: bool = True
    agent: BaseAgent = field(init=False)
    reward_fn: RewardFunction = field(init=False)
    strategy: ExplorationStrategy = field(init=False)
    bird: Bird | None = field(default=None, init=False)
    best_score: int = 0
    total_pipes: int = 0
    generation: int = 0
    parent_color: tuple[int, int, int] | None = field(default=None, init=False)
    prev_obs: np.ndarray | None = field(default=None, init=False)
    last_exploring: bool = field(default=True, init=False)
    last_q_values: np.ndarray | None = field(default=None, init=False)
    last_activations: list | None = field(default=None, init=False)
    last_viz_data: dict | None = field(default=None, init=False)

    def __post_init__(self):
        agent_cls = AGENT_MAP[self.algo]
        common = dict(
            state_dim=self.state_dim, action_dim=ACTION_DIM,
            epsilon_start=self.epsilon_start, epsilon_end=0.01,
            epsilon_decay=self.epsilon_decay,
        )
        if self.algo in ("random_forest", "gradient_boost", "knn", "svm"):
            # retrain_every=100: ~1 episode, faster adaptation early on
            # min_samples=30: start learning sooner (was 50)
            self.agent = agent_cls(
                **common, max_samples=5000, retrain_every=100, min_samples=30,
            )
        elif self.algo == "q_learning":
            self.agent = agent_cls(
                **common, lr=self.lr, gamma=0.95, n_bins=10,
            )
        elif self.algo == "reinforce":
            # Larger network [128,64] for more capacity; EMA baseline
            # with alpha=0.1 smooths across short episodes (50-200 frames)
            self.agent = agent_cls(
                **common, lr=self.lr, gamma=0.95, hidden_dims=[128, 64],
                baseline_ema_alpha=0.1,
            )
        elif self.algo == "ppo":
            # rollout_size=64: better for short episodes (50-200 frames),
            #   ensures more regular updates for longer-surviving birds
            # n_epochs=4: good data reuse for small rollouts
            # clip_eps=0.2: standard, works well
            # entropy_coef=0.02: slightly higher for better exploration
            #   in short episodes where the policy can collapse quickly
            # gae_lambda=0.95: standard, appropriate for gamma=0.95
            self.agent = agent_cls(
                **common, lr=self.lr, gamma=0.95, hidden_dims=[128, 64],
                rollout_size=64, n_epochs=4, clip_eps=0.2,
                entropy_coef=0.02, gae_lambda=0.95,
            )
        else:
            # DQN, Double DQN, Dueling DQN
            # Optimized via hyperparameter sweep:
            # hidden_dims=[128,64]: more capacity for better generalization
            # train_intensity=4: more gradient steps per training trigger
            # tau=0.005: stable target network updates (tau=0.01 hurts)
            # batch_size=64: good balance of gradient noise/stability
            self.agent = agent_cls(
                **common, lr=self.lr, gamma=0.95,
                hidden_dims=[128, 64],
                buffer_size=10000, batch_size=64, tau=0.005,
                train_every=1, train_intensity=4,
            )
        if self.reward == "smart":
            self.reward_fn = SmartReward(death_penalty=self.death_penalty)
        else:
            self.reward_fn = REWARD_MAP[self.reward]()
        self._create_strategy()

    def _create_strategy(self):
        """Build the exploration strategy with current params."""
        cls = STRATEGY_MAP[self.strategy_name]
        self.strategy = cls(
            noise=self.strategy_noise,
            threshold=self.flap_threshold,
            flap_prob=0.12,
        )

    @property
    def display_name(self) -> str:
        return f"{ALGO_DISPLAY[self.algo]} {REWARD_DISPLAY[self.reward]} [{STRATEGY_DISPLAY[self.strategy_name]}]"

    @property
    def q_display(self) -> str:
        if self.last_q_values is None:
            return ""
        q = self.last_q_values
        best = "noop" if q[0] >= q[1] else "FLAP"
        return f"Q:{q[0]:+.1f}|{q[1]:+.1f}->{best}"


class RaceManager:
    """Manages the multi-bird race loop."""

    def __init__(self, render: bool = True):
        self.engine = FlappyBirdEngine()
        self.entries: list[BirdEntry] = []
        self.render_enabled = render
        self.speed = 1
        self.paused = False
        self.evolution_enabled = False
        self.mutation_scale = 0.02
        self._color_index = 0
        self.round_scores: list[int] = []
        self.ghost_trail: list[float] = []
        self.best_ever_score: int = 0

    def add_bird(self, algo: str, reward: str, strategy: str = "guided",
                 epsilon_start: float = 1.0, lr: float = 3e-4,
                 epsilon_decay: float = 1.0,
                 episode_epsilon_decay: float = 0.98,
                 death_penalty: float = 20.0,
                 pipe_bonus: float = 15.0, alive_reward: float = 0.02,
                 flap_threshold: float = 0.04, strategy_noise: float = 0.10,
                 safety_filter: bool = True,
                 ) -> BirdEntry:
        color = BIRD_COLORS[self._color_index % len(BIRD_COLORS)]
        self._color_index += 1
        entry = BirdEntry(
            algo=algo, reward=reward, strategy_name=strategy, color=color,
            epsilon_start=epsilon_start, lr=lr,
            epsilon_decay=epsilon_decay,
            episode_epsilon_decay=episode_epsilon_decay,
            death_penalty=death_penalty,
            pipe_bonus=pipe_bonus, alive_reward=alive_reward,
            flap_threshold=flap_threshold, strategy_noise=strategy_noise,
            safety_filter=safety_filter,
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

    def cycle_strategy(self, index: int):
        if 0 <= index < len(self.entries):
            entry = self.entries[index]
            cur = STRATEGY_OPTIONS.index(entry.strategy_name)
            entry.strategy_name = STRATEGY_OPTIONS[(cur + 1) % len(STRATEGY_OPTIONS)]
            entry._create_strategy()

    def halve_epsilon(self, index: int):
        if 0 <= index < len(self.entries):
            self.entries[index].agent.epsilon = max(
                0.01, self.entries[index].agent.epsilon / 2)

    def boost_all(self):
        for entry in self.entries:
            entry.agent.epsilon = 0.05

    def evolve(self):
        """All birds copy weights from the best bird (+ mutation)."""
        if len(self.entries) < 2:
            return
        # Best = highest score, tiebreak by steps alive
        best = max(self.entries,
                   key=lambda e: (e.bird.score, e.bird.steps_alive))
        weights = best.agent.get_weights()
        for entry in self.entries:
            if entry is best:
                continue
            if type(entry.agent) is not type(best.agent):
                continue
            entry.agent.set_weights(weights)
            entry.agent.mutate(noise_scale=self.mutation_scale)
            entry.agent.epsilon = best.agent.epsilon
            entry.generation += 1
            entry.parent_color = best.color

    def reset_round(self):
        if self.evolution_enabled:
            self.evolve()
        if self.entries:
            best = max(self.entries, key=lambda e: (e.bird.score, e.bird.steps_alive))
            self.round_scores.append(best.bird.score)
            if len(self.round_scores) > 100:
                self.round_scores.pop(0)
            if best.bird.score > self.best_ever_score:
                self.best_ever_score = best.bird.score
                self.ghost_trail = list(best.bird.trail)
        # Per-episode epsilon decay: decay once per round (not per frame).
        # This gives the agent more time under guided exploration early on,
        # then smoothly transitions to exploitation as training progresses.
        for entry in self.entries:
            entry.agent.epsilon = max(
                getattr(entry.agent, 'epsilon_end', 0.01),
                entry.agent.epsilon * entry.episode_epsilon_decay,
            )
        self.engine.reset()
        for entry in self.entries:
            entry.prev_obs = self.engine.get_observation(entry.bird)

    def step_frame(self) -> bool:
        if not self.entries:
            return True

        actions = {}
        for entry in self.entries:
            if entry.bird.alive:
                obs = self.engine.get_observation(entry.bird)
                # Q-values + activations for display
                if hasattr(entry.agent, '_get_q_values'):
                    entry.last_q_values = entry.agent._get_q_values(obs)
                    if hasattr(entry.agent, 'get_activations'):
                        entry.last_activations = entry.agent.get_activations(obs)
                # Feature importances for sklearn agents
                if hasattr(entry.agent, 'get_feature_importances'):
                    importances = entry.agent.get_feature_importances()
                    entry.last_viz_data = {
                        "type": entry.algo,
                        "importances": importances,
                        "confidence": entry.last_q_values,
                        "trained": getattr(entry.agent, '_trained', False),
                        "samples": len(getattr(entry.agent, '_buffer', [])),
                    }
                elif hasattr(entry.agent, 'q_table') and hasattr(entry.agent, '_discretize'):
                    key = entry.agent._discretize(obs)
                    entry.last_q_values = entry.agent.q_table[key].copy()
                # Exploration with strategy vs exploitation with learned policy
                exploring = np.random.random() < entry.agent.epsilon
                if exploring:
                    action = entry.strategy.explore(obs)
                else:
                    action = entry.agent.select_action(obs, training=False)
                    # Safety filter: override obviously dangerous agent actions.
                    # If agent says "flap" but bird is well above the gap,
                    # or says "no flap" but bird is falling far below the gap,
                    # override with the guided strategy's recommendation.
                    if entry.safety_filter:
                        delta_y = obs[0]   # positive = below gap
                        velocity = obs[1]  # negative = going up
                        # Bird well above gap and rising -> flap is dangerous
                        if action == 1 and delta_y < -0.10 and velocity < 0:
                            action = 0
                        # Bird far below gap and falling fast -> must flap
                        elif action == 0 and delta_y > 0.12 and velocity > 0.3:
                            action = 1
                entry.last_exploring = exploring
                actions[entry.bird.bird_id] = action
                entry.prev_obs = obs

        prev_scores = {e.bird.bird_id: e.bird.score for e in self.entries}

        obs_dict, _, done_dict, info = self.engine.step(actions)

        for entry in self.entries:
            bid = entry.bird.bird_id
            if entry.prev_obs is not None:
                obs = obs_dict.get(bid, entry.prev_obs)
                terminated = done_dict.get(bid, False)
                raw_reward = 0.1 if not terminated else -1.0
                reward = entry.reward_fn.compute(obs, raw_reward, terminated, False)
                # Configurable bonuses on top of reward function
                if not terminated:
                    reward += entry.alive_reward
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
        for i, entry in enumerate(self.entries):
            configs[entry.bird.bird_id] = {
                "name": entry.display_name,
                "algo": ALGO_DISPLAY[entry.algo],
                "reward": REWARD_DISPLAY[entry.reward],
                "strategy": STRATEGY_DISPLAY[entry.strategy_name],
                "best_score": entry.best_score,
                "total_pipes": entry.total_pipes,
                "epsilon": round(entry.agent.epsilon, 3),
                "exploring": entry.last_exploring,
                "q_display": entry.q_display,
                "generation": entry.generation,
                "parent_color": entry.parent_color,
                "activations": entry.last_activations,
                "viz_data": entry.last_viz_data,
                "index": i,
            }
        configs["_round_scores"] = self.round_scores
        configs["_ghost_trail"] = self.ghost_trail
        configs["_best_ever"] = self.best_ever_score
        return configs
