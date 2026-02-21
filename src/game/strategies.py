"""Exploration strategies for the multi-bird RL race.

Physics recap (why this matters):
    flap = sets vel_y to -9 (strong upward impulse)
    gravity = +1 per frame
    One flap lifts the bird ~45 pixels before it starts falling.
    Pipe gap = 100 pixels.

So flapping is a BIG action. Smart strategies must account for velocity
to avoid constant overshooting (which makes birds look "dumb").

Observation format (8 relative features):
  [delta_y1, velocity, dist_pipe1, delta_y2, dist_pipe2,
   gap_position, proximity_danger, vertical_speed_direction]
  obs[0] delta_y1:   (bird_y - gap_center1) / SCREEN_HEIGHT  (positive=below gap)
  obs[1] velocity:   vel_y / MAX_VEL_Y  (negative=up, positive=down)
  obs[2] dist_pipe1: distance to next pipe / SCREEN_WIDTH
  obs[3] delta_y2:   (bird_y - gap_center2) / SCREEN_HEIGHT  (positive=below gap)
  obs[4] dist_pipe2: distance to second pipe / SCREEN_WIDTH
  obs[5] gap_position: -1 above gap, 0 inside gap, +1 below gap
  obs[6] proximity_danger: urgency signal when close to pipe (< 0.2)
  obs[7] vertical_speed_direction: sign(vel) * vel^2
"""

from abc import ABC, abstractmethod

import numpy as np


class ExplorationStrategy(ABC):
    """Base class for exploration strategies."""

    name: str = "?"

    @abstractmethod
    def explore(self, obs: np.ndarray) -> int:
        pass


class RandomStrategy(ExplorationStrategy):
    """Pure random 50/50.  Terrible for Flappy Bird — demo only."""

    name = "Random"

    def __init__(self, **_kwargs):
        pass

    def explore(self, obs: np.ndarray) -> int:
        # Safety: never flap if way above gap (near ceiling)
        if obs[0] < -0.25:
            return 0
        # Safety: never flap if above gap top
        if obs[0] < -0.08:
            return 0
        # Safety: must flap if far below gap and falling
        if obs[0] > 0.30 and obs[1] > 0.2:
            return 1
        return int(np.random.randint(2))


class GravityAwareStrategy(ExplorationStrategy):
    """Biased random with configurable flap probability.
    Default 12% compensates for flap(-9) >> gravity(+1)."""

    name = "Gravity"

    def __init__(self, flap_prob: float = 0.12, **_kwargs):
        self.flap_prob = flap_prob

    def explore(self, obs: np.ndarray) -> int:
        # Safety: never flap if way above gap (near ceiling)
        if obs[0] < -0.25:
            return 0
        # Safety: never flap if above gap top
        if obs[0] < -0.08:
            return 0
        # Safety: must flap if far below gap and falling
        if obs[0] > 0.30 and obs[1] > 0.2:
            return 1
        return 1 if np.random.random() < self.flap_prob else 0


class HeuristicStrategy(ExplorationStrategy):
    """PD-controller heuristic: accounts for POSITION and VELOCITY.

    urgency = (how far below gap) - damping * (how fast going up)

    Only flaps when urgency exceeds threshold. This prevents the
    constant overshoot that makes birds look stupid:
    - If below gap but already rising fast → DON'T flap (momentum is enough)
    - If below gap and falling → flap (need correction)
    - If above gap → never flap (let gravity bring you back)

    One flap = 45px rise. With threshold ~0.04 (~20px), the bird
    oscillates smoothly within the 100px gap.
    """

    name = "Heuristic"

    def __init__(self, noise: float = 0.10, threshold: float = 0.04, **_kwargs):
        self.noise = noise
        self.threshold = threshold

    def explore(self, obs: np.ndarray) -> int:
        diff = obs[0]         # delta_y1: positive = below gap
        velocity = obs[1]     # negative = going up

        # Hard rule: NEVER flap if above the top of the gap
        if diff < -0.08:
            return 0

        upward = max(0.0, -velocity)      # how fast going up (always >= 0)
        urgency = diff - 0.5 * upward     # damped position error

        action = 1 if urgency > self.threshold else 0

        if np.random.random() < self.noise:
            return 1 - action
        return action


class GuidedStrategy(ExplorationStrategy):
    """Always-on PD-controller heuristic with adaptive noise.

    Uses the velocity-aware heuristic at ALL distances (not just near pipes).
    Noise decays based on a performance score (pipes passed), so early
    exploration is high but tightens as the agent improves.

    Near-pipe precision: reduces noise and tightens threshold when close to
    a pipe, and anticipates the second pipe when about to pass the first.
    """

    name = "Guided"

    def __init__(self, flap_prob: float = 0.12, noise: float = 0.02,
                 threshold: float = 0.04, noise_decay: float = 0.995,
                 min_noise: float = 0.005, **_kwargs):
        self._base_noise = noise
        self._current_noise = noise
        self._noise_decay = noise_decay
        self._min_noise = min_noise
        self._base_threshold = threshold
        self.heuristic = HeuristicStrategy(noise=noise, threshold=threshold)

    def explore(self, obs: np.ndarray) -> int:
        dist_pipe1 = obs[2]  # distance to next pipe

        # Near-pipe precision: reduce noise when close to pipe
        if dist_pipe1 < 0.15:
            self.heuristic.noise = self._min_noise
        else:
            self.heuristic.noise = self._current_noise

        # Adaptive threshold: tighter when close, looser when far
        self.heuristic.threshold = self._base_threshold * (0.5 + dist_pipe1)

        # Second-pipe anticipation: blend delta_y2 when about to pass pipe1
        if dist_pipe1 < 0.08:
            diff = 0.7 * obs[0] + 0.3 * obs[3]
        else:
            diff = obs[0]

        # Override obs[0] for the heuristic with the blended diff
        modified_obs = obs.copy()
        modified_obs[0] = diff
        return self.heuristic.explore(modified_obs)

    def on_episode_end(self, score: int):
        """Decay noise after each episode. Higher scores decay faster."""
        decay = self._noise_decay ** max(1, score)
        self._current_noise = max(self._min_noise, self._current_noise * decay)


class BoltzmannStrategy(ExplorationStrategy):
    """Temperature-based action selection using Q-values.

    Instead of epsilon-greedy (random with probability epsilon), Boltzmann
    exploration selects actions proportional to exp(Q/temperature).
    High temperature -> uniform random; low temperature -> greedy.

    Designed for DQN-family agents that expose Q-values.
    Falls back to the PD-controller heuristic when no Q-values are available.
    """

    name = "Boltzmann"

    def __init__(self, temperature: float = 1.0, min_temperature: float = 0.1,
                 temp_decay: float = 0.999, threshold: float = 0.04,
                 noise: float = 0.02, **_kwargs):
        self.temperature = temperature
        self.min_temperature = min_temperature
        self.temp_decay = temp_decay
        self._fallback = HeuristicStrategy(noise=noise, threshold=threshold)
        self._q_values: np.ndarray | None = None

    def set_q_values(self, q_values: np.ndarray):
        """Provide Q-values for the current state (call before explore)."""
        self._q_values = q_values

    def explore(self, obs: np.ndarray) -> int:
        # Hard rule: NEVER flap if above the top of the gap
        if obs[0] < -0.08:
            self._q_values = None
            return 0

        if self._q_values is None:
            return self._fallback.explore(obs)
        q = self._q_values
        # Numerically stable softmax
        q_shifted = q - np.max(q)
        probs = np.exp(q_shifted / max(self.temperature, 1e-8))
        probs = probs / (probs.sum() + 1e-8)
        action = int(np.random.choice(len(probs), p=probs))
        self._q_values = None  # reset for next call
        return action

    def on_episode_end(self, score: int):
        """Anneal temperature after each episode."""
        self.temperature = max(self.min_temperature,
                               self.temperature * self.temp_decay)


def cosine_epsilon_schedule(episode: int, total_episodes: int,
                            epsilon_start: float = 1.0,
                            epsilon_end: float = 0.01) -> float:
    """Cosine annealing schedule for epsilon.

    Smoother than exponential decay — avoids both the sharp early drop
    and the long flat tail. Returns epsilon in [epsilon_end, epsilon_start].

    Args:
        episode: Current episode number (0-indexed).
        total_episodes: Total number of training episodes.
        epsilon_start: Initial epsilon value.
        epsilon_end: Minimum epsilon value.

    Returns:
        The epsilon value for the given episode.
    """
    if total_episodes <= 1:
        return epsilon_end
    progress = min(1.0, episode / (total_episodes - 1))
    cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
    return epsilon_end + (epsilon_start - epsilon_end) * cosine_decay


STRATEGY_MAP = {
    "random": RandomStrategy,
    "gravity": GravityAwareStrategy,
    "heuristic": HeuristicStrategy,
    "guided": GuidedStrategy,
    "boltzmann": BoltzmannStrategy,
}

STRATEGY_OPTIONS = list(STRATEGY_MAP.keys())
