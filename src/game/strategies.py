"""Exploration strategies for the multi-bird RL race.

During epsilon-greedy training, the standard approach picks a random action
with probability epsilon. In Flappy Bird, this is catastrophic:

    flap force = -9 (strong upward)
    gravity    = +1 (weak downward)

With 50/50 random, the bird averages velocity ~ -4 per frame → flies into
the ceiling every time.  These strategies replace the random action with
smarter exploration so birds actually survive long enough to learn.
"""

from abc import ABC, abstractmethod

import numpy as np


class ExplorationStrategy(ABC):
    """Base class for exploration strategies."""

    name: str = "?"

    @abstractmethod
    def explore(self, obs: np.ndarray) -> int:
        """Choose an exploration action given the current observation.

        Args:
            obs: [player_y, velocity, dist_next, gap_center] (all normalized)

        Returns:
            0 (no flap) or 1 (flap)
        """
        pass


class RandomStrategy(ExplorationStrategy):
    """Pure random: 50/50 flap.  BAD for Flappy Bird — included as a demo
    of why naive exploration fails."""

    name = "Random"

    def explore(self, obs: np.ndarray) -> int:
        return int(np.random.randint(2))


class GravityAwareStrategy(ExplorationStrategy):
    """Biased random: low flap probability to counter the physics asymmetry.
    Since flap (-9) >> gravity (+1), we only flap ~15 % of the time."""

    name = "Gravity"

    def __init__(self, flap_prob: float = 0.15):
        self.flap_prob = flap_prob

    def explore(self, obs: np.ndarray) -> int:
        return 1 if np.random.random() < self.flap_prob else 0


class HeuristicStrategy(ExplorationStrategy):
    """Rule-based: flap if below the gap center, don't flap if above.
    A noise parameter adds randomness for exploration diversity."""

    name = "Heuristic"

    def __init__(self, noise: float = 0.15):
        self.noise = noise

    def explore(self, obs: np.ndarray) -> int:
        player_y = obs[0]
        gap_center = obs[3]
        ideal = 1 if player_y > gap_center else 0
        if np.random.random() < self.noise:
            return 1 - ideal
        return ideal


class GuidedStrategy(ExplorationStrategy):
    """Smart combo: gravity-aware when far from a pipe, heuristic when near.
    Best of both worlds — bird stays alive AND targets gaps."""

    name = "Guided"

    def __init__(self, flap_prob: float = 0.15, noise: float = 0.15,
                 near_threshold: float = 0.4):
        self.gravity = GravityAwareStrategy(flap_prob)
        self.heuristic = HeuristicStrategy(noise)
        self.near_threshold = near_threshold

    def explore(self, obs: np.ndarray) -> int:
        dist = obs[2]  # dist_next normalised [0, 1]
        if dist < self.near_threshold:
            return self.heuristic.explore(obs)
        return self.gravity.explore(obs)


STRATEGY_MAP = {
    "random": RandomStrategy,
    "gravity": GravityAwareStrategy,
    "heuristic": HeuristicStrategy,
    "guided": GuidedStrategy,
}
