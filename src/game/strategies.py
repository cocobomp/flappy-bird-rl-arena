"""Exploration strategies for the multi-bird RL race.

During epsilon-greedy training, the standard approach picks a random action
with probability epsilon. In Flappy Bird, this is catastrophic:

    flap force = -9 (strong upward)
    gravity    = +1 (weak downward)

With 50/50 random, the bird averages velocity ~ -4 per frame -> flies into
the ceiling every time.  These strategies replace the random action with
smarter exploration so birds actually survive long enough to learn.

Observation format: [player_y, velocity, dist_next, gap_center]
  - player_y:   bird vertical position / SCREEN_HEIGHT  (0=top, ~0.8=ground)
  - velocity:   bird vel_y / MAX_VEL_Y  (negative=going up, positive=going down)
  - dist_next:  distance to next pipe / SCREEN_WIDTH  (0=at pipe, 1=far away)
  - gap_center: center of pipe gap / SCREEN_HEIGHT
"""

from abc import ABC, abstractmethod

import numpy as np


class ExplorationStrategy(ABC):
    """Base class for exploration strategies."""

    name: str = "?"

    @abstractmethod
    def explore(self, obs: np.ndarray) -> int:
        """Choose an exploration action given the current observation.

        Returns:
            0 (no flap) or 1 (flap)
        """
        pass


class RandomStrategy(ExplorationStrategy):
    """Pure random: 50/50 flap.  BAD for Flappy Bird -- included as a demo
    of why naive exploration fails."""

    name = "Random"

    def explore(self, obs: np.ndarray) -> int:
        return int(np.random.randint(2))


class GravityAwareStrategy(ExplorationStrategy):
    """Biased random: low flap probability to counter the physics asymmetry.
    Since flap (-9) >> gravity (+1), we only flap ~12% of the time."""

    name = "Gravity"

    def __init__(self, flap_prob: float = 0.12):
        self.flap_prob = flap_prob

    def explore(self, obs: np.ndarray) -> int:
        return 1 if np.random.random() < self.flap_prob else 0


class HeuristicStrategy(ExplorationStrategy):
    """Velocity-aware rule: flap if below the gap AND not already rising fast.

    Key insight: a single flap gives vel=-9 which lasts many frames.
    So we only flap when truly needed:
      - Bird is below gap center AND not already going up fast -> flap
      - Bird is above gap center OR already rising -> don't flap

    This produces smooth, intelligent trajectories that look like real play.
    """

    name = "Heuristic"

    def __init__(self, noise: float = 0.10):
        self.noise = noise

    def explore(self, obs: np.ndarray) -> int:
        player_y = obs[0]
        velocity = obs[1]   # negative = going up
        gap_center = obs[3]

        diff = player_y - gap_center  # positive = below gap

        # Flap only when below gap AND not already rising
        if diff > 0.02 and velocity > -0.3:
            action = 1
        # Close to gap and moving up gently: let gravity do the work
        else:
            action = 0

        if np.random.random() < self.noise:
            return 1 - action
        return action


class GuidedStrategy(ExplorationStrategy):
    """Smart combo: gravity-aware when far from a pipe, heuristic when near.

    Far from pipe: just stay alive with low flap rate.
    Near pipe: use velocity-aware heuristic to target the gap precisely.
    """

    name = "Guided"

    def __init__(self, flap_prob: float = 0.12, noise: float = 0.10,
                 near_threshold: float = 0.5):
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

STRATEGY_OPTIONS = list(STRATEGY_MAP.keys())
