"""Exploration strategies for the multi-bird RL race.

Physics recap (why this matters):
    flap = sets vel_y to -9 (strong upward impulse)
    gravity = +1 per frame
    One flap lifts the bird ~45 pixels before it starts falling.
    Pipe gap = 100 pixels.

So flapping is a BIG action. Smart strategies must account for velocity
to avoid constant overshooting (which makes birds look "dumb").

Observation format (5 relative features):
  [delta_y1, velocity, dist_pipe1, delta_y2, dist_pipe2]
  obs[0] delta_y1:   (bird_y - gap_center1) / SCREEN_HEIGHT  (positive=below gap)
  obs[1] velocity:   vel_y / MAX_VEL_Y  (negative=up, positive=down)
  obs[2] dist_pipe1: distance to next pipe / SCREEN_WIDTH
  obs[3] delta_y2:   (bird_y - gap_center2) / SCREEN_HEIGHT  (positive=below gap)
  obs[4] dist_pipe2: distance to second pipe / SCREEN_WIDTH
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
        return int(np.random.randint(2))


class GravityAwareStrategy(ExplorationStrategy):
    """Biased random with configurable flap probability.
    Default 12% compensates for flap(-9) >> gravity(+1)."""

    name = "Gravity"

    def __init__(self, flap_prob: float = 0.12, **_kwargs):
        self.flap_prob = flap_prob

    def explore(self, obs: np.ndarray) -> int:
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

        upward = max(0.0, -velocity)      # how fast going up (always >= 0)
        urgency = diff - 0.5 * upward     # damped position error

        action = 1 if urgency > self.threshold else 0

        if np.random.random() < self.noise:
            return 1 - action
        return action


class GuidedStrategy(ExplorationStrategy):
    """Always-on PD-controller heuristic with low noise.

    Uses the velocity-aware heuristic at ALL distances (not just near pipes).
    Previous version switched to random 12% flap far from pipes, which caused
    the bird to drift wildly between pipes and die at the second gap.
    """

    name = "Guided"

    def __init__(self, flap_prob: float = 0.12, noise: float = 0.02,
                 threshold: float = 0.04, **_kwargs):
        self.heuristic = HeuristicStrategy(noise=noise, threshold=threshold)

    def explore(self, obs: np.ndarray) -> int:
        return self.heuristic.explore(obs)


STRATEGY_MAP = {
    "random": RandomStrategy,
    "gravity": GravityAwareStrategy,
    "heuristic": HeuristicStrategy,
    "guided": GuidedStrategy,
}

STRATEGY_OPTIONS = list(STRATEGY_MAP.keys())
