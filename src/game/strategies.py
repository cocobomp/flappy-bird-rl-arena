"""Exploration strategies for the multi-bird RL race.

Physics recap (why this matters):
    flap = sets vel_y to -9 (strong upward impulse)
    gravity = +1 per frame
    One flap lifts the bird ~45 pixels before it starts falling.
    Pipe gap = 100 pixels.

So flapping is a BIG action. Smart strategies must account for velocity
to avoid constant overshooting (which makes birds look "dumb").

Observation format: [player_y, velocity, dist_next, gap_center]
  player_y:  bird position / SCREEN_HEIGHT  (0=top, ~0.8=ground)
  velocity:  bird vel_y / MAX_VEL_Y  (negative=up, positive=down)
  dist_next: distance to pipe / SCREEN_WIDTH  (0=at pipe, 1=far)
  gap_center: gap center / SCREEN_HEIGHT
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
        player_y = obs[0]
        velocity = obs[1]     # negative = going up
        gap_center = obs[3]

        diff = player_y - gap_center      # positive = below gap
        upward = max(0.0, -velocity)      # how fast going up (always >= 0)
        urgency = diff - 0.5 * upward     # damped position error

        action = 1 if urgency > self.threshold else 0

        if np.random.random() < self.noise:
            return 1 - action
        return action


class GuidedStrategy(ExplorationStrategy):
    """Best of both: gravity-aware far from pipe, PD-heuristic near pipe.

    Far from pipe: stay alive with low flap rate.
    Near pipe: use velocity-aware heuristic to thread the gap.
    """

    name = "Guided"

    def __init__(self, flap_prob: float = 0.12, noise: float = 0.10,
                 threshold: float = 0.04, **_kwargs):
        self.gravity = GravityAwareStrategy(flap_prob=flap_prob)
        self.heuristic = HeuristicStrategy(noise=noise, threshold=threshold)

    def explore(self, obs: np.ndarray) -> int:
        dist = obs[2]
        if dist < 0.5:
            return self.heuristic.explore(obs)
        return self.gravity.explore(obs)


STRATEGY_MAP = {
    "random": RandomStrategy,
    "gravity": GravityAwareStrategy,
    "heuristic": HeuristicStrategy,
    "guided": GuidedStrategy,
}

STRATEGY_OPTIONS = list(STRATEGY_MAP.keys())
