"""Game renderer for visualising trained RL agents playing Flappy Bird.

Creates a Pygame window that shows the FlappyBird-v0 environment with the
chosen observation/reward wrappers and a live metrics overlay.
"""

from __future__ import annotations

import sys

import flappy_bird_gymnasium  # noqa: F401 — registers FlappyBird-v0
import gymnasium
import pygame

from src.agents.base_agent import BaseAgent
from src.environments.rewards import BasicReward, CenteredReward, DistanceReward
from src.environments.wrappers import (
    CustomRewardWrapper,
    EnrichedObsWrapper,
    SimpleObsWrapper,
)
from src.visualization.overlay import Overlay

# Mapping from CLI-friendly names to wrapper/reward classes
OBS_WRAPPER_MAP: dict[str, type | None] = {
    "simple": SimpleObsWrapper,
    "enriched": EnrichedObsWrapper,
    "raw": None,
}

REWARD_MAP: dict[str, type] = {
    "basic": BasicReward,
    "distance": DistanceReward,
    "centered": CenteredReward,
}


class GameRenderer:
    """Run a trained agent in the FlappyBird-v0 environment with visual rendering.

    Wraps the base environment with the same reward and observation wrappers
    used during training, and draws a live metrics overlay on each frame.

    Args:
        agent: A trained ``BaseAgent`` instance.
        obs_type: Observation wrapper key (``"simple"``, ``"enriched"``, or ``"raw"``).
        reward_type: Reward function key (``"basic"``, ``"distance"``, or ``"centered"``).
        fps: Target frames per second for the rendering loop.
    """

    def __init__(
        self,
        agent: BaseAgent,
        obs_type: str = "simple",
        reward_type: str = "basic",
        fps: int = 30,
    ) -> None:
        self.agent = agent
        self.obs_type = obs_type
        self.reward_type = reward_type
        self.fps = fps
        self.overlay = Overlay()

    def _build_env(self) -> gymnasium.Env:
        """Create and wrap the FlappyBird-v0 environment.

        Applies the reward wrapper first, then the observation wrapper,
        matching the order used during training.

        Returns:
            A fully wrapped gymnasium environment with ``render_mode="human"``.
        """
        env = gymnasium.make(
            "FlappyBird-v0", use_lidar=False, render_mode="human"
        )

        # Apply reward wrapper
        reward_cls = REWARD_MAP.get(self.reward_type)
        if reward_cls is not None:
            env = CustomRewardWrapper(env, reward_cls())

        # Apply observation wrapper
        obs_cls = OBS_WRAPPER_MAP.get(self.obs_type)
        if obs_cls is not None:
            env = obs_cls(env)

        return env

    def run(self, num_episodes: int = 10) -> None:
        """Run the agent for the given number of episodes with rendering.

        Handles Pygame QUIT and ESCAPE events for early exit.

        Args:
            num_episodes: How many episodes to play.
        """
        env = self._build_env()
        clock = pygame.time.Clock()
        agent_name = type(self.agent).__name__

        try:
            for episode in range(1, num_episodes + 1):
                obs, info = env.reset()
                done = False
                total_reward = 0.0
                steps = 0
                score = 0

                while not done:
                    # Handle Pygame events
                    if self._should_quit():
                        return

                    action = self.agent.select_action(obs, training=False)
                    obs, reward, terminated, truncated, info = env.step(action)

                    done = terminated or truncated
                    total_reward += reward
                    steps += 1
                    score = info.get("score", score)

                    # Update and draw overlay
                    self.overlay.update(
                        episode=episode,
                        total_episodes=num_episodes,
                        score=score,
                        total_reward=total_reward,
                        steps=steps,
                        agent_info=self.agent.get_info(),
                        agent_name=agent_name,
                    )

                    # The env with render_mode="human" renders to screen
                    # automatically. We draw the overlay on the active display.
                    display_surface = pygame.display.get_surface()
                    if display_surface is not None:
                        self.overlay.draw(display_surface)
                        pygame.display.flip()

                    clock.tick(self.fps)

                print(
                    f"Episode {episode}/{num_episodes} -- "
                    f"Score: {score}, Reward: {total_reward:.1f}, "
                    f"Steps: {steps}"
                )
        finally:
            env.close()

    @staticmethod
    def _should_quit() -> bool:
        """Check Pygame events for quit or escape key.

        Returns:
            True if the user requested to quit, False otherwise.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_ESCAPE
            ):
                return True
        return False
