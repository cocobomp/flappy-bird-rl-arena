"""Pygame metrics overlay for real-time RL agent monitoring.

Draws a semi-transparent panel on top of the game surface showing
episode progress, score, reward, step count, and agent-specific
metrics (e.g. epsilon, Q-table size, loss).
"""

from __future__ import annotations

import pygame


class Overlay:
    """HUD overlay that renders agent metrics on a Pygame surface.

    Usage:
        overlay = Overlay()
        # inside game loop:
        overlay.update(episode=1, total_episodes=10, score=3,
                       total_reward=42.5, steps=120,
                       agent_info={"epsilon": 0.05}, agent_name="DQN")
        overlay.draw(screen)
    """

    # Layout constants
    _PADDING = 8
    _LINE_HEIGHT = 22
    _PANEL_WIDTH = 280
    _FONT_SIZE = 18

    # Colours
    _KEY_COLOUR = (255, 255, 255)
    _VALUE_COLOUR = (0, 255, 100)
    _BG_ALPHA = 160

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self._font: pygame.font.Font | None = None

    def _get_font(self, size: int = _FONT_SIZE) -> pygame.font.Font:
        """Return a monospace bold font, lazily initialised.

        Pygame's font module must be initialised before this is called
        (which happens automatically when a display is created).

        Args:
            size: Font point size.

        Returns:
            A ``pygame.font.Font`` instance.
        """
        if self._font is None:
            if not pygame.font.get_init():
                pygame.font.init()
            self._font = pygame.font.SysFont("monospace", size, bold=True)
        return self._font

    def update(
        self,
        episode: int,
        total_episodes: int,
        score: int,
        total_reward: float,
        steps: int,
        agent_info: dict,
        agent_name: str,
    ) -> None:
        """Refresh the overlay data dictionary.

        Args:
            episode: Current episode number (1-based).
            total_episodes: Total number of episodes to run.
            score: Current game score (pipes passed).
            total_reward: Cumulative shaped reward this episode.
            steps: Step count within the current episode.
            agent_info: Dict returned by ``agent.get_info()``.
            agent_name: Human-readable agent identifier.
        """
        self.data = {
            "Agent": agent_name,
            "Episode": f"{episode}/{total_episodes}",
            "Score": str(score),
            "Reward": f"{total_reward:.1f}",
            "Steps": str(steps),
        }

        # Append agent-specific metrics
        for key, value in agent_info.items():
            display_key = key.replace("_", " ").title()
            if isinstance(value, float):
                self.data[display_key] = f"{value:.4f}"
            else:
                self.data[display_key] = str(value)

    def draw(self, surface: pygame.Surface) -> None:
        """Render the overlay onto the given Pygame surface.

        Draws a semi-transparent black panel in the top-left corner
        with each metric as ``key: value``. Keys are white, values
        are green.

        Args:
            surface: The Pygame surface to draw on (typically the screen).
        """
        if not self.data:
            return

        font = self._get_font()
        num_lines = len(self.data)
        panel_height = 2 * self._PADDING + num_lines * self._LINE_HEIGHT

        # Create a semi-transparent background panel
        panel = pygame.Surface(
            (self._PANEL_WIDTH, panel_height), pygame.SRCALPHA
        )
        panel.fill((0, 0, 0, self._BG_ALPHA))
        surface.blit(panel, (0, 0))

        # Render each metric line
        y = self._PADDING
        for key, value in self.data.items():
            key_surface = font.render(f"{key}: ", True, self._KEY_COLOUR)
            value_surface = font.render(str(value), True, self._VALUE_COLOUR)

            surface.blit(key_surface, (self._PADDING, y))
            surface.blit(
                value_surface,
                (self._PADDING + key_surface.get_width(), y),
            )
            y += self._LINE_HEIGHT
