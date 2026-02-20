"""Pygame renderer for the multi-bird Flappy Bird game."""
import pygame
import numpy as np
from src.game.engine import (
    FlappyBirdEngine, Bird,
    SCREEN_WIDTH, SCREEN_HEIGHT, GROUND_Y,
    PLAYER_WIDTH, PLAYER_HEIGHT, PIPE_WIDTH, PIPE_HEIGHT, PIPE_GAP,
)

PANEL_WIDTH = 280
WINDOW_WIDTH = SCREEN_WIDTH + PANEL_WIDTH
WINDOW_HEIGHT = SCREEN_HEIGHT

# Colors
SKY_COLOR = (78, 192, 202)
GROUND_COLOR = (222, 216, 149)
PIPE_COLOR = (83, 164, 60)
PIPE_BORDER_COLOR = (60, 120, 40)
PANEL_BG = (30, 30, 40)
TEXT_COLOR = (220, 220, 220)
HIGHLIGHT_COLOR = (0, 255, 120)
DEAD_COLOR = (150, 60, 60)
ALIVE_COLOR = (60, 200, 100)
BUTTON_COLOR = (60, 60, 80)
BUTTON_HOVER = (80, 80, 110)
BUTTON_TEXT = (220, 220, 220)


class GameRenderer:
    """Renders the multi-bird game + side panel."""

    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("RL Flappy Bird Race")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_title = pygame.font.SysFont("monospace", 22, bold=True)

    def draw(self, engine: FlappyBirdEngine, speed: int, paused: bool,
             bird_configs: dict, buttons: list[dict]):
        """Draw the full frame: game area + panel."""
        self._draw_game(engine)
        self._draw_panel(engine, speed, paused, bird_configs, buttons)
        pygame.display.flip()

    def _draw_game(self, engine: FlappyBirdEngine):
        """Draw game area: sky, pipes, birds, ground."""
        game_surface = self.screen.subsurface((0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))
        # Sky
        game_surface.fill(SKY_COLOR)

        # Pipes
        for pipe in engine.pipes:
            px = int(pipe["x"])
            # Upper pipe
            upper_rect = pygame.Rect(px, 0, PIPE_WIDTH, pipe["gap_y"])
            pygame.draw.rect(game_surface, PIPE_COLOR, upper_rect)
            pygame.draw.rect(game_surface, PIPE_BORDER_COLOR, upper_rect, 2)
            # Lip
            lip = pygame.Rect(px - 2, pipe["gap_y"] - 20, PIPE_WIDTH + 4, 20)
            pygame.draw.rect(game_surface, PIPE_BORDER_COLOR, lip)

            # Lower pipe
            lower_top = pipe["gap_y"] + PIPE_GAP
            lower_rect = pygame.Rect(px, lower_top, PIPE_WIDTH, SCREEN_HEIGHT - lower_top)
            pygame.draw.rect(game_surface, PIPE_COLOR, lower_rect)
            pygame.draw.rect(game_surface, PIPE_BORDER_COLOR, lower_rect, 2)
            # Lip
            lip2 = pygame.Rect(px - 2, lower_top, PIPE_WIDTH + 4, 20)
            pygame.draw.rect(game_surface, PIPE_BORDER_COLOR, lip2)

        # Ground
        ground_rect = pygame.Rect(0, GROUND_Y, SCREEN_WIDTH, SCREEN_HEIGHT - GROUND_Y)
        pygame.draw.rect(game_surface, GROUND_COLOR, ground_rect)
        pygame.draw.line(game_surface, (180, 170, 110), (0, GROUND_Y), (SCREEN_WIDTH, GROUND_Y), 2)

        # Birds
        for bird in engine.birds:
            if not bird.alive:
                continue
            bx, by = int(bird.x), int(bird.y)
            # Body (filled circle-ish rect with color)
            body = pygame.Rect(bx, by, PLAYER_WIDTH, PLAYER_HEIGHT)
            pygame.draw.ellipse(game_surface, bird.color, body)
            pygame.draw.ellipse(game_surface, (0, 0, 0), body, 2)
            # Eye
            eye_x = bx + PLAYER_WIDTH - 8
            eye_y = by + 6
            pygame.draw.circle(game_surface, (255, 255, 255), (eye_x, eye_y), 5)
            pygame.draw.circle(game_surface, (0, 0, 0), (eye_x, eye_y), 2)

    def _draw_panel(self, engine: FlappyBirdEngine, speed: int, paused: bool,
                    bird_configs: dict, buttons: list[dict]):
        """Draw the side panel with stats and controls."""
        panel = self.screen.subsurface((SCREEN_WIDTH, 0, PANEL_WIDTH, WINDOW_HEIGHT))
        panel.fill(PANEL_BG)

        y = 10
        # Title
        title = self.font_title.render(f"ROUND {engine.round_num}", True, HIGHLIGHT_COLOR)
        panel.blit(title, (10, y))
        y += 30

        speed_txt = self.font.render(f"Speed: x{speed}  {'PAUSED' if paused else ''}", True, TEXT_COLOR)
        panel.blit(speed_txt, (10, y))
        y += 25

        # Separator
        pygame.draw.line(panel, (80, 80, 100), (10, y), (PANEL_WIDTH - 10, y), 1)
        y += 10

        # Bird stats
        for bird in engine.birds:
            config = bird_configs.get(bird.bird_id, {})
            name = config.get("name", f"Bird {bird.bird_id}")
            algo = config.get("algo", "?")
            reward = config.get("reward", "?")
            best = config.get("best_score", 0)

            # Color indicator
            indicator = pygame.Rect(10, y, 12, 12)
            pygame.draw.rect(panel, bird.color, indicator)
            pygame.draw.rect(panel, (200, 200, 200), indicator, 1)

            # Status
            status_color = ALIVE_COLOR if bird.alive else DEAD_COLOR
            status = "alive" if bird.alive else "dead"

            label = self.font.render(f" {algo} + {reward}", True, TEXT_COLOR)
            panel.blit(label, (26, y - 2))
            y += 18

            score_txt = self.font.render(
                f"   Score: {bird.score}  Best: {best}  [{status}]", True, status_color
            )
            panel.blit(score_txt, (10, y - 2))
            y += 22

        # Separator
        pygame.draw.line(panel, (80, 80, 100), (10, y), (PANEL_WIDTH - 10, y), 1)
        y += 10

        # Buttons
        mouse_pos = pygame.mouse.get_pos()
        mouse_x = mouse_pos[0] - SCREEN_WIDTH  # relative to panel
        mouse_y = mouse_pos[1]

        for btn in buttons:
            btn_rect = pygame.Rect(10, y, PANEL_WIDTH - 20, 30)
            btn["rect"] = btn_rect  # store for click detection
            hover = btn_rect.collidepoint(mouse_x, mouse_y)
            color = BUTTON_HOVER if hover else BUTTON_COLOR
            pygame.draw.rect(panel, color, btn_rect, border_radius=4)
            pygame.draw.rect(panel, (100, 100, 120), btn_rect, 1, border_radius=4)
            txt = self.font.render(btn["label"], True, BUTTON_TEXT)
            tx = btn_rect.x + (btn_rect.width - txt.get_width()) // 2
            ty = btn_rect.y + (btn_rect.height - txt.get_height()) // 2
            panel.blit(txt, (tx, ty))
            y += 38

    def quit(self):
        pygame.quit()
