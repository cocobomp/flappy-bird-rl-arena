"""Pygame renderer for the multi-bird Flappy Bird game."""
import pygame
from src.game.engine import (
    FlappyBirdEngine, Bird,
    SCREEN_WIDTH, SCREEN_HEIGHT, GROUND_Y,
    PLAYER_WIDTH, PLAYER_HEIGHT, PIPE_WIDTH, PIPE_HEIGHT, PIPE_GAP,
)

PANEL_WIDTH = 280
EDUCATION_WIDTH = 300
WINDOW_WIDTH = SCREEN_WIDTH + PANEL_WIDTH + EDUCATION_WIDTH
WINDOW_HEIGHT = SCREEN_HEIGHT

# Colors
SKY_COLOR = (78, 192, 202)
GROUND_COLOR = (222, 216, 149)
PIPE_COLOR = (83, 164, 60)
PIPE_BORDER_COLOR = (60, 120, 40)
PANEL_BG = (30, 30, 40)
EDU_BG = (25, 28, 38)
TEXT_COLOR = (220, 220, 220)
HIGHLIGHT_COLOR = (0, 255, 120)
DEAD_COLOR = (150, 60, 60)
ALIVE_COLOR = (60, 200, 100)
BUTTON_COLOR = (60, 60, 80)
BUTTON_HOVER = (80, 80, 110)
BUTTON_TEXT = (220, 220, 220)
SECTION_COLOR = (100, 200, 255)
DIM_COLOR = (130, 130, 160)

# Tooltip descriptions
ALGO_TOOLTIPS = {
    "QL": "Q-Learning : methode tabulaire\nstocke les Q-valeurs pour\nchaque etat discretise",
    "DQN": "Deep Q-Network : reseau de neurones\napproxime les Q-valeurs avec\nexperience replay + target network",
    "DDQN": "Double DQN : variante du DQN\nreduit la surestimation des\nQ-valeurs (selection != evaluation)",
}

REWARD_TOOLTIPS = {
    "Basic": "Reward Basic : +1 en vie\n-1000 a la mort",
    "Dist": "Reward Distance : bonus\nproportionnel a la proximite\ndu prochain tuyau",
    "Center": "Reward Centered : bonus\npour rester centre dans\nl'ouverture du tuyau",
    "Smart": "Reward Smart : combine centrage\n+ direction + progression.\nRecommande pour l'apprentissage",
}

# Education panel content
EDUCATION_SECTIONS = [
    ("COMMENT CA MARCHE", None, [
        "Chaque oiseau est pilote par un",
        "agent RL qui apprend en jouant.",
        "",
    ]),
    ("Exploration (e-greedy)", SECTION_COLOR, [
        "Au debut e=1.0 : actions 100%",
        "aleatoires. 50% de flap = monte",
        "toujours ! e diminue au fil du",
        "temps, l'agent utilise alors ses",
        "connaissances acquises.",
        "",
    ]),
    ("Algorithmes", SECTION_COLOR, [
        "QL  Table de Q-valeurs. Simple",
        "    mais etats discretises.",
        "DQN Reseau de neurones + replay",
        "    buffer. Etats continus.",
        "DDQN Corrige la surestimation",
        "    des Q-valeurs du DQN.",
        "",
    ]),
    ("Recompenses", SECTION_COLOR, [
        "Basic  +1 vie, -1000 mort",
        "       Signal faible (sparse)",
        "Dist   Bonus proximite tuyau",
        "Center Bonus centrage dans gap",
        "Smart  Tout combine ! Centrage",
        "       + direction + progression",
        "",
    ]),
    ("Conseils", SECTION_COLOR, [
        "* Reward Smart = apprend vite",
        "* e petit = moins aleatoire",
        "  mais moins d'exploration",
        "* Learning rate : equilibre",
        "  vitesse / stabilite",
        "* Observez e diminuer : c'est",
        "  l'agent qui apprend !",
        "* Comparez les algos avec la",
        "  meme reward pour voir la",
        "  difference d'apprentissage",
    ]),
]


class GameRenderer:
    """Renders the multi-bird game + stats panel + education panel."""

    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("RL Flappy Bird Race")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_title = pygame.font.SysFont("monospace", 22, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 11)
        self._algo_rects = []

    def draw(self, engine: FlappyBirdEngine, speed: int, paused: bool,
             bird_configs: dict, buttons: list[dict]):
        """Draw the full frame: game area + panel + education."""
        self._algo_rects = []
        self._draw_game(engine)
        self._draw_panel(engine, speed, paused, bird_configs, buttons)
        self._draw_education_panel()
        self._draw_tooltips()
        pygame.display.flip()

    def _draw_game(self, engine: FlappyBirdEngine):
        """Draw game area: sky, pipes, birds, ground."""
        game_surface = self.screen.subsurface((0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))
        game_surface.fill(SKY_COLOR)

        # Pipes
        for pipe in engine.pipes:
            px = int(pipe["x"])
            upper_rect = pygame.Rect(px, 0, PIPE_WIDTH, pipe["gap_y"])
            pygame.draw.rect(game_surface, PIPE_COLOR, upper_rect)
            pygame.draw.rect(game_surface, PIPE_BORDER_COLOR, upper_rect, 2)
            lip = pygame.Rect(px - 2, pipe["gap_y"] - 20, PIPE_WIDTH + 4, 20)
            pygame.draw.rect(game_surface, PIPE_BORDER_COLOR, lip)

            lower_top = pipe["gap_y"] + PIPE_GAP
            lower_rect = pygame.Rect(px, lower_top, PIPE_WIDTH, SCREEN_HEIGHT - lower_top)
            pygame.draw.rect(game_surface, PIPE_COLOR, lower_rect)
            pygame.draw.rect(game_surface, PIPE_BORDER_COLOR, lower_rect, 2)
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
            body = pygame.Rect(bx, by, PLAYER_WIDTH, PLAYER_HEIGHT)
            pygame.draw.ellipse(game_surface, bird.color, body)
            pygame.draw.ellipse(game_surface, (0, 0, 0), body, 2)
            eye_x = bx + PLAYER_WIDTH - 8
            eye_y = by + 6
            pygame.draw.circle(game_surface, (255, 255, 255), (eye_x, eye_y), 5)
            pygame.draw.circle(game_surface, (0, 0, 0), (eye_x, eye_y), 2)

        # Connecting lines from alive birds to next pipe gap
        if engine.pipes:
            next_pipe = None
            for pipe in engine.pipes:
                if pipe["x"] + PIPE_WIDTH > PLAYER_WIDTH:
                    next_pipe = pipe
                    break
            if next_pipe:
                gap_center = int(next_pipe["gap_y"] + PIPE_GAP / 2)
                px = int(next_pipe["x"])
                pygame.draw.circle(game_surface, (255, 255, 0), (px, gap_center), 5, 2)
                for bird in engine.birds:
                    if not bird.alive:
                        continue
                    bx = int(bird.x + PLAYER_WIDTH)
                    by = int(bird.y + PLAYER_HEIGHT // 2)
                    pygame.draw.line(game_surface, bird.color, (bx, by), (px, gap_center), 1)

    def _draw_panel(self, engine: FlappyBirdEngine, speed: int, paused: bool,
                    bird_configs: dict, buttons: list[dict]):
        """Draw the side panel with stats and controls."""
        panel = self.screen.subsurface((SCREEN_WIDTH, 0, PANEL_WIDTH, WINDOW_HEIGHT))
        panel.fill(PANEL_BG)

        y = 10
        title = self.font_title.render(f"ROUND {engine.round_num}", True, HIGHLIGHT_COLOR)
        panel.blit(title, (10, y))
        y += 30

        speed_txt = self.font.render(f"Speed: x{speed}  {'PAUSED' if paused else ''}", True, TEXT_COLOR)
        panel.blit(speed_txt, (10, y))
        y += 25

        pygame.draw.line(panel, (80, 80, 100), (10, y), (PANEL_WIDTH - 10, y), 1)
        y += 10

        # Bird stats
        for bird in engine.birds:
            config = bird_configs.get(bird.bird_id, {})
            algo = config.get("algo", "?")
            reward = config.get("reward", "?")
            best = config.get("best_score", 0)
            epsilon = config.get("epsilon", 0)

            # Color indicator
            indicator = pygame.Rect(10, y, 12, 12)
            pygame.draw.rect(panel, bird.color, indicator)
            pygame.draw.rect(panel, (200, 200, 200), indicator, 1)

            status_color = ALIVE_COLOR if bird.alive else DEAD_COLOR
            status = "alive" if bird.alive else "dead"

            label = self.font.render(f" {algo} + {reward}", True, TEXT_COLOR)
            panel.blit(label, (26, y - 2))
            # Store rect for tooltip
            label_rect = pygame.Rect(SCREEN_WIDTH + 26, y - 2, label.get_width(), label.get_height())
            tooltip = ALGO_TOOLTIPS.get(algo, "") + "\n\n" + REWARD_TOOLTIPS.get(reward, "")
            self._algo_rects.append((label_rect, tooltip))
            y += 16

            score_txt = self.font.render(
                f"  Score:{bird.score} Best:{best} [{status}]", True, status_color
            )
            panel.blit(score_txt, (10, y - 2))
            y += 14

            # Epsilon + pipe info
            next_pipe = None
            for pipe in engine.pipes:
                if pipe["x"] + PIPE_WIDTH > bird.x:
                    next_pipe = pipe
                    break
            if next_pipe:
                dist = int(next_pipe["x"] - bird.x)
                gap_top = next_pipe["gap_y"]
                gap_bot = next_pipe["gap_y"] + PIPE_GAP
                info_txt = self.font_small.render(
                    f"  e={epsilon}  pipe:d={dist} gap={gap_top}-{gap_bot}", True, DIM_COLOR
                )
            else:
                info_txt = self.font_small.render(f"  e={epsilon}", True, DIM_COLOR)
            panel.blit(info_txt, (10, y - 2))
            y += 16

        # Separator
        pygame.draw.line(panel, (80, 80, 100), (10, y), (PANEL_WIDTH - 10, y), 1)
        y += 10

        # Buttons
        mouse_pos = pygame.mouse.get_pos()
        mouse_x = mouse_pos[0] - SCREEN_WIDTH
        mouse_y = mouse_pos[1]

        for btn in buttons:
            btn_rect = pygame.Rect(10, y, PANEL_WIDTH - 20, 30)
            btn["rect"] = btn_rect
            hover = btn_rect.collidepoint(mouse_x, mouse_y)
            color = BUTTON_HOVER if hover else BUTTON_COLOR
            pygame.draw.rect(panel, color, btn_rect, border_radius=4)
            pygame.draw.rect(panel, (100, 100, 120), btn_rect, 1, border_radius=4)
            txt = self.font.render(btn["label"], True, BUTTON_TEXT)
            tx = btn_rect.x + (btn_rect.width - txt.get_width()) // 2
            ty = btn_rect.y + (btn_rect.height - txt.get_height()) // 2
            panel.blit(txt, (tx, ty))
            y += 38

    def _draw_education_panel(self):
        """Draw the right-side education/pedagogy panel."""
        edu_x = SCREEN_WIDTH + PANEL_WIDTH
        edu = self.screen.subsurface((edu_x, 0, EDUCATION_WIDTH, WINDOW_HEIGHT))
        edu.fill(EDU_BG)

        y = 10
        for section_title, title_color, lines in EDUCATION_SECTIONS:
            if title_color is None:
                # Main title
                t = self.font_large.render(section_title, True, HIGHLIGHT_COLOR)
                edu.blit(t, (10, y))
                y += 24
            else:
                # Section title
                t = self.font.render(f"► {section_title}", True, title_color)
                edu.blit(t, (10, y))
                y += 18
            for line in lines:
                if line == "":
                    y += 6
                    continue
                t = self.font_small.render(line, True, TEXT_COLOR)
                edu.blit(t, (14, y))
                y += 14

        # Separator
        pygame.draw.line(edu, (60, 60, 80), (10, y), (EDUCATION_WIDTH - 10, y))

    def _draw_tooltips(self):
        """Draw tooltip if mouse hovers over an algorithm label."""
        mouse_pos = pygame.mouse.get_pos()
        for rect, text in self._algo_rects:
            if rect.collidepoint(mouse_pos):
                lines = text.split("\n")
                rendered = [self.font_small.render(line, True, TEXT_COLOR) for line in lines if line]
                if not rendered:
                    break
                max_w = max(r.get_width() for r in rendered)
                total_h = sum(r.get_height() for r in rendered)
                tx = mouse_pos[0] - max_w - 15
                ty = mouse_pos[1] - 10
                if tx < 0:
                    tx = mouse_pos[0] + 15
                if ty + total_h + 8 > WINDOW_HEIGHT:
                    ty = WINDOW_HEIGHT - total_h - 8
                bg = pygame.Rect(tx - 6, ty - 4, max_w + 12, total_h + 8)
                pygame.draw.rect(self.screen, (20, 20, 35), bg, border_radius=4)
                pygame.draw.rect(self.screen, (100, 100, 140), bg, 1, border_radius=4)
                cy = ty
                for r in rendered:
                    self.screen.blit(r, (tx, cy))
                    cy += r.get_height()
                break

    def quit(self):
        pygame.quit()
