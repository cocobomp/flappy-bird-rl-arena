"""Pygame renderer for the multi-bird Flappy Bird game."""
import pygame
from src.game.engine import (
    FlappyBirdEngine, Bird,
    SCREEN_WIDTH, SCREEN_HEIGHT, GROUND_Y,
    PLAYER_WIDTH, PLAYER_HEIGHT, PIPE_WIDTH, PIPE_HEIGHT, PIPE_GAP,
)
from src.game.ui import Slider

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
EXPLORE_COLOR = (255, 180, 50)
EXPLOIT_COLOR = (50, 200, 255)
WARNING_COLOR = (255, 100, 100)
MINI_BTN = (55, 55, 75)
MINI_BTN_HOVER = (75, 75, 100)

# Tooltip descriptions
ALGO_TOOLTIPS = {
    "QL": "Q-Learning : methode tabulaire\nstocke les Q-valeurs pour\nchaque etat discretise",
    "DQN": "Deep Q-Network : reseau de neurones\napproxime les Q-valeurs avec\nexperience replay + target network",
    "DDQN": "Double DQN : variante du DQN\nreduit la surestimation des\nQ-valeurs (selection != evaluation)",
}

REWARD_TOOLTIPS = {
    "Basic": "+1 en vie, -1000 a la mort",
    "Dist": "Bonus proximite prochain tuyau",
    "Center": "Bonus centrage dans le gap",
    "Smart": "Centrage + direction + progression",
}

STRATEGY_TOOLTIPS = {
    "Random": "50/50 aleatoire (monte toujours!)",
    "Gravity": "12% de flap (compense physique)",
    "Heurist.": "Flap si sous porte + velocite",
    "Guided": "Gravity loin + Heuristique pres",
}

# Education panel content
EDUCATION_SECTIONS = [
    ("COMMENT CA MARCHE", None, [
        "Chaque oiseau est un agent RL",
        "qui apprend a jouer en jouant.",
        "",
    ]),
    ("Physique du jeu", WARNING_COLOR, [
        "Flap=-9 (fort!) Gravite=+1",
        "50% flap = monte au plafond!",
        "",
    ]),
    ("Strategies", SECTION_COLOR, [
        "Random  50/50 (nul!)",
        "Gravity 12% flap seulement",
        "Heurist PD: position+velocite",
        "Guided  Gravity+Heuristique",
        "",
    ]),
    ("Parametres", SECTION_COLOR, [
        "Epsilon = taux exploration",
        "LR      = vitesse apprentissage",
        "Decay   = reduction epsilon",
        "Mort    = penalite deces",
        "Porte   = bonus par tuyau",
        "Survie  = bonus par frame",
        "Seuil   = sensibilite flap",
        "Bruit   = aleatoire strategie",
        "",
    ]),
    ("Entrainement", SECTION_COLOR, [
        "EXPLORE = utilise strategie",
        "APPREND = utilise le reseau!",
        "Q-values: quand elles varient",
        "le reseau a appris!",
        "",
    ]),
    ("Controles", SECTION_COLOR, [
        "[Strat] Change strategie",
        "[e/2]   Epsilon / 2",
        "[X] Supprime  [Boost] e=0.05",
        "",
    ]),
    ("Conseils", SECTION_COLOR, [
        "* Guided + Smart = apprend vite",
        "* Vitesse x32-x64 pour aller",
        "  vite, puis [e/2] quand les",
        "  Q-values commencent a varier",
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
        self.font_tiny = pygame.font.SysFont("monospace", 10)
        self._hover_rects = []
        self._bird_buttons = []  # [(rect_panel_coords, action_str, bird_index)]
        # Speed slider in panel coords
        self._speed_slider = Slider(10, 50, PANEL_WIDTH - 20, 1, 64, 1, "Vitesse")

    @property
    def speed(self):
        return max(1, round(self._speed_slider.value))

    @speed.setter
    def speed(self, val):
        self._speed_slider.value = max(1, min(64, val))

    def handle_event(self, event) -> tuple | None:
        """Handle mouse events for panel controls.

        Returns:
            ("cycle_strategy", bird_index) or ("halve_epsilon", bird_index)
            or ("remove", bird_index) or None.
        """
        # Forward to speed slider (translate to panel coords)
        self._speed_slider.handle_event(event, dx=SCREEN_WIDTH, dy=0)

        if event.type == pygame.MOUSEBUTTONDOWN:
            mx = event.pos[0] - SCREEN_WIDTH
            my = event.pos[1]
            for rect, action, idx in self._bird_buttons:
                if rect.collidepoint(mx, my):
                    return (action, idx)
        return None

    def draw(self, engine: FlappyBirdEngine, speed_val: int, paused: bool,
             bird_configs: dict, buttons: list[dict]):
        """Draw the full frame: game area + panel + education."""
        self._hover_rects = []
        self._bird_buttons = []
        self._draw_game(engine)
        self._draw_panel(engine, paused, bird_configs, buttons)
        self._draw_education_panel()
        self._draw_tooltips()
        pygame.display.flip()

    def _draw_game(self, engine: FlappyBirdEngine):
        """Draw game area: sky, pipes, birds, ground."""
        game_surface = self.screen.subsurface((0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))
        game_surface.fill(SKY_COLOR)

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

        # Lines from alive birds to next pipe gap
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

    def _draw_panel(self, engine: FlappyBirdEngine, paused: bool,
                    bird_configs: dict, buttons: list[dict]):
        """Draw the side panel with speed slider, bird stats, controls, and buttons."""
        panel = self.screen.subsurface((SCREEN_WIDTH, 0, PANEL_WIDTH, WINDOW_HEIGHT))
        panel.fill(PANEL_BG)

        y = 8
        title = self.font_title.render(f"ROUND {engine.round_num}", True, HIGHLIGHT_COLOR)
        panel.blit(title, (10, y))

        # Paused label
        if paused:
            p = self.font.render("PAUSED", True, WARNING_COLOR)
            panel.blit(p, (PANEL_WIDTH - p.get_width() - 10, y + 4))
        y += 28

        # Speed slider
        self._speed_slider.draw(panel, self.font_small)
        y = 72

        pygame.draw.line(panel, (80, 80, 100), (10, y), (PANEL_WIDTH - 10, y), 1)
        y += 8

        # Mouse for hover detection on mini-buttons
        mouse_pos = pygame.mouse.get_pos()
        panel_mx = mouse_pos[0] - SCREEN_WIDTH
        panel_my = mouse_pos[1]

        # Bird stats
        for bird in engine.birds:
            config = bird_configs.get(bird.bird_id, {})
            algo = config.get("algo", "?")
            reward = config.get("reward", "?")
            strategy = config.get("strategy", "?")
            best = config.get("best_score", 0)
            total_pipes = config.get("total_pipes", 0)
            epsilon = config.get("epsilon", 0)
            exploring = config.get("exploring", True)
            q_display = config.get("q_display", "")
            bird_idx = config.get("index", 0)

            status_color = ALIVE_COLOR if bird.alive else DEAD_COLOR

            # Line 1: color indicator + algo/reward/strategy
            indicator = pygame.Rect(10, y, 10, 10)
            pygame.draw.rect(panel, bird.color, indicator)
            label = self.font.render(f"{algo} {reward} [{strategy}]", True, TEXT_COLOR)
            panel.blit(label, (24, y - 2))
            # Tooltip
            label_rect = pygame.Rect(SCREEN_WIDTH + 24, y - 2, label.get_width(), label.get_height())
            tooltip = (ALGO_TOOLTIPS.get(algo, "") + "\n"
                       + REWARD_TOOLTIPS.get(reward, "") + "\n"
                       + STRATEGY_TOOLTIPS.get(strategy, ""))
            self._hover_rects.append((label_rect, tooltip))
            y += 15

            # Line 2: score + best + pipes + status
            score_txt = self.font_small.render(
                f" S:{bird.score} B:{best} P:{total_pipes} {'alive' if bird.alive else 'dead'}",
                True, status_color,
            )
            panel.blit(score_txt, (10, y))
            y += 13

            # Line 3: epsilon + Q-values + mode badge
            mode_color = EXPLORE_COLOR if exploring else EXPLOIT_COLOR
            mode_label = "EXPLORE" if exploring else "APPREND"

            eps_txt = self.font_tiny.render(f" e={epsilon}", True, DIM_COLOR)
            panel.blit(eps_txt, (10, y))

            if q_display:
                q_txt = self.font_tiny.render(q_display, True, (160, 180, 220))
                panel.blit(q_txt, (80, y))

            mode_txt = self.font_tiny.render(f"[{mode_label}]", True, mode_color)
            panel.blit(mode_txt, (PANEL_WIDTH - mode_txt.get_width() - 10, y))
            y += 13

            # Line 4: mini control buttons [Strat] [e/2] [X]
            btn_w = 60
            btn_h = 16
            gap = 6
            bx = 14

            # [Strat] button
            strat_rect = pygame.Rect(bx, y, btn_w, btn_h)
            hover = strat_rect.collidepoint(panel_mx, panel_my)
            pygame.draw.rect(panel, MINI_BTN_HOVER if hover else MINI_BTN, strat_rect, border_radius=3)
            st = self.font_tiny.render("Strat >", True, SECTION_COLOR)
            panel.blit(st, (bx + 4, y + 2))
            self._bird_buttons.append((strat_rect, "cycle_strategy", bird_idx))
            bx += btn_w + gap

            # [e/2] button
            eps_rect = pygame.Rect(bx, y, btn_w, btn_h)
            hover = eps_rect.collidepoint(panel_mx, panel_my)
            pygame.draw.rect(panel, MINI_BTN_HOVER if hover else MINI_BTN, eps_rect, border_radius=3)
            et = self.font_tiny.render("e / 2", True, EXPLORE_COLOR)
            panel.blit(et, (bx + 10, y + 2))
            self._bird_buttons.append((eps_rect, "halve_epsilon", bird_idx))
            bx += btn_w + gap

            # [X] button
            x_rect = pygame.Rect(bx, y, 30, btn_h)
            hover = x_rect.collidepoint(panel_mx, panel_my)
            pygame.draw.rect(panel, (120, 40, 40) if hover else (80, 40, 40), x_rect, border_radius=3)
            xt = self.font_tiny.render("X", True, WARNING_COLOR)
            panel.blit(xt, (bx + 10, y + 2))
            self._bird_buttons.append((x_rect, "remove", bird_idx))

            y += 20

        # Separator
        pygame.draw.line(panel, (80, 80, 100), (10, y), (PANEL_WIDTH - 10, y), 1)
        y += 8

        # Main buttons
        for btn in buttons:
            btn_rect = pygame.Rect(10, y, PANEL_WIDTH - 20, 26)
            btn["rect"] = btn_rect
            hover = btn_rect.collidepoint(panel_mx, panel_my)
            color = BUTTON_HOVER if hover else BUTTON_COLOR
            pygame.draw.rect(panel, color, btn_rect, border_radius=4)
            pygame.draw.rect(panel, (100, 100, 120), btn_rect, 1, border_radius=4)
            txt = self.font.render(btn["label"], True, BUTTON_TEXT)
            tx = btn_rect.x + (btn_rect.width - txt.get_width()) // 2
            ty = btn_rect.y + (btn_rect.height - txt.get_height()) // 2
            panel.blit(txt, (tx, ty))
            y += 32

    def _draw_education_panel(self):
        """Draw the right-side education/pedagogy panel."""
        edu_x = SCREEN_WIDTH + PANEL_WIDTH
        edu = self.screen.subsurface((edu_x, 0, EDUCATION_WIDTH, WINDOW_HEIGHT))
        edu.fill(EDU_BG)

        y = 10
        for section_title, title_color, lines in EDUCATION_SECTIONS:
            if title_color is None:
                t = self.font_large.render(section_title, True, HIGHLIGHT_COLOR)
                edu.blit(t, (10, y))
                y += 22
            else:
                t = self.font.render(f"* {section_title}", True, title_color)
                edu.blit(t, (10, y))
                y += 16
            for line in lines:
                if line == "":
                    y += 3
                    continue
                t = self.font_tiny.render(line, True, TEXT_COLOR)
                edu.blit(t, (14, y))
                y += 12

        pygame.draw.line(edu, (60, 60, 80), (10, y), (EDUCATION_WIDTH - 10, y))

    def _draw_tooltips(self):
        """Draw tooltip if mouse hovers over a label."""
        mouse_pos = pygame.mouse.get_pos()
        for rect, text in self._hover_rects:
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
