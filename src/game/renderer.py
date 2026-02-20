"""Pygame renderer for the multi-bird Flappy Bird game."""
import numpy as np
import pygame
from src.game.engine import (
    FlappyBirdEngine, Bird,
    SCREEN_WIDTH, SCREEN_HEIGHT, GROUND_Y,
    PLAYER_WIDTH, PLAYER_HEIGHT, PIPE_WIDTH, PIPE_HEIGHT, PIPE_GAP,
    PIPE_VEL_X,
)
from src.game.ui import Slider

PANEL_WIDTH = 280
EDUCATION_WIDTH = 380
WINDOW_WIDTH = SCREEN_WIDTH + PANEL_WIDTH + EDUCATION_WIDTH
WINDOW_HEIGHT = 700

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
    ("CERVEAU DE L'OISEAU", None, []),
    ("Comment il apprend", SECTION_COLOR, [
        "1. EXPLORE: strategie guidee (PD controller)",
        "2. Stocke chaque experience en memoire",
        "3. S'entraine 4x/frame sur des mini-lots",
        "4. Epsilon decroit: explore -> apprend",
        "",
    ]),
    ("Ce que l'oiseau voit (5 inputs)", SECTION_COLOR, [
        "Ecart au centre porte 1 + Vitesse",
        "Distance porte 1",
        "Ecart au centre porte 2 + Distance",
        "Sortie: ne rien faire OU sauter",
        "",
    ]),
    ("Conseils", SECTION_COLOR, [
        "Vitesse x32-x64 pour accelerer",
        "[e/2] = forcer exploitation",
        "Evolution ON = copie du champion",
    ]),
]

# Neural network visualization config
NN_LAYER_LABELS = [
    ["dy1", "vel", "d1", "dy2", "d2"],  # input (5D relative)
    None,                                # hidden1 (sampled)
    None,                                # hidden2 (sampled)
    ["noop", "FLAP"],                    # output
]
NN_SAMPLE_NODES = 8  # how many nodes to show per hidden layer


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
        self._draw_game(engine, bird_configs)
        self._draw_panel(engine, paused, bird_configs, buttons)
        self._draw_education_panel(bird_configs)
        self._draw_tooltips()
        pygame.display.flip()

    def _draw_game(self, engine: FlappyBirdEngine, bird_configs: dict = None):
        """Draw game area: sky, pipes, birds, trails, ghost, ground."""
        # Fill area below game with ground color (window taller than game)
        if WINDOW_HEIGHT > SCREEN_HEIGHT:
            extra = pygame.Rect(0, SCREEN_HEIGHT, SCREEN_WIDTH, WINDOW_HEIGHT - SCREEN_HEIGHT)
            pygame.draw.rect(self.screen, GROUND_COLOR, extra)
        game_surface = self.screen.subsurface((0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))
        game_surface.fill(SKY_COLOR)

        # Ghost trail from all-time best run
        ghost_trail = (bird_configs or {}).get("_ghost_trail", [])
        best_ever = (bird_configs or {}).get("_best_ever", 0)
        if ghost_trail and engine.frame < len(ghost_trail):
            gy = int(ghost_trail[engine.frame])
            ghost_surf = pygame.Surface((PLAYER_WIDTH, PLAYER_HEIGHT), pygame.SRCALPHA)
            pygame.draw.ellipse(ghost_surf, (255, 255, 255, 60), (0, 0, PLAYER_WIDTH, PLAYER_HEIGHT))
            gx = int(engine.birds[0].x if engine.birds else 57)
            game_surface.blit(ghost_surf, (gx, gy))
            gl = self.font_tiny.render(f"RECORD: {best_ever}", True, (255, 255, 255))
            game_surface.blit(gl, (gx - 2, gy - 10))

        # Bird trails (fading dots behind each alive bird)
        trail_surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        for bird in engine.birds:
            if not bird.alive or not bird.trail:
                continue
            n = len(bird.trail)
            for i, ty in enumerate(bird.trail):
                age = n - 1 - i  # 0 = newest
                alpha = max(20, 180 - age * 4)
                tx = int(bird.x - (age * abs(PIPE_VEL_X)))
                if tx < 0:
                    continue
                r, g, b = bird.color
                pygame.draw.circle(trail_surf, (r, g, b, alpha), (tx, int(ty) + PLAYER_HEIGHT // 2), 2)
        game_surface.blit(trail_surf, (0, 0))

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

            # Line 2: score + best + pipes + generation + status
            gen = config.get("generation", 0)
            gen_str = f" G:{gen}" if gen > 0 else ""
            score_txt = self.font_small.render(
                f" S:{bird.score} B:{best} P:{total_pipes}{gen_str} {'alive' if bird.alive else 'dead'}",
                True, status_color,
            )
            panel.blit(score_txt, (10, y))
            # Parent color indicator
            parent_color = config.get("parent_color")
            if parent_color and gen > 0:
                px = PANEL_WIDTH - 20
                pygame.draw.circle(panel, parent_color, (px, y + 6), 4)
                pygame.draw.circle(panel, (200, 200, 200), (px, y + 6), 4, 1)
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

    def _draw_education_panel(self, bird_configs: dict = None):
        """Draw the right-side panel: neural net viz + score graph + tips."""
        edu_x = SCREEN_WIDTH + PANEL_WIDTH
        edu = self.screen.subsurface((edu_x, 0, EDUCATION_WIDTH, WINDOW_HEIGHT))
        edu.fill(EDU_BG)
        bird_configs = bird_configs or {}

        y = 6
        # Title
        for section_title, title_color, lines in EDUCATION_SECTIONS:
            if title_color is None:
                t = self.font_large.render(section_title, True, HIGHLIGHT_COLOR)
                edu.blit(t, (10, y))
                y += 20
            else:
                t = self.font.render(f"* {section_title}", True, title_color)
                edu.blit(t, (10, y))
                y += 14
            for line in lines:
                if line == "":
                    y += 2
                    continue
                t = self.font_tiny.render(line, True, TEXT_COLOR)
                edu.blit(t, (14, y))
                y += 11

        # --- Neural Network Visualization ---
        activations = None
        best_bird_id = None
        best_score = -1
        for k, v in bird_configs.items():
            if isinstance(k, str) and k.startswith("_"):
                continue
            if isinstance(v, dict) and v.get("activations"):
                sc = v.get("best_score", 0)
                if sc > best_score:
                    best_score = sc
                    activations = v["activations"]
                    best_bird_id = k

        if activations and len(activations) >= 3:
            self._draw_neural_net(edu, 10, y, EDUCATION_WIDTH - 20, 200, activations)
            y += 205
        else:
            # Placeholder
            t = self.font_small.render("(en attente du reseau...)", True, DIM_COLOR)
            edu.blit(t, (14, y + 10))
            y += 35

        pygame.draw.line(edu, (60, 60, 80), (10, y), (EDUCATION_WIDTH - 10, y))
        y += 6

        # --- Score History Graph ---
        round_scores = bird_configs.get("_round_scores", [])
        self._draw_score_graph(edu, 10, y, EDUCATION_WIDTH - 20, 120, round_scores)
        y += 130

        pygame.draw.line(edu, (60, 60, 80), (10, y), (EDUCATION_WIDTH - 10, y))
        y += 6

        # --- Observation en direct (best bird) ---
        self._draw_obs_live(edu, 10, y, EDUCATION_WIDTH - 20, bird_configs)
        y += 110

        pygame.draw.line(edu, (60, 60, 80), (10, y), (EDUCATION_WIDTH - 10, y))
        y += 6

        # --- RL stats ---
        best_ever = bird_configs.get("_best_ever", 0)
        t = self.font.render("Statistiques RL", True, SECTION_COLOR)
        edu.blit(t, (10, y))
        y += 16
        t = self.font_small.render(f"Record absolu: {best_ever}", True, HIGHLIGHT_COLOR)
        edu.blit(t, (14, y))
        y += 14
        # Show epsilon + training info for each bird
        for k, v in bird_configs.items():
            if isinstance(k, str) and k.startswith("_"):
                continue
            if not isinstance(v, dict):
                continue
            eps = v.get("epsilon", 0)
            gen = v.get("generation", 0)
            algo = v.get("algo", "?")
            pct_exploit = max(0, 100 - int(eps * 100))
            bar_w = int(pct_exploit * 1.2)
            t = self.font_tiny.render(f"{algo} e={eps:.3f} [{pct_exploit}% apprend]", True, DIM_COLOR)
            edu.blit(t, (14, y))
            # Mini progress bar
            bar_x = EDUCATION_WIDTH - 140
            bar_rect = pygame.Rect(bar_x, y + 2, 120, 8)
            pygame.draw.rect(edu, (40, 40, 55), bar_rect, border_radius=2)
            fill_rect = pygame.Rect(bar_x, y + 2, bar_w, 8)
            color = EXPLOIT_COLOR if pct_exploit > 50 else EXPLORE_COLOR
            pygame.draw.rect(edu, color, fill_rect, border_radius=2)
            y += 13

    def _draw_obs_live(self, surface, x, y, w, bird_configs):
        """Draw live observation values as horizontal bars."""
        t = self.font.render("Observation en direct", True, SECTION_COLOR)
        surface.blit(t, (x + 5, y + 2))
        y += 18

        # Find best alive bird's activations (input layer = obs)
        obs = None
        for k, v in bird_configs.items():
            if isinstance(k, str) and k.startswith("_"):
                continue
            if isinstance(v, dict) and v.get("activations"):
                act = v["activations"]
                if act and len(act) > 0:
                    obs = np.asarray(act[0])
                    break
        if obs is None:
            t = self.font_tiny.render("(en attente...)", True, DIM_COLOR)
            surface.blit(t, (x + 10, y))
            return

        labels = ["Delta Y1", "Vitesse", "Dist P1", "Delta Y2", "Dist P2"]
        bar_w = w - 100
        for i, val in enumerate(obs[:5]):
            lbl = labels[i] if i < len(labels) else f"obs[{i}]"
            val = float(val)
            # Label
            lt = self.font_tiny.render(f"{lbl}", True, DIM_COLOR)
            surface.blit(lt, (x + 5, y))
            # Bar background
            bx = x + 65
            bar_rect = pygame.Rect(bx, y + 1, bar_w, 9)
            pygame.draw.rect(surface, (30, 33, 45), bar_rect, border_radius=2)
            # Fill (clamp to [0, 1] for display)
            fill_w = int(max(0, min(1, val)) * bar_w)
            if fill_w > 0:
                # Color gradient: blue (low) -> green (high)
                g = int(val * 200)
                b = int((1 - val) * 200)
                fill = pygame.Rect(bx, y + 1, fill_w, 9)
                pygame.draw.rect(surface, (40, min(255, 80 + g), min(255, 80 + b)), fill, border_radius=2)
            # Value text
            vt = self.font_tiny.render(f"{val:.2f}", True, TEXT_COLOR)
            surface.blit(vt, (bx + bar_w + 4, y))
            y += 11

    def _draw_neural_net(self, surface, x, y, w, h, activations):
        """Draw a live neural network diagram with colored activations."""
        # activations: [input(8), hidden1(128), hidden2(128), output(2)]
        n_layers = len(activations)
        if n_layers < 2:
            return

        # Determine nodes per layer for display
        display_nodes = []
        for i, act in enumerate(activations):
            act = np.asarray(act)
            if len(act) <= NN_SAMPLE_NODES:
                display_nodes.append(act)
            else:
                # Sample evenly spaced nodes
                indices = np.linspace(0, len(act) - 1, NN_SAMPLE_NODES, dtype=int)
                display_nodes.append(act[indices])

        # Layout
        layer_x_positions = []
        margin_x = 30
        usable_w = w - 2 * margin_x
        for i in range(n_layers):
            lx = x + margin_x + int(i * usable_w / max(1, n_layers - 1))
            layer_x_positions.append(lx)

        margin_y = 25
        usable_h = h - 2 * margin_y

        # Draw label
        t = self.font.render("Reseau de neurones (live)", True, SECTION_COLOR)
        surface.blit(t, (x + 5, y + 2))

        # Draw connections first (behind nodes)
        for li in range(n_layers - 1):
            n1 = len(display_nodes[li])
            n2 = len(display_nodes[li + 1])
            x1 = layer_x_positions[li]
            x2 = layer_x_positions[li + 1]
            for i in range(n1):
                y1 = y + margin_y + int(i * usable_h / max(1, n1 - 1))
                val1 = float(display_nodes[li][i])
                for j in range(n2):
                    y2 = y + margin_y + int(j * usable_h / max(1, n2 - 1))
                    # Connection brightness based on both activations
                    val2 = float(display_nodes[li + 1][j])
                    strength = min(1.0, (abs(val1) + abs(val2)) / 4.0)
                    alpha = int(20 + strength * 60)
                    color = (60 + int(strength * 80), 60 + int(strength * 80), 80 + int(strength * 100))
                    pygame.draw.line(surface, color, (x1, y1), (x2, y2), 1)

        # Draw nodes
        labels = NN_LAYER_LABELS
        for li, nodes in enumerate(display_nodes):
            n = len(nodes)
            lx = layer_x_positions[li]
            for i, val in enumerate(nodes):
                ny = y + margin_y + int(i * usable_h / max(1, n - 1))
                val = float(val)
                # Color: blue (negative/zero) -> green (small positive) -> red (high positive)
                intensity = min(1.0, abs(val) / 2.0)
                if val > 0:
                    color = (int(50 + 200 * intensity), int(200 - 100 * intensity), 50)
                else:
                    color = (50, int(80 + 100 * intensity), int(50 + 200 * intensity))
                radius = 5 + int(intensity * 4)
                pygame.draw.circle(surface, color, (lx, ny), radius)
                pygame.draw.circle(surface, (180, 180, 200), (lx, ny), radius, 1)

                # Labels for input/output layers
                if li < len(labels) and labels[li] and i < len(labels[li]):
                    lbl = self.font_tiny.render(labels[li][i], True, TEXT_COLOR)
                    if li == 0:
                        surface.blit(lbl, (lx - lbl.get_width() - 8, ny - 5))
                    else:
                        surface.blit(lbl, (lx + 10, ny - 5))

        # Layer labels
        layer_names = ["Entree", "Cache 1", "Cache 2", "Sortie"]
        for i, lx in enumerate(layer_x_positions):
            if i < len(layer_names):
                lt = self.font_tiny.render(layer_names[i], True, DIM_COLOR)
                surface.blit(lt, (lx - lt.get_width() // 2, y + h - 10))

    def _draw_score_graph(self, surface, x, y, w, h, scores):
        """Draw a line chart of score history."""
        # Title
        t = self.font.render("Progression des scores", True, SECTION_COLOR)
        surface.blit(t, (x + 5, y + 2))

        graph_y = y + 18
        graph_h = h - 24
        graph_w = w - 10
        graph_x = x + 5

        # Background
        bg = pygame.Rect(graph_x, graph_y, graph_w, graph_h)
        pygame.draw.rect(surface, (15, 18, 28), bg, border_radius=3)
        pygame.draw.rect(surface, (50, 50, 70), bg, 1, border_radius=3)

        if not scores:
            t = self.font_tiny.render("(en attente de donnees...)", True, DIM_COLOR)
            surface.blit(t, (graph_x + 10, graph_y + graph_h // 2 - 5))
            return

        # Grid lines
        max_score = max(max(scores), 1)
        for i in range(5):
            gy = graph_y + int(i * graph_h / 4)
            pygame.draw.line(surface, (35, 38, 50), (graph_x, gy), (graph_x + graph_w, gy), 1)

        # Y-axis labels
        for i in range(5):
            val = max_score * (4 - i) / 4
            gy = graph_y + int(i * graph_h / 4)
            lt = self.font_tiny.render(f"{val:.0f}", True, DIM_COLOR)
            surface.blit(lt, (graph_x + 2, gy - 4))

        # Plot line
        n = len(scores)
        if n < 2:
            return
        points = []
        for i, s in enumerate(scores):
            px = graph_x + int(i * graph_w / (n - 1))
            py = graph_y + graph_h - int(s * graph_h / max_score)
            py = max(graph_y, min(graph_y + graph_h, py))
            points.append((px, py))

        # Gradient fill under the line
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            bottom = graph_y + graph_h
            fill_color = (30, 100, 60, 80)
            fill_surf = pygame.Surface((max(1, x2 - x1), bottom - min(y1, y2)), pygame.SRCALPHA)
            fill_surf.fill(fill_color)
            surface.blit(fill_surf, (x1, min(y1, y2)))

        # Line
        if len(points) >= 2:
            pygame.draw.lines(surface, HIGHLIGHT_COLOR, False, points, 2)

        # Current score dot
        pygame.draw.circle(surface, (255, 255, 100), points[-1], 4)

        # Max score label
        best = max(scores)
        lt = self.font_small.render(f"Best: {best}", True, HIGHLIGHT_COLOR)
        surface.blit(lt, (graph_x + graph_w - lt.get_width() - 4, graph_y + 2))

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
